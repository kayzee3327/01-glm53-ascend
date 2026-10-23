"""Real-NPU GLM KDA checkpoint, resumed-prefix and graph/decode regression tests,
run against the Ascend C ``kda_varlen_recurrent`` op.

A copy of sglang's ``test/manual/ascend/test_glm53_kda_prefix.py`` with one
change: ``glm_kda_varlen_recurrent_npu`` is replaced by the dispatcher in
``kda_ascendc_dispatch``, which calls the Ascend C op and falls back to Triton
only on the op's own rejections. Every assertion and tolerance below is
upstream's. Keep in sync.

Run after sourcing the native CANN environment. This standalone test loads no
model and uses only --device (default NPU 0). JSON is flushed after every case,
including failed cases, so an interrupted run cannot look like a complete pass.
"""

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

# Match native serving/test initialization before any Triton compilation.
# Importing this backend after the first NPU JIT changes autotuner decoration.
import sglang.srt.layers.quantization  # noqa: F401
import kda_ascendc_dispatch

# Rebind before the backend is imported, so its in-call import of the
# recurrence resolves to the dispatcher.
kda_ascendc_dispatch.patch()
from kda_ascendc_dispatch import (  # noqa: E402
    glm_kda_varlen_recurrent_dispatch as glm_kda_varlen_recurrent_npu,
)

from sglang.srt.hardware_backend.npu.attention.ascend_kda_backend import (  # noqa: E402
    AscendKDAAttnBackend,
)

DEVICE = "npu:0"


def clone_state(state):
    """Preserve native [V,K] backing; torch_npu.clone makes views contiguous."""
    if state.stride(-2) == 1:
        return state.transpose(-1, -2).clone().transpose(-1, -2)
    return state.clone()


def offsets(lengths):
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return torch.tensor(result, dtype=torch.int32, device=DEVICE)


def make_inputs(total, heads):
    result = {
        name: (torch.randn(1, total, heads, 128, device=DEVICE) * 0.2).bfloat16()
        for name in ("q", "k", "v", "a")
    }
    result["b"] = torch.randn(1, total, heads, device=DEVICE).bfloat16()
    return result


def make_constants(heads):
    return dict(
        A_log=torch.randn(heads, device=DEVICE) * 0.1,
        dt_bias=torch.randn(heads * 128, device=DEVICE) * 0.1,
        lower_bound=-5.0,
    )


def cpu_reference(initial, inputs, constants):
    """Independent FP32 torch recurrence in logical [H,K,V] layout."""
    h = initial.detach().cpu().float().clone()
    tensors = {name: x.detach().cpu().float()[0] for name, x in inputs.items()}
    alog = constants["A_log"].detach().cpu().reshape(-1, 1)
    bias = constants["dt_bias"].detach().cpu().reshape(-1, 128)
    outputs, snapshots = [], [h.clone()]
    for t in range(tensors["q"].shape[0]):
        q, k, v, a, b = (tensors[name][t] for name in ("q", "k", "v", "a", "b"))
        q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) / 128**0.5
        k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        g = constants["lower_bound"] * torch.sigmoid(alog.exp() * (a + bias))
        h = h * g.exp().unsqueeze(-1)
        update = (v - (h * k.unsqueeze(-1)).sum(-2)) * b.sigmoid().unsqueeze(-1)
        h = h + k.unsqueeze(-1) * update.unsqueeze(-2)
        snapshots.append(h.clone())
        outputs.append((h * q.unsqueeze(-1)).sum(-2))
    return torch.stack(outputs).unsqueeze(0).bfloat16(), snapshots


def check_case(lengths, cuts, prepared, heads=4, transpose=True, slot_values=None,
               destination_values=None, reference=False, index_dtype=torch.int64):
    torch.manual_seed(194195 + sum(lengths))
    os.environ["SGLANG_GLM53_KDA_PREFILL_PREPARE"] = str(int(prepared))
    batch = len(lengths)
    initial = torch.randn(12, heads, 128, 128, device=DEVICE) * 0.1
    if transpose:
        initial = initial.transpose(-1, -2)
    slot_values = slot_values if slot_values is not None else [3, 1, 5][:batch]
    destination_values = destination_values if destination_values is not None else [9, 7, 0][:batch]
    slots = torch.tensor(slot_values, dtype=index_dtype, device=DEVICE)
    destinations = torch.tensor(destination_values, dtype=index_dtype, device=DEVICE)
    starts = offsets(lengths)
    inputs, constants = make_inputs(sum(lengths), heads), make_constants(heads)

    def run(state, tensors, indices, starts, **kwargs):
        return glm_kda_varlen_recurrent_npu(
            **constants, **tensors, initial_state_source=state,
            initial_state_indices=indices, cu_seqlens=starts, prefill=True, **kwargs,
        )

    baseline = clone_state(initial)
    expected = run(baseline, inputs, slots, starts)
    tracked = clone_state(initial)
    actual = run(
        tracked, inputs, slots, starts, track_state_indices=destinations,
        track_state_lens=torch.tensor(cuts, dtype=torch.int32, device=DEVICE),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(tracked[slots.long()], baseline[slots.long()], rtol=0, atol=0)
    touched = set(slot_values + destination_values) - {0}
    for unused in set(range(initial.shape[0])) - touched:
        torch.testing.assert_close(tracked[unused], initial[unused], rtol=0, atol=0)

    errors = []
    begin = 0
    for row, (length, cut, slot, destination) in enumerate(
        zip(lengths, cuts, slot_values, destination_values)
    ):
        if destination == 0:
            begin += length
            continue
        # Packed or <128-token invocations use the generic recurrence. Use the
        # same path for split checks whenever possible and a numerical tolerance
        # when a short suffix necessarily falls back from prepared to generic.
        os.environ["SGLANG_GLM53_KDA_PREFILL_PREPARE"] = str(int(prepared and batch == 1))
        prefix_state = clone_state(initial)
        if cut > 0:
            prefix_inputs = {name: t[:, begin:begin + cut] for name, t in inputs.items()}
            run(prefix_state, prefix_inputs, slots[row:row + 1], offsets([cut]))
        if slot > 0:
            reference_state = prefix_state[slot]
        elif cut == 0:
            reference_state = torch.zeros_like(prefix_state[0])
        else:
            # Padding source slot zero is never updated by the kernel. Its
            # consumed state therefore needs an independent reference.
            _, prefix_snapshots = cpu_reference(
                torch.zeros_like(initial[0]), prefix_inputs, constants
            )
            reference_state = prefix_snapshots[-1].to(DEVICE)
        checkpoint = tracked[destination]
        torch.testing.assert_close(checkpoint, reference_state, rtol=2e-4, atol=2e-5)
        errors.append((checkpoint - reference_state).abs().max().item())
        if cut < length and slot > 0:
            resumed = clone_state(initial)
            resumed[slot] = checkpoint
            suffix_inputs = {name: t[:, begin + cut:begin + length] for name, t in inputs.items()}
            suffix = run(resumed, suffix_inputs, slots[row:row + 1], offsets([length - cut]))
            torch.testing.assert_close(suffix, expected[:, begin + cut:begin + length], rtol=0.008, atol=0.001)
            torch.testing.assert_close(resumed[slot], baseline[slot], rtol=2e-4, atol=2e-5)
        if reference:
            seq_inputs = {name: t[:, begin:begin + length] for name, t in inputs.items()}
            start_state = initial[slot] if slot > 0 else torch.zeros_like(initial[0])
            ref_output, ref_states = cpu_reference(start_state, seq_inputs, constants)
            torch.testing.assert_close(actual[:, begin:begin + length].cpu(), ref_output, rtol=0.01, atol=0.001)
            torch.testing.assert_close(checkpoint.cpu(), ref_states[cut], rtol=2e-4, atol=2e-5)
            if slot > 0:
                torch.testing.assert_close(tracked[slot].cpu(), ref_states[-1], rtol=2e-4, atol=2e-5)
        begin += length
    torch.npu.synchronize()
    return dict(lengths=lengths, cuts=cuts, prepared=prepared, heads=heads,
                transpose=transpose, slots=slot_values, destinations=destination_values,
                independent_cpu_reference=reference, index_dtype=str(index_dtype),
                state_stride=list(tracked.stride()),
                checkpoint_max_abs_error=max(errors, default=0))


def check_graph(prepared=False, decode_batch=None, steps=1, synchronize_inputs=True):
    torch.manual_seed(20260915 + steps)
    os.environ["SGLANG_GLM53_KDA_PREFILL_PREPARE"] = str(int(prepared))
    decode = decode_batch is not None
    lengths = [steps] * decode_batch if decode else ([257] if prepared else [129, 65, 63])
    batch = len(lengths)
    size = max(12, 2 * batch + 3)
    initial = (torch.randn(size, 4, 128, 128, device=DEVICE) * 0.1).transpose(-1, -2)
    work, reference = clone_state(initial), clone_state(initial)
    inputs, constants = make_inputs(sum(lengths), 4), make_constants(4)
    slots = torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE)
    destinations = slots + batch + 1
    cuts = torch.tensor([min(x, 64) for x in lengths], dtype=torch.int32, device=DEVICE)
    starts = offsets(lengths)
    intermediate = (torch.empty(batch, steps, 4, 128, 128, device=DEVICE).transpose(-1, -2)
                    if decode and steps > 1 else None)
    # torch_npu.empty_like does not preserve the strided native [V,K] view for
    # this tensor. Allocate and transpose exactly as the production MTP pool.
    eager_intermediate = (
        torch.empty(batch, steps, 4, 128, 128, device=DEVICE).transpose(-1, -2)
        if intermediate is not None else None
    )
    layout = dict(
        state_stride=list(work.stride()),
        intermediate_stride=list(intermediate.stride()) if intermediate is not None else None,
        eager_intermediate_stride=list(eager_intermediate.stride()) if eager_intermediate is not None else None,
        empty_like_stride=list(torch.empty_like(intermediate).stride()) if intermediate is not None else None,
    )
    print(json.dumps(dict(kind="graph_layout", batch=batch, steps=steps, **layout)), flush=True)

    def run(state, middle):
        return glm_kda_varlen_recurrent_npu(
            **constants, **inputs, initial_state_source=state,
            initial_state_indices=slots, cu_seqlens=starts, prefill=not decode,
            intermediate_state=middle,
            track_state_indices=None if decode else destinations,
            track_state_lens=None if decode else cuts,
        )

    for _ in range(3):
        work.copy_(initial)
        run(work, intermediate)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured_output = run(work, intermediate)
    padding_max_abs_error = 0.0
    for replay in range(8):
        # Change values behind captured pointers, including padded source slot
        # zero, non-contiguous request slots and packed sequence boundaries.
        active = [(row + replay) % (batch + 1) for row in range(batch)]
        slots.copy_(torch.tensor(active, dtype=slots.dtype, device=DEVICE))
        if not decode:
            dest = [batch + 2 + ((row + replay) % batch) for row in range(batch)]
            if replay % 3 == 0:
                dest[-1] = 0
            destinations.copy_(torch.tensor(dest, dtype=destinations.dtype, device=DEVICE))
            lens = lengths if prepared or replay % 2 == 0 else [65, 129, 63]
            starts.copy_(offsets(lens))
            boundaries = [0, 64, 128, 257] if prepared else [0, 32, 63, 64]
            cuts.copy_(torch.tensor([min(lens[row], boundaries[(row + replay) % 4])
                                    if dest[row] else -12345 for row in range(batch)],
                                   dtype=cuts.dtype, device=DEVICE))
        for tensor in inputs.values():
            tensor.copy_(torch.randn_like(tensor) * 0.2)
        work.copy_(initial)
        reference.copy_(initial)
        if intermediate is not None:
            intermediate.fill_(17.0)
            eager_intermediate.fill_(17.0)
        expected = run(reference, eager_intermediate)
        if synchronize_inputs:
            # NPU graphs may replay on the capture stream. Complete default
            # stream writes and the eager oracle before comparing the replay.
            torch.npu.synchronize()
        # Freeze the oracle outside graph/device allocator memory before replay.
        expected_cpu = expected.cpu()
        reference_cpu = reference.cpu()
        intermediate_cpu = eager_intermediate.cpu() if eager_intermediate is not None else None
        graph.replay()
        torch.npu.synchronize()
        actual_cpu = captured_output.cpu()
        work_cpu = work.cpu()
        print(json.dumps(dict(kind="graph_replay_errors", batch=batch, steps=steps,
                              replay=replay, active=active,
                              output_max_abs_error=(actual_cpu - expected_cpu).abs().max().item(),
                              state_max_abs_error=(work_cpu - reference_cpu).abs().max().item())), flush=True)
        torch.testing.assert_close(work_cpu, reference_cpu, rtol=0, atol=0)
        if decode:
            # Slot zero is a graph-padding row, whose output is discarded by
            # the scheduler. Check every real token bitwise and every state
            # slot below, including untouched padding/intermediate sentinels.
            valid = [row * steps + t for row in range(batch) if active[row] > 0
                     for t in range(steps)]
            padding = [row * steps + t for row in range(batch) if active[row] == 0
                       for t in range(steps)]
            torch.testing.assert_close(actual_cpu[:, valid], expected_cpu[:, valid], rtol=0, atol=0)
            if padding:
                error = (actual_cpu[:, padding] - expected_cpu[:, padding]).abs().max().item()
                padding_max_abs_error = max(padding_max_abs_error, error)
        else:
            torch.testing.assert_close(actual_cpu, expected_cpu, rtol=0, atol=0)
        if intermediate is not None:
            torch.testing.assert_close(intermediate.cpu(), intermediate_cpu, rtol=0, atol=0)
            torch.testing.assert_close(work, initial, rtol=0, atol=0)
    return dict(prepared=prepared, decode=decode, batch=batch, steps=steps, replays=8,
                synchronize_inputs=synchronize_inputs,
                compared_outputs="active tokens" if decode else "all tokens",
                padding_max_abs_error=padding_max_abs_error, **layout)


def check_validation():
    inputs, constants = make_inputs(4, 4), make_constants(4)
    state = torch.zeros(8, 4, 128, 128, device=DEVICE)
    slots = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    track = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    lens = torch.tensor([2], dtype=torch.int32, device=DEVICE)
    base = dict(**inputs, **constants, initial_state_source=state,
                initial_state_indices=slots, cu_seqlens=offsets([4]), prefill=True,
                track_state_indices=track, track_state_lens=lens)
    cases = [
        dict(track_state_indices=None), dict(track_state_lens=None),
        dict(prefill=False), dict(track_state_indices=track.float()),
        dict(track_state_lens=lens.reshape(1, 1)),
        dict(track_state_lens=torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)),
        dict(track_state_indices=track.cpu()),
        dict(initial_state_source=state.bfloat16()),
        dict(intermediate_state=torch.empty(1, 4, 4, 128, 128, device=DEVICE)),
    ]
    for changes in cases:
        try:
            glm_kda_varlen_recurrent_npu(**(base | changes))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid tracking metadata accepted: {tuple(changes)}")
    torch.testing.assert_close(state, torch.zeros_like(state), rtol=0, atol=0)
    return dict(rejected_metadata_cases=len(cases))


def check_backend(prepared, conv_window):
    """Exercise real Ascend causal-conv + KDA checkpoint and resumed prefix."""
    torch.manual_seed(20260915 + conv_window)
    os.environ["SGLANG_GLM53_KDA_PREFILL_PREPARE"] = str(int(prepared))
    total, cut, source, destination = 513, 256, 3, 7
    channels = 3 * 4 * 128
    conv_seed = (torch.randn(10, conv_window, channels, device=DEVICE) * 0.1).bfloat16()
    temporal_seed = torch.randn(10, 4, 128, 128, device=DEVICE) * 0.1
    mixed = (torch.randn(total, channels, device=DEVICE) * 0.2).bfloat16()
    a = (torch.randn(1, total, 512, device=DEVICE) * 0.2).bfloat16()
    b = torch.randn(1, total, 4, device=DEVICE).bfloat16()
    layer = SimpleNamespace(
        layer_id=0, conv_weights=torch.randn(channels, 4, device=DEVICE) * 0.1,
        bias=None, q_dim=512, k_dim=512, v_dim=512, head_q_dim=128,
        head_k_dim=128, head_v_dim=128, num_v_heads=4, **make_constants(4),
    )

    def new_cache():
        return SimpleNamespace(conv=[conv_seed.clone()], temporal=temporal_seed.clone())

    def run(cache, begin, end, tracking):
        backend = object.__new__(AscendKDAAttnBackend)
        backend.glm_bounded_recurrence = True
        backend.req_to_token_pool = SimpleNamespace(mamba2_layer_cache=lambda _: cache)
        backend.forward_metadata = SimpleNamespace(
            query_start_loc=offsets([end - begin]),
            mamba_cache_indices=torch.tensor([source], dtype=torch.int32, device=DEVICE),
            has_mamba_track_mask=tracking,
            track_conv_indices=torch.arange(cut - conv_window, cut, device=DEVICE).unsqueeze(0),
            conv_states_mask_indices=torch.tensor([destination], device=DEVICE),
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_target_verify=lambda: False),
            extend_prefix_lens=torch.tensor([512 + begin], dtype=torch.int32, device=DEVICE),
            mamba_track_mask=torch.tensor([tracking], device=DEVICE),
            mamba_track_indices=torch.tensor([destination], dtype=torch.int64, device=DEVICE),
            mamba_track_aligned_lens=lambda: torch.tensor([cut], dtype=torch.int32, device=DEVICE),
        )
        return backend.forward_extend(layer, batch, mixed[begin:end], a[:, begin:end], b[:, begin:end])

    baseline, tracked, prefix = new_cache(), new_cache(), new_cache()
    expected = run(baseline, 0, total, False)
    actual = run(tracked, 0, total, True)
    run(prefix, 0, cut, False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(tracked.temporal[source], baseline.temporal[source], rtol=0, atol=0)
    torch.testing.assert_close(tracked.temporal[destination], prefix.temporal[source], rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(tracked.conv[0][destination], mixed[cut - conv_window:cut], rtol=0, atol=0)
    torch.testing.assert_close(tracked.conv[0][destination, -3:], prefix.conv[0][source, -3:], rtol=0, atol=0)
    resumed = new_cache()
    resumed.temporal[source] = tracked.temporal[destination]
    resumed.conv[0][source] = tracked.conv[0][destination]
    suffix = run(resumed, cut, total, False)
    torch.testing.assert_close(suffix, expected[:, cut:], rtol=0.008, atol=0.001)
    torch.testing.assert_close(resumed.temporal[source], baseline.temporal[source], rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(resumed.conv[0][source, -3:], baseline.conv[0][source, -3:], rtol=0, atol=0)
    for slot in set(range(10)) - {source, destination}:
        torch.testing.assert_close(tracked.temporal[slot], temporal_seed[slot], rtol=0, atol=0)
        torch.testing.assert_close(tracked.conv[0][slot], conv_seed[slot], rtol=0, atol=0)
    torch.npu.synchronize()
    return dict(prepared=prepared, conv_window=conv_window, total=total, cut=cut,
                prefix_len=512, real_causal_conv=True,
                checkpoint_max_abs_error=(tracked.temporal[destination] - prefix.temporal[source]).abs().max().item())


def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--suite", choices=("all", "prefix", "graph", "validation", "backend"), default="all")
    args = parser.parse_args()
    DEVICE = f"npu:{args.device}"
    torch.npu.set_device(args.device)
    torch.set_num_threads(4)
    cases = []
    if args.suite in ("all", "prefix"):
        for prepared in (False, True):
            for lengths, cuts in (
                ([513], [256]), ([512], [512]), ([257], [128]), ([65], [0]),
                ([513, 769, 257], [256, 512, -777]),
                ([0, 129, 65], [0, 64, -777]),
            ):
                cases.append(("checkpoint", lambda l=lengths, c=cuts, p=prepared: check_case(l, c, p)))
        cases.extend([
            ("checkpoint_heads8", lambda: check_case([257, 513], [64, 256], False, heads=8)),
            ("checkpoint_contiguous", lambda: check_case([129], [64], False, transpose=False, reference=True)),
            ("checkpoint_cpu", lambda: check_case([17], [7], False, reference=True, index_dtype=torch.int32)),
            ("checkpoint_cpu_prepared", lambda: check_case([129], [128], True, reference=True)),
            ("checkpoint_zero_source", lambda: check_case([17], [7], False, slot_values=[0], reference=True)),
            ("checkpoint_disabled", lambda: check_case([129], [-12345], True, destination_values=[0])),
        ])
    if args.suite in ("all", "graph"):
        cases.extend([
            ("prefix_graph_generic", lambda: check_graph()),
            ("prefix_graph_prepared", lambda: check_graph(prepared=True)),
        ])
        for batch in (1, 3, 16):
            for steps in (1, 4):
                cases.append(("decode_graph", lambda b=batch, s=steps: check_graph(decode_batch=b, steps=s)))
    if args.suite in ("all", "validation"):
        cases.append(("metadata_validation", check_validation))
    if args.suite in ("all", "backend"):
        for prepared in (False, True):
            for window in (3, 6):
                cases.append(("backend_checkpoint", lambda p=prepared, w=window: check_backend(p, w)))
    report = dict(passed=False, complete=False, expected_cases=len(cases), results=[],
                  torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
                  device=DEVICE, device_name=torch.npu.get_device_name(args.device),
                  test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, fn in cases:
        start = time.monotonic()
        row = dict(name=name, passed=False)
        try:
            row.update(fn(), passed=True)
        except Exception:
            row["traceback"] = traceback.format_exc()
        row["duration_seconds"] = time.monotonic() - start
        report["results"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    passed = all(row["passed"] for row in report["results"])
    # A run where the op was never reached would otherwise pass on Triton alone.
    coverage = kda_ascendc_dispatch.report()
    report["ascendc_coverage"] = coverage
    print(json.dumps(dict(kind="ascendc_coverage", **coverage)), flush=True)
    passed = passed and coverage["ascendc_calls"] > 0
    report.update(passed=passed, complete=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=passed, complete=True, cases=len(cases))), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
