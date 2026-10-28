"""Builds CANN 9.2.0's ``ChunkKdaFwd`` as a custom op package, from source.

Why this is needed at all: the installed toolkit is CANN 9.0.0, which does not
ship ``ChunkKdaFwd`` -- ``libopapi.so`` has no ``aclnnChunkKdaFwd`` symbol and
there is no kernel binary for it. The operator was added later, so the only way
to benchmark the official implementation on this machine is to compile the
9.2.0 sources against the 9.0.0 toolkit and install the result as a *custom* op
package, which loads alongside the built-in ops rather than replacing them.

That is a cross-version build, and it is the part most likely to break. Two
things make it plausible rather than hopeless:

  * the operator declares ``ascend910b`` support directly
    (``op_host/chunk_kda_fwd_def.cpp``: ``AICore().AddConfig("ascend910b",
    ...)``, alongside ``ascend910_93`` and ``ascend950``), so this is a
    supported configuration of the op and not an off-label target.
  * the repo's own toolkit-compatibility gate is already disabled upstream --
    ``cmake/config.cmake`` declares ``CHECK_COMPATIBLE`` and then immediately
    forces it ``OFF`` -- so a version mismatch against ``version.info`` does
    not by itself fail the build. ``--disable-check-compatible`` is passed
    anyway, so this does not depend on that line staying as it is.

What is *not* guaranteed is that the 9.2.0 sources only use APIs present in the
9.0.0 toolkit. If a future checkout starts depending on a 9.2-only header, this
script is where it surfaces, as a compile error naming the header.

One artifact comes out of it:

  * the custom op run package, installed under
    ``${ASCEND_OPP_PATH}/vendors/<vendor>_transformer/``, holding the host
    tiling, the kernel binary and the aclnn entry point.

The sibling bench also builds a ``cann_ops_transformer_<vendor>`` wheel, because
``recurrent_kda`` ships a ``torch_extension/`` wrapper that is only reachable
once the wheel installs it. ``chunk_kda_fwd`` has no such wrapper, so there is
nothing for a wheel to carry: ``cann_op.py`` binds the aclnn symbol itself.
Hence ``NEEDS_WHEEL`` below, which is the only behavioural difference between
the two scripts.

Usage, from anywhere:

    python build_cann_op.py              # build and install, skip if present
    python build_cann_op.py --force      # rebuild even if installed
    python build_cann_op.py --check      # report status, build nothing

The build takes a while (tens of minutes; it compiles kernel binaries) and is
serialized by a lock file shared with ``bench/kda_recurrent/build_cann_op.py``,
because the ops-transformer tree has one shared ``build/`` directory and a
concurrent build in the same checkout would corrupt it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

OP_NAME = "chunk_kda_fwd"
# One vendor package per operator, deliberately *not* shared with the sibling
# bench. The installer wipes same-named files under a vendor dir it is
# reinstalling into -- and every single-op build ships its own
# `libcust_opapi.so` -- so two ops under one vendor name cannot coexist: the
# second install silently removes the first operator. Distinct names let both
# sit in `opp/vendors/` at once, and the aclnn dlsym search walks every vendor
# dir, so each bench still finds its own op.
VENDOR = "glm53chunk"
SOC = "ascend910b"

# This op needs no Python wheel: it ships no `torch_extension/` wrapper, so a
# vendor wheel would expose no entry point for it and `cann_op.py` compiles its
# own aclnn bridge instead. The sibling bench sets this True, where the wheel is
# the only place the op's Python entry point exists.
NEEDS_WHEEL = False
WHEEL_PACKAGE = f"cann_ops_transformer_{VENDOR}"

REPO = Path(os.environ.get("OPS_TRANSFORMER_DIR", "/data/wangkaize/ops-transformer"))
# Shared with the sibling bench on purpose: the vendor packages are separate,
# but ops-transformer builds them through one `build/` tree.
# The name is historical and intentionally *not* per-vendor: the two benches
# must name the same file for the lock to serialize them. Rename in both or
# neither.
LOCK = REPO / ".build-lock-glm53bench"
LOG = Path(__file__).resolve().parent / "build_cann_op.log"


def ascend_home() -> Path:
    """The installed toolkit, from the environment CANN's set_env.sh exports."""
    for var in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(var)
        if value:
            return Path(value)
    raise SystemExit(
        "CANN environment is not set: no ASCEND_HOME_PATH. "
        "Source the toolkit's set_env.sh first."
    )


def opp_path() -> Path:
    value = os.environ.get("ASCEND_OPP_PATH")
    return Path(value) if value else ascend_home() / "opp"


def vendor_dir() -> Path:
    return opp_path() / "vendors" / f"{VENDOR}_transformer"


def installed() -> bool:
    """Whether every half of the install is present.

    The wheel counts only where the bench reaches the operator through its
    Python entry point; see ``NEEDS_WHEEL``.
    """
    root = vendor_dir()
    if not root.is_dir():
        return False
    if not any(root.rglob("libcust_opapi.so")):
        return False
    if not aclnn_symbols():
        return False
    return wheel_installed() if NEEDS_WHEEL else True


def aclnn_symbols() -> list[str]:
    """The ``aclnn*Kda*`` entry points the installed package exports.

    The real check that a build succeeded: the package directory can exist and
    hold a ``libcust_opapi.so`` that does not actually export this operator,
    which is what a build that silently skipped the op looks like from outside.
    """
    libs = list(vendor_dir().rglob("libcust_opapi.so"))
    if not libs:
        return []
    try:
        out = subprocess.run(
            ["nm", "-D", "--defined-only", str(libs[0])],
            capture_output=True, text=True, check=False,
        ).stdout
    except OSError:
        return []
    found = {
        line.split()[-1]
        for line in out.splitlines()
        if "aclnn" in line and "Kda" in line
    }
    return sorted(s for s in found if s.startswith("aclnn"))


def toolkit_version() -> str:
    info = ascend_home() / "compiler" / "version.info"
    if info.is_file():
        for line in info.read_text().splitlines():
            if line.startswith("Version="):
                return line.split("=", 1)[1].strip()
    return "unknown"


def repo_version() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_package() -> Path | None:
    """The newest custom op ``.run`` the build produced, if any.

    ``build/`` first because that is where ``--pkg`` actually leaves it:
    CPack's ``DIR_MOVE`` step moves the package to ``CPACK_PACKAGE_DIRECTORY``,
    which is the CMake binary dir, and ``collect_rpm_deb_package`` returns early
    for ``PACKAGE_TYPE=run`` so nothing is ever copied to ``build_out/``. The
    other two are kept because other build modes do write there.
    """
    candidates = [
        *(REPO / "build").glob("*.run"),
        *(REPO / "build_out").glob("*.run"),
        *(REPO / "output").glob("*.run"),
    ]
    runs = [p for p in candidates if VENDOR in p.name]
    if not runs:
        runs = [p for p in candidates if "custom" in p.name] or candidates
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def acquire_lock(timeout: float = 7200.0) -> None:
    """Serialize against another build in the same checkout.

    ops-transformer builds into one shared ``build/`` tree, so two concurrent
    builds out of this checkout -- a plausible thing to happen, since a
    neighbouring bench script builds a different operator from it -- would
    interleave and corrupt each other's objects.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode())
            os.close(fd)
            return
        except FileExistsError:
            holder = ""
            try:
                holder = LOCK.read_text().strip()
            except OSError:
                pass
            # A lock left behind by a dead process should not block forever.
            if holder.isdigit() and not Path(f"/proc/{holder}").exists():
                print(f"removing stale build lock from pid {holder}")
                LOCK.unlink(missing_ok=True)
                continue
            if time.monotonic() > deadline:
                raise SystemExit(f"timed out waiting for build lock {LOCK}")
            print(f"waiting for build lock held by pid {holder or '?'} ...")
            time.sleep(30)


def release_lock() -> None:
    LOCK.unlink(missing_ok=True)


def build(jobs: int) -> None:
    """``build.sh --pkg`` for this one operator, then install the run package."""
    if not REPO.is_dir():
        raise SystemExit(f"ops-transformer checkout not found at {REPO}")

    print(f"toolkit CANN {toolkit_version()}, ops-transformer @ {repo_version()}")
    print(f"building {OP_NAME} for {SOC} as vendor '{VENDOR}'")
    print(f"log: {LOG}")

    cmd = [
        "bash", "build.sh", "--pkg",
        f"--ops={OP_NAME}",
        f"--vendor_name={VENDOR}",
        f"--soc={SOC}",
        # Deliberately *no* `--opkernel`, despite its name suggesting it is what
        # builds the device kernel. It sets `ENABLE_OPKERNEL`, which `build.sh`
        # tests *before* the `--ops`/`--vendor_name` branch in its dispatch
        # chain, so it wins and runs `build_kernel` alone -- compiling the
        # kernel binaries and then exiting without ever invoking CPack. The
        # build reports success and produces no `.run` at all.
        #
        # Dropping it lands on the custom-op branch, which sets
        # `ENABLE_OPS_KERNEL=ON` itself and then calls `build_package`. So the
        # kernel binary is still built; it is also packaged.
        #
        # The sources are 9.2.0 and the toolkit is 9.0.0. The repo's gate is
        # already forced off in cmake/config.cmake, but say so explicitly so
        # this does not start failing if that changes.
        "--disable-check-compatible",
        f"-j{jobs}",
    ]
    print("  " + " ".join(cmd))

    with LOG.open("w") as log:
        proc = subprocess.run(
            cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True
        )
    if proc.returncode != 0:
        tail = "\n".join(LOG.read_text().splitlines()[-40:])
        raise SystemExit(
            f"build failed (exit {proc.returncode}). Last lines of {LOG}:\n{tail}"
        )

    package = run_package()
    if package is None:
        raise SystemExit(
            f"build reported success but produced no .run package; see {LOG}"
        )

    print(f"installing {package.name}")
    # No --install-path: the installer uses that value *as* the vendors parent,
    # so passing the toolkit root lands the package in `<cann>/vendors/` --
    # which nothing looks at -- while still reporting success. Left unset it
    # falls back to ASCEND_OPP_PATH and installs to `opp/vendors/`, where the
    # loader and OpBuilder search. The installer prompts when a vendor dir
    # already holds files, which would hang an unattended build; --quiet plus
    # the non-interactive stdin keeps it from waiting on a reply it will never
    # get.
    install = subprocess.run(
        ["bash", str(package), "--quiet"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
        env={**os.environ, "ASCEND_OPP_PATH": str(opp_path())},
    )
    if install.returncode != 0:
        raise SystemExit(
            f"installing {package} failed:\n{install.stdout}\n{install.stderr}"
        )

    if NEEDS_WHEEL:
        build_wheel(jobs)

    symbols = aclnn_symbols()
    if not symbols:
        raise SystemExit(
            f"install completed but {vendor_dir()} exports no aclnn*Kda* symbol; "
            f"see {LOG}"
        )
    print(f"installed to {vendor_dir()}")
    for symbol in symbols:
        print(f"  {symbol}")


def build_wheel(jobs: int) -> None:
    """Build and install ``cann_ops_transformer_<vendor>``: the Python binding.

    Only reached when ``NEEDS_WHEEL``. Kept here so the two benches' build
    scripts stay the same code: the wheel is quick (pure Python, no kernel
    compile) and the decision is the constant, not this function.
    """
    print(f"building {WHEEL_PACKAGE} wheel")
    with LOG.open("a") as log:
        proc = subprocess.run(
            ["bash", "build.sh", "--torch_extension",
             f"--ops={OP_NAME}", f"--vendor_name={VENDOR}", f"-j{jobs}"],
            cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True,
        )
    if proc.returncode != 0:
        raise SystemExit(f"wheel build failed; see {LOG}")

    wheels = sorted(
        [*(REPO / "build_out").glob(f"{WHEEL_PACKAGE}*.whl"),
         *(REPO / "torch_extension" / "dist").glob(f"{WHEEL_PACKAGE}*.whl")],
        key=lambda p: p.stat().st_mtime,
    )
    if not wheels:
        raise SystemExit(
            f"wheel build reported success but produced no {WHEEL_PACKAGE} wheel; "
            f"see {LOG}"
        )

    print(f"installing {wheels[-1].name}")
    pip = subprocess.run(
        [sys.executable, "-m", "pip", "install", str(wheels[-1]),
         "--force-reinstall", "--no-deps"],
        capture_output=True, text=True,
    )
    if pip.returncode != 0:
        raise SystemExit(f"installing {wheels[-1]} failed:\n{pip.stderr}")


def wheel_installed() -> bool:
    """Whether the vendor wheel exposing this op's entry point is importable."""
    import importlib.util

    return importlib.util.find_spec(WHEEL_PACKAGE) is not None


def report() -> None:
    print(f"operator      : {OP_NAME}")
    print(f"vendor        : {VENDOR}")
    print(f"soc           : {SOC}")
    print(f"toolkit       : CANN {toolkit_version()} at {ascend_home()}")
    print(f"sources       : {REPO} @ {repo_version()}")
    print(f"install dir   : {vendor_dir()}")
    if NEEDS_WHEEL:
        print(f"python binding: {WHEEL_PACKAGE} "
              f"({'installed' if wheel_installed() else 'missing'})")
    else:
        print("python binding: cann_op.py's own aclnn bridge, no wheel needed")
    print(f"installed     : {'yes' if installed() else 'no'}")
    for symbol in aclnn_symbols():
        print(f"  {symbol}")
    if not installed():
        print()
        print("build it with: python build_cann_op.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the package is already installed")
    parser.add_argument("--check", action="store_true",
                        help="report status and exit without building")
    parser.add_argument("-j", "--jobs", type=int,
                        default=max(1, (os.cpu_count() or 8) // 2),
                        help="parallel compile jobs")
    args = parser.parse_args()

    if args.check:
        report()
        return

    if installed() and not args.force:
        print(f"{OP_NAME} already installed at {vendor_dir()}; --force to rebuild")
        return

    acquire_lock()
    try:
        build(jobs=args.jobs)
    finally:
        release_lock()


if __name__ == "__main__":
    main()
