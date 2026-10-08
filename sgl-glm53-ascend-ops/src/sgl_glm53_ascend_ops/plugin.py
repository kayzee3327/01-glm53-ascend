from sglang.srt.plugins.hook_registry import HookRegistry, HookType


def _prepared_recurrent(*args, **kwargs):
    from .ops.kda import prepared_recurrent

    return prepared_recurrent(*args, **kwargs)


def register() -> None:
    HookRegistry.register(
        "SGLANG_TARGET_PATH",
        _prepared_recurrent,
        HookType.REPLACE,
    )