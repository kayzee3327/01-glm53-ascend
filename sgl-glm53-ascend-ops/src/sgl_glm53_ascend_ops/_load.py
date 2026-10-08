from __future__ import annotations

from pathlib import Path

import torch


_LOADED = False

_LIB_NAME = "sgl_glm53_ascend_ops_lib"


def _find_library() -> Path:
    import sgl_glm53_ascend_ops

    search_dirs: list[Path] = []
    for p in getattr(sgl_glm53_ascend_ops, "__path__", []):
        search_dirs.append(Path(p))
    search_dirs.append(Path(__file__).resolve().parent)
    # Dev fallback: the Makefile builds into build/ at the project root
    search_dirs.append(Path(__file__).resolve().parent.parent.parent.parent / "build")

    libraries: list[Path] = []
    seen: set[Path] = set()
    for d in search_dirs:
        d = d.resolve()
        if d in seen:
            continue
        seen.add(d)
        libraries.extend(d.glob(f"{_LIB_NAME}*.so"))

    if not libraries:
        raise RuntimeError(
            f"Cannot find {_LIB_NAME}*.so in {[str(d) for d in seen]}. "
            "Run 'make' to build the library."
        )

    return libraries[0]


def load_library() -> None:
    global _LOADED

    if _LOADED:
        return

    torch.ops.load_library(str(_find_library()))
    _LOADED = True
