# sgl-glm53-ascend-ops

Out-of-tree Ascend C operator extensions for GLM-5.3-Flash on SGLang.

## Prerequisites

- CANN toolkit with `bisheng` compiler (tested with CANN 9.0)
- PyTorch with `torch_npu`
- CMake ≥ 3.20, Ninja

## Build

### Configure

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

### Build one kernel

```bash
cmake --build build --target prepared_recurrent_kernel
```

### Build all kernels

```bash
cmake --build build --target kernels
```

### Build the final Python extension

```bash
cmake --build build --target custom_ops_lib
```

### Selective kernel inclusion

Build only specific kernels by setting `SGL_OPS` at configure time:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DSGL_OPS="prepared_recurrent;reduce"
```

### Editable install

```bash
pip install -e .
```

This uses `scikit-build-core` to configure, build, and link the extension
automatically. Ninja is the default build backend.

### Full rebuild (non-editable)

```bash
pip install .
```

## Available kernel targets

| Target | Source |
|---|---|
| `helloworld_kernel` | `csrc/helloworld/helloworld.cpp` |
| `prepared_recurrent_kernel` | `csrc/kda/prepared_recurrent.cpp` |
| `kda_varlen_recurrent_kernel` | `csrc/kda/kda_varlen_recurrent*.cpp` |
| `kda_varlen_chunk_kernel` | `csrc/kda/kda_varlen_chunk*.cpp` |
| `reduce_kernel` | `csrc/reduce/reduce_sum.cpp` |

`SGL_OPS` is cached at configure time. If a newly added kernel reports
`Kernel skipped`, the cache holds an explicit list from an earlier configure —
pass the full list again (or delete `build/CMakeCache.txt`):

```bash
cmake -B build -G Ninja -DSGL_OPS="helloworld;prepared_recurrent;reduce;kda_varlen_recurrent;kda_varlen_chunk"
```
