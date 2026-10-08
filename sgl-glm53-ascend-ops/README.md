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
| `reduce_kernel` | `csrc/reduce/reduce_sum.cpp` |
