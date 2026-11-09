# Compilers BTP

## Project Status

[![MLIR CI Pipeline](https://github.com/SanjaySivaA/compilers-btp/actions/workflows/mlir-ci.yml/badge.svg)](https://github.com/SanjaySivaA/compilers-btp/actions/workflows/mlir-ci.yml)


## Planned project structure

```
compilers-btp/
├── CMakeLists.txt                    # Top-level unified CMake build
├── README.md                         # Pipeline documentation & quickstart
│
├── frontend/                         # [Deliverable 1: PyTorch -> Linalg]
│   ├── requirements.txt              # torch, torch-mlir dependencies
│   ├── compile.py                    # Driver: PyTorch (torch.nn.Module) -> linalg-on-tensors MLIR
│   └── models/                       # Test modules (e.g., 2D conv, 16x16 matmul, MLP)
│
├── compiler/                         # [Deliverable 2: Dialect & Lowering Passes]
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── Coral/
│   │       ├── IR/
│   │       │   ├── CoralDialect.h
│   │       │   └── CoralOps.td       # TableGen op definitions
│   │       ├── Transforms/
│   │       │   ├── Passes.td         # TableGen pass declarations
│   │       │   └── Passes.h          # Tiling to 16x16, bufferization pipeline
│   │       └── Conversion/
│   │           ├── LinalgToCoral/
│   │           │   └── Passes.h
│   │           └── CoralToLLVM/
│   │               └── Passes.h
│   ├── lib/
│   │   ├── IR/
│   │   │   └── CoralDialect.cpp
│   │   ├── Transforms/
│   │   │   └── TileAndFuse.cpp       # Tiles linalg ops to 16x16 so patterns match
│   │   └── Conversion/
│   │       ├── LinalgToCoral/
│   │       │   └── LinalgToCoralPass.cpp
│   │       └── CoralToLLVM/
│   │           └── CoralToLLVMPass.cpp
│   └── tools/
│       └── coral-opt/
│           ├── CMakeLists.txt
│           └── main.cpp              # Opt driver registering all dialects and passes
│
├── runtime/                          # [Deliverable 3: Simulator & Runtime Harness]
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── coral_runtime.h           # Device memory alloc, launch kernel, sync APIs
│   ├── src/
│   │   ├── runtime.cpp               # Host runtime implementation
│   │   └── async_queue.cpp           # (Stretch) Host-NPU command queue / double-buffering
│   └── sim/
│       ├── CMakeLists.txt
│       └── simulator_interface.cpp   # Wrapper calling your RISC-V simulator (e.g. Spike / custom sim)
│
├── test/                             # Regression & Verification Suite
│   ├── CMakeLists.txt
│   ├── lit.cfg.py                    # LLVM lit test configuration
│   ├── Conversion/
│   │   ├── linalg-to-coral.mlir      # lit + FileCheck tests
│   │   └── coral-to-llvm.mlir
│   └── EndToEnd/
│       └── test_matmul_e2e.py        # PyTorch -> MLIR -> Sim execution -> Verify outputs
│
└── .github/
    └── workflows/
        └── mlir-ci.yml               # Builds compiler + runtime with RISCV support & runs lit
```