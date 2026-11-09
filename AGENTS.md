# AGENTS.md

## Project Overview
This repository contains an MLIR-based compiler pipeline targeting a RISC-V Coral NPU (Neural Processing Unit). The objective is to design a custom dialect (`coral`) positioned between high-level operations (`linalg`) and target-specific LLVM IR (`llvm`), optimizing for hardware-aware operations while maintaining strict numerical and structural correctness.

## High-Level Pipeline Flow
1. **Frontend**: PyTorch (`torch.nn.Module`) $\rightarrow$ Torch-MLIR $\rightarrow$ `linalg-on-tensors` (static shapes).
2. **High-Level Partitioning & Tiling**: Partition compute kernels into Coral hardware blocks (16x16 tiles) and bufferize (`tensor` $\rightarrow$ `memref`).
3. **Coral Dialect**: Represent hardware-accelerated primitives (`coral.matmul`, `coral.add`, `coral.conv2d`).
4. **Lowering to LLVM Dialect**: Convert `coral` operations into RISC-V vector inline assembly or runtime calls targeting the Coral NPU.
5. **Codegen & Simulator**: Generate RISC-V ELF executables and run on the Coral NPU simulator.
6. **Host Runtime & Async**: Coordinate execution and offload between host CPU and NPU accelerator.

---

## Architecture & Hardware Constraints
- **Target ISA**: RISC-V 64-bit with Vector Extension (`RVV` / `rv64gcv`).
- **Coral NPU Microarchitecture**:
  - Fixed 16x16 tile accelerator units for matrix operations.
  - Operations: 8-bit integer (`i8`) inputs with 32-bit integer (`i32`) accumulators.
- **LLVM Version Target**: Tracking LLVM `main` (LLVM 20+ / trunk). Code must remain compatible with modern LLVM/MLIR trunk APIs.

---

## Dialect & Lowering Rules

### 1. Operations & TableGen
- All operations, types, and traits must be formally specified in TableGen (`.td`) files before C++ implementations are modified.
- Operations that mutate memory buffers must properly capture memory side effects (`MemoryEffectsOpInterface`).

### 2. Tiling and Bufferization Invariants
- `coral.*` operations require statically tiled shapes matching hardware constraints (e.g., 16x16).
- **Bufferization is mandatory before LLVM lowering**: Do not convert unbufferized `tensor` types directly to LLVM raw pointers. Always run structured bufferization (`one-shot-bufferize`) so operations act on `memref` before lowering to the LLVM dialect.
- When replacing operations in pattern rewriters, never emit invalid dummy casts (e.g. `UnrealizedConversionCastOp` with empty inputs). Every value and memory location must be valid.

### 3. Assembly Generation
- RISC-V vector instructions emitted via `LLVM::InlineAsmOp` must properly preserve and handle destination buffers (e.g., storing accumulators back to destination memory via `vse32.v`).

---

## Repository Structure & Submodules
- `compiler/` (or `coral-mlir/`): Core dialect definitions, TableGen files, passes, and `coral-opt` driver tool.
- `frontend/`: PyTorch model export and `torch-mlir` ingestion scripts.
- `runtime/`: Host runtime library, loader, and simulator dispatch interface.
- `third_party/`: Pinned Git submodules (`llvm-project`, `torch-mlir`, `coralnpu`). **Do not directly edit code inside `third_party/`.**
- `test/`: Lit and FileCheck-based unit tests for all MLIR passes.

---

## Engineering Guidelines for AI Agents
1. **Preserve Documentation & Comments**: Maintain existing docstrings and architectural notes.
2. **Verified Transformations**: Every compiler pass must have a corresponding `.mlir` test under `test/` checked with `FileCheck`.
3. **Build Target Awareness**: Always ensure CMake and LLVM builds configure the `RISCV` backend (`-DLLVM_TARGETS_TO_BUILD="RISCV;X86"`).
