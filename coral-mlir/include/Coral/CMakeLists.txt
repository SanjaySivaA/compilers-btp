set(LLVM_TARGET_DEFINITIONS CoralOps.td)
mlir_tablegen(CoralOps.h.inc -gen-op-decls)
mlir_tablegen(CoralOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(CoralOpsIncGen)
