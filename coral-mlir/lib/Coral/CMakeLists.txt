add_mlir_dialect_library(CoralDialect
  CoralDialect.cpp
  LinalgToCoralPass.cpp
  CoralToLLVMPass.cpp

  DEPENDS
  CoralOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRTransforms
  MLIRLLVMDialect
)