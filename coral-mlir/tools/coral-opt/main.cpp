#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// Pull in the custom Coral dialect
#include "Coral/CoralDialect.h"

// Linalg to Coral pass declaration
void registerLinalgToCoralPass();

int main(int argc, char **argv) {
  // 1. Create the registry
  mlir::DialectRegistry registry;

  // 2. Explicitly load only the core AI dialects we need for our pipeline
  registry.insert<mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect,
                  mlir::vector::VectorDialect,
                  mlir::coral::CoralDialect>();

  // Register the Linalg to Coral pass
  registerLinalgToCoralPass();

  // 3. Hand control over to the core MLIR command-line interface
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Coral NPU Optimizer Tool\n", registry));
}
