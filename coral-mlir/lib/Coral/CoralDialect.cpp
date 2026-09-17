#include "Coral/CoralDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::coral;

// 1. Pull in the auto-generated TableGen logic (the actual implementation)
#define GET_OP_CLASSES
#include "Coral/CoralOps.cpp.inc"

// 2. The Constructor: Registering the dialect with the core MLIR context
CoralDialect::CoralDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<CoralDialect>()) {
  
  // 3. The Magic Librarian Step! 
  // We feed a macro to TableGen that automatically lists all the operations 
  // it created (like MatMulOp) and loads them into the compiler.
  addOperations<
#define GET_OP_LIST
#include "Coral/CoralOps.cpp.inc"
      >();
}
