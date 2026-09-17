#ifndef CORAL_DIALECT_H
#define CORAL_DIALECT_H

#include "mlir/IR/Dialect.h"
// --- The Missing Pieces ---
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
// --------------------------

// 1. Pull in the auto-generated TableGen declarations!
#define GET_OP_CLASSES
#include "Coral/CoralOps.h.inc"

namespace mlir {
namespace coral {

// 2. Define our custom Dialect class
class CoralDialect : public ::mlir::Dialect {
public:
  // The constructor where initialization happens
  explicit CoralDialect(::mlir::MLIRContext *context);
  
  // Tells the MLIR engine that our prefix is "coral"
  static ::llvm::StringRef getDialectNamespace() { return "coral"; }
};

} // namespace coral
} // namespace mlir

#endif // CORAL_DIALECT_H
