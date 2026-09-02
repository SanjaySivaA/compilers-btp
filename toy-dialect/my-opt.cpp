#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
struct PrintOpsPass : PassWrapper<PrintOpsPass, OperationPass<ModuleOp>> {
    // This creates a unique id for the pass
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintOpsPass);

    StringRef getArgument() const override { return "print-opt"; }
    StringRef getDescription() const override { return "This is a pass that will print the operation names. This is an analysis pass."; }

    // pass logic
    void runOnOperation() override {
        // getOperation() returns the top-level ops that this pass is running on (Usually the Module)
        // .walk() automatically iterates through every nested operation inside it
        getOperation()->walk([](Operation *op){
            llvm::outs() << "Visiting Op: " << op->getName() << ";\n";
        });
    }
};
} // end anonymous namespace

int main(int argc, char **argv) {
    // create a registry and load all standard MLIR dialects
    DialectRegistry registry;
    registerAllDialects(registry);

    // register our custom pass with the command line parser
    PassRegistration<PrintOpsPass>();

    // run the tool (handles command line arguments, parsing, etc.)
    return asMainReturnCode(
        MlirOptMain(argc, argv, "My Custom MLIR Optimizer", registry)
    );
}