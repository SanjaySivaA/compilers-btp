#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "Coral/CoralDialect.h"

using namespace mlir;

namespace {

// The Pattern: Hunt for linalg.matmul (Only check for linalg.matmul operation)
struct ConvertTiledMatmulToCoral : public OpRewritePattern<linalg::MatmulOp> {
  ConvertTiledMatmulToCoral(MLIRContext *context)
      : OpRewritePattern<linalg::MatmulOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    
    // Extracts the inputs (matrix A and B) and the output (matrix C)
    auto inputs = op.getDpsInputs();
    auto outputs = op.getDpsInits();

    if (inputs.size() != 2 || outputs.size() != 1)
      return failure();

    // Checks the shape of A
    auto tensorType = dyn_cast<ShapedType>(inputs[0].getType());
    if (!tensorType || !tensorType.hasStaticShape())
      return failure();

    ArrayRef<int64_t> shape = tensorType.getShape();
    // Accepting only 16x16 matrices 
    if (shape.size() != 2 || shape[0] != 16 || shape[1] != 16) {
      return failure();
    }

    // Buids the new coral.matmul operation 
    OperationState state(op.getLoc(), "coral.matmul");
    state.addOperands(inputs);
    state.addTypes(outputs[0].getType());

    // Create the new operation and save a pointer to it
    Operation *newOp = rewriter.create(state);
    
    // Replaces linalg.matmul with coral.matmul 
    rewriter.replaceOp(op, newOp->getResults());

    return success();
  }
};

// The Pass Wrapper : wraps the pattern into a command line tool
struct LinalgToCoralPass : public PassWrapper<LinalgToCoralPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToCoralPass)

  StringRef getArgument() const final { return "convert-linalg-to-coral"; }
  StringRef getDescription() const final { return "Hijack 16x16 linalg.matmul and convert to coral.matmul"; }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::coral::CoralDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertTiledMatmulToCoral>(&getContext());

    // Run the pattern greedily on the function operation
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace

// Expose the registration function
void registerLinalgToCoralPass() {
  PassRegistration<LinalgToCoralPass>();
}