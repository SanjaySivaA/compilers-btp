#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "Coral/CoralDialect.h"

using namespace mlir;

namespace {

struct ConvertTiledMatmulToCoral : public OpRewritePattern<linalg::MatmulOp> {
  ConvertTiledMatmulToCoral(MLIRContext *context)
      : OpRewritePattern<linalg::MatmulOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getDpsInputs();
    auto outputs = op.getDpsInits();

    if (inputs.size() != 2 || outputs.size() != 1)
      return failure();

    auto tensorType = dyn_cast<ShapedType>(inputs[0].getType());
    if (!tensorType || !tensorType.hasStaticShape())
      return failure();

    ArrayRef<int64_t> shape = tensorType.getShape();
    if (shape.size() != 2 || shape[0] != 16 || shape[1] != 16) {
      return failure();
    }

    OperationState state(op.getLoc(), "coral.matmul");
    state.addOperands(inputs);
    state.addTypes(outputs[0].getType());

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());

    return success();
  }
};

struct ConvertArithAddIToCoral : public OpRewritePattern<arith::AddIOp> {
  ConvertArithAddIToCoral(MLIRContext *context)
      : OpRewritePattern<arith::AddIOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    
    auto tensorType = dyn_cast<ShapedType>(op.getType());
    if (!tensorType || !tensorType.hasStaticShape())
      return failure();

    ArrayRef<int64_t> shape = tensorType.getShape();
    if (shape.size() != 2 || shape[0] != 16 || shape[1] != 16) {
      return failure();
    }

    OperationState state(op.getLoc(), "coral.add");
    state.addOperands(op.getOperands());
    state.addTypes(op.getType());

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());

    return success();
  }
};

// The Pattern: Hunt for NHWC 2D Convolutions
struct ConvertLinalgConv2DToCoral : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  ConvertLinalgConv2DToCoral(MLIRContext *context)
      : OpRewritePattern<linalg::Conv2DNhwcHwcfOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp op,
                                PatternRewriter &rewriter) const override {
    
    auto inputs = op.getDpsInputs();
    auto outputs = op.getDpsInits();

    if (inputs.size() != 2 || outputs.size() != 1)
      return failure();

    OperationState state(op.getLoc(), "coral.conv2d");
    state.addOperands(inputs);
    state.addTypes(outputs[0].getType());

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());

    return success();
  }
};

struct LinalgToCoralPass : public PassWrapper<LinalgToCoralPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToCoralPass)

  StringRef getArgument() const final { return "convert-linalg-to-coral"; }
  StringRef getDescription() const final { return "Hijack linalg ops (matmul/conv2d) and arith.addi, converting to coral ops"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::coral::CoralDialect, mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertTiledMatmulToCoral>(&getContext());
    patterns.add<ConvertArithAddIToCoral>(&getContext());
    patterns.add<ConvertLinalgConv2DToCoral>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace

void registerLinalgToCoralPass() {
  PassRegistration<LinalgToCoralPass>();
}