#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "Coral/CoralDialect.h"
#include "mlir/Transforms/DialectConversion.h"

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

    // Checks the shape of A (ShapedType works for both Tensors and MemRefs)
    auto shapedType = dyn_cast<ShapedType>(inputs[0].getType());
    if (!shapedType || !shapedType.hasStaticShape())
      return failure();

    ArrayRef<int64_t> shape = shapedType.getShape();
    // Accepting only 16x16 matrices 
    if (shape.size() != 2 || shape[0] != 16 || shape[1] != 16) {
      return failure();
    }

    // --- NEW MEMREF LOGIC ---

    // 1. Combine A, B, and C into a single list of 3 physical memory pointers
    SmallVector<Value, 3> allOperands;
    allOperands.push_back(inputs[0]);  // Pointer to A
    allOperands.push_back(inputs[1]);  // Pointer to B
    allOperands.push_back(outputs[0]); // Pointer to C (Destination)

    // 2. Build the new coral.matmul operation (We do not add return types anymore)
    OperationState state(op.getLoc(), "coral.matmul");
    state.addOperands(allOperands);

    // Create the new operation
    rewriter.create(state);
    
    // 3. Erase the old linalg.matmul (Since there are no results to replace)
    rewriter.eraseOp(op);

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
  StringRef getDescription() const final { return "Partition linalg/arith ops between CPU and NPU"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::coral::CoralDialect, mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    
    // Load existing, untouched patterns
    patterns.add<ConvertTiledMatmulToCoral>(context);
    patterns.add<ConvertArithAddIToCoral>(context);
    patterns.add<ConvertLinalgConv2DToCoral>(context);

    // 1. Initialize the Partitioning Target
    ConversionTarget target(*context);

    // 2. Default CPU Fallback: Mark everything else dynamically legal. 
    // This ensures loops, func.func, standard arith, etc., are safely kept for the CPU.
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    // 3. NPU Territory: The Coral dialect is always legal.
    target.addLegalDialect<mlir::coral::CoralDialect>();

    // 4. Partitioning Rules: Target specific ops for the NPU based on constraints
    target.addDynamicallyLegalOp<linalg::MatmulOp>([](linalg::MatmulOp op) {
      auto inputs = op.getDpsInputs();
      if (inputs.empty()) return true; // Keep on CPU
      
      auto tensorType = dyn_cast<ShapedType>(inputs[0].getType());
      if (tensorType && tensorType.hasStaticShape()) {
        ArrayRef<int64_t> shape = tensorType.getShape();
        if (shape.size() == 2 && shape[0] == 16 && shape[1] == 16) {
          return false; // ILLEGAL: Matches our 16x16 constraint. Must go to NPU!
        }
      }
      return true; // LEGAL: Keep on CPU
    });

    target.addDynamicallyLegalOp<arith::AddIOp>([](arith::AddIOp op) {
      auto tensorType = dyn_cast<ShapedType>(op.getType());
      if (tensorType && tensorType.hasStaticShape()) {
        ArrayRef<int64_t> shape = tensorType.getShape();
        if (shape.size() == 2 && shape[0] == 16 && shape[1] == 16) {
          return false; // ILLEGAL: NPU bound
        }
      }
      return true; // LEGAL: Keep on CPU
    });

    target.addDynamicallyLegalOp<linalg::Conv2DNhwcHwcfOp>([](linalg::Conv2DNhwcHwcfOp op) {
      // Mirroring pattern's requirements
      if (op.getDpsInputs().size() == 2 && op.getDpsInits().size() == 1) {
        return false; // ILLEGAL: NPU bound
      }
      return true; // LEGAL: Keep on CPU
    });

    // 5. Apply the partition
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
}

void registerLinalgToCoralPass() {
  PassRegistration<LinalgToCoralPass>();
}