#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "Coral/CoralDialect.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

// ============================================================================
// STAGE 1 PATTERNS: Compute-Heavy Kernels
// ============================================================================

// 1. Tiled Matmul Pattern (16x16 static shape) -> coral.matmul
struct ConvertTiledMatmulToCoral : public OpRewritePattern<linalg::MatmulOp> {
  ConvertTiledMatmulToCoral(MLIRContext *context)
      : OpRewritePattern<linalg::MatmulOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getDpsInputs();
    auto outputs = op.getDpsInits();

    if (inputs.size() != 2 || outputs.size() != 1)
      return failure();

    auto shapedType = dyn_cast<ShapedType>(inputs[0].getType());
    if (!shapedType || !shapedType.hasStaticShape())
      return failure();

    ArrayRef<int64_t> shape = shapedType.getShape();
    if (shape.size() != 2 || shape[0] != 16 || shape[1] != 16) {
      return failure();
    }

    SmallVector<Value, 3> allOperands;
    allOperands.push_back(inputs[0]);  // Pointer to A
    allOperands.push_back(inputs[1]);  // Pointer to B
    allOperands.push_back(outputs[0]); // Pointer to C (Destination)

    OperationState state(op.getLoc(), "coral.matmul");
    state.addOperands(allOperands);

    if (op.hasTensorSemantics()) {
      state.addTypes(outputs[0].getType());
      Operation *newOp = rewriter.create(state);
      rewriter.replaceOp(op, newOp->getResults());
    } else {
      rewriter.create(state);
      rewriter.eraseOp(op);
    }

    return success();
  }
};

// 2. Tiled Integer Addition Pattern (16x16) -> coral.add
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

// 3. NCHW 2D Convolution Pattern (Torch-MLIR default) -> coral.conv2d
struct ConvertLinalgConv2DNchwToCoral : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
  ConvertLinalgConv2DNchwToCoral(MLIRContext *context)
      : OpRewritePattern<linalg::Conv2DNchwFchwOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getDpsInputs();
    auto outputs = op.getDpsInits();

    if (inputs.size() != 2 || outputs.size() != 1)
      return failure();

    SmallVector<Value, 3> allOperands;
    allOperands.push_back(inputs[0]);  // Input tensor/memref [N, C, H, W]
    allOperands.push_back(inputs[1]);  // Filter tensor/memref [F, C, KH, KW]
    allOperands.push_back(outputs[0]); // Output destination [N, F, OH, OW]

    OperationState state(op.getLoc(), "coral.conv2d");
    state.addOperands(allOperands);

    if (auto strides = op.getStrides())
      state.addAttribute("strides", strides);
    if (auto dilations = op.getDilations())
      state.addAttribute("dilations", dilations);

    if (op.hasTensorSemantics()) {
      state.addTypes(outputs[0].getType());
      Operation *newOp = rewriter.create(state);
      rewriter.replaceOp(op, newOp->getResults());
    } else {
      rewriter.create(state);
      rewriter.eraseOp(op);
    }

    return success();
  }
};

// 4. NHWC 2D Convolution Pattern (Fallback) -> coral.conv2d
struct ConvertLinalgConv2DNhwcToCoral : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  ConvertLinalgConv2DNhwcToCoral(MLIRContext *context)
      : OpRewritePattern<linalg::Conv2DNhwcHwcfOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getDpsInputs();
    auto outputs = op.getDpsInits();

    if (inputs.size() != 2 || outputs.size() != 1)
      return failure();

    SmallVector<Value, 3> allOperands;
    allOperands.push_back(inputs[0]);
    allOperands.push_back(inputs[1]);
    allOperands.push_back(outputs[0]);

    OperationState state(op.getLoc(), "coral.conv2d");
    state.addOperands(allOperands);

    if (auto strides = op.getStrides())
      state.addAttribute("strides", strides);
    if (auto dilations = op.getDilations())
      state.addAttribute("dilations", dilations);

    if (op.hasTensorSemantics()) {
      state.addTypes(outputs[0].getType());
      Operation *newOp = rewriter.create(state);
      rewriter.replaceOp(op, newOp->getResults());
    } else {
      rewriter.create(state);
      rewriter.eraseOp(op);
    }

    return success();
  }
};

// ============================================================================
// STAGE 2 PATTERNS: Elementwise Fusion (BatchNorm + ReLU & Residual Additions)
// ============================================================================

// 5. BatchNorm + ReLU Pattern -> coral.bn_relu
// Matches: subf -> mulf -> addf -> cmpf -> select inside linalg.generic
struct FuseBatchNormReLUPattern : public OpRewritePattern<linalg::GenericOp> {
  FuseBatchNormReLUPattern(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    Block &block = genericOp.getBlock();
    auto &ops = block.getOperations();

    if (ops.size() < 6)
      return failure();

    auto it = ops.begin();
    auto subOp    = dyn_cast<arith::SubFOp>(&*it++);
    auto mulOp    = dyn_cast<arith::MulFOp>(&*it++);
    auto addOp    = dyn_cast<arith::AddFOp>(&*it++);
    auto cmpOp    = dyn_cast<arith::CmpFOp>(&*it++);
    auto selectOp = dyn_cast<arith::SelectOp>(&*it++);
    auto yieldOp  = dyn_cast<linalg::YieldOp>(&*it++);

    if (!subOp || !mulOp || !addOp || !cmpOp || !selectOp || !yieldOp)
      return failure();

    // Verify SSA value dataflow
    if (mulOp.getLhs() != subOp.getResult() && mulOp.getRhs() != subOp.getResult())
      return failure();
    if (addOp.getLhs() != mulOp.getResult() && addOp.getRhs() != mulOp.getResult())
      return failure();
    if (selectOp.getTrueValue() != addOp.getResult())
      return failure();

    auto inputs  = genericOp.getDpsInputs();  // [input, mean, scale, bias]
    auto outputs = genericOp.getDpsInits();   // [output destination]

    if (inputs.size() < 4 || outputs.empty())
      return failure();

    SmallVector<Value, 5> operands;
    operands.push_back(inputs[0]); // Input X
    operands.push_back(inputs[1]); // Mean
    operands.push_back(inputs[2]); // Scale
    operands.push_back(inputs[3]); // Bias
    operands.push_back(outputs[0]);// Output Buffer

    OperationState state(genericOp.getLoc(), "coral.bn_relu");
    state.addOperands(operands);

    if (genericOp.hasTensorSemantics()) {
      state.addTypes(outputs[0].getType());
      Operation *fusedOp = rewriter.create(state);
      rewriter.replaceOp(genericOp, fusedOp->getResults());
    } else {
      rewriter.create(state);
      rewriter.eraseOp(genericOp);
    }

    return success();
  }
};

// 6. Residual Addition + ReLU Pattern -> coral.add_relu
// Matches: addf -> cmpf -> select inside linalg.generic
struct FuseResidualAddReLUPattern : public OpRewritePattern<linalg::GenericOp> {
  FuseResidualAddReLUPattern(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    Block &block = genericOp.getBlock();
    auto &ops = block.getOperations();

    if (ops.size() < 3 || ops.size() > 4)
      return failure();

    auto it = ops.begin();
    auto addOp = dyn_cast<arith::AddFOp>(&*it++);
    if (!addOp)
      return failure();

    arith::SelectOp selectOp = nullptr;
    if (ops.size() == 4) {
      auto cmpOp = dyn_cast<arith::CmpFOp>(&*it++);
      selectOp   = dyn_cast<arith::SelectOp>(&*it++);
      if (!cmpOp || !selectOp)
        return failure();
    } else {
      selectOp = dyn_cast<arith::SelectOp>(&*it++);
      if (!selectOp)
        return failure();
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(&*it++);
    if (!yieldOp)
      return failure();

    auto inputs  = genericOp.getDpsInputs();  // [x, f(x)]
    auto outputs = genericOp.getDpsInits();   // [out]

    if (inputs.size() != 2 || outputs.empty())
      return failure();

    SmallVector<Value, 3> operands;
    operands.push_back(inputs[0]);
    operands.push_back(inputs[1]);
    operands.push_back(outputs[0]);

    OperationState state(genericOp.getLoc(), "coral.add_relu");
    state.addOperands(operands);

    if (genericOp.hasTensorSemantics()) {
      state.addTypes(outputs[0].getType());
      Operation *fusedOp = rewriter.create(state);
      rewriter.replaceOp(genericOp, fusedOp->getResults());
    } else {
      rewriter.create(state);
      rewriter.eraseOp(genericOp);
    }

    return success();
  }
};

// ============================================================================
// STAGE 3 PATTERNS: Shape & Memory Operations
// ============================================================================

// 7. Tensor Padding Pattern -> coral.dma_pad
struct LowerTensorPadToCoralDMA : public OpRewritePattern<tensor::PadOp> {
  LowerTensorPadToCoralDMA(MLIRContext *context)
      : OpRewritePattern<tensor::PadOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    Location loc = padOp.getLoc();
    Value source = padOp.getSource();

    SmallVector<int64_t> lowPad  = padOp.getStaticLow();
    SmallVector<int64_t> highPad = padOp.getStaticHigh();

    OperationState state(loc, "coral.dma_pad");
    state.addOperands({source});
    state.addAttribute("low_pad", rewriter.getI64ArrayAttr(lowPad));
    state.addAttribute("high_pad", rewriter.getI64ArrayAttr(highPad));

    if (padOp.getType()) {
      state.addTypes(padOp.getType());
      Operation *dmaOp = rewriter.create(state);
      rewriter.replaceOp(padOp, dmaOp->getResults());
    } else {
      rewriter.create(state);
      rewriter.eraseOp(padOp);
    }

    return success();
  }
};

// ============================================================================
// PASS DEFINITION
// ============================================================================

struct LinalgToCoralPass : public PassWrapper<LinalgToCoralPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToCoralPass)

  StringRef getArgument() const final { return "convert-linalg-to-coral"; }
  StringRef getDescription() const final {
    return "Partition linalg/arith ops, fold BatchNorm/ReLU, and offload memory padding to Coral NPU ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::coral::CoralDialect, mlir::arith::ArithDialect, mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    
    // Register Stage 1 patterns
    patterns.add<ConvertTiledMatmulToCoral>(context);
    patterns.add<ConvertArithAddIToCoral>(context);
    patterns.add<ConvertLinalgConv2DNchwToCoral>(context);
    patterns.add<ConvertLinalgConv2DNhwcToCoral>(context);

    // Register Stage 2 patterns
    patterns.add<FuseBatchNormReLUPattern>(context);
    patterns.add<FuseResidualAddReLUPattern>(context);

    // Register Stage 3 patterns
    patterns.add<LowerTensorPadToCoralDMA>(context);

    // Initialize Conversion Target
    ConversionTarget target(*context);

    // Default CPU Fallback: Keep un-matched ops dynamically legal for CPU execution
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    // Coral Dialect is the valid target legal dialect
    target.addLegalDialect<mlir::coral::CoralDialect>();

    // Stage 1 Legality Rules
    target.addDynamicallyLegalOp<linalg::MatmulOp>([](linalg::MatmulOp op) {
      auto inputs = op.getDpsInputs();
      if (inputs.empty()) return true;
      
      auto tensorType = dyn_cast<ShapedType>(inputs[0].getType());
      if (tensorType && tensorType.hasStaticShape()) {
        ArrayRef<int64_t> shape = tensorType.getShape();
        if (shape.size() == 2 && shape[0] == 16 && shape[1] == 16) {
          return false; // ILLEGAL -> Convert to NPU
        }
      }
      return true;
    });

    target.addDynamicallyLegalOp<arith::AddIOp>([](arith::AddIOp op) {
      auto tensorType = dyn_cast<ShapedType>(op.getType());
      if (tensorType && tensorType.hasStaticShape()) {
        ArrayRef<int64_t> shape = tensorType.getShape();
        if (shape.size() == 2 && shape[0] == 16 && shape[1] == 16) {
          return false; // ILLEGAL -> Convert to NPU
        }
      }
      return true;
    });

    target.addDynamicallyLegalOp<linalg::Conv2DNchwFchwOp>([](linalg::Conv2DNchwFchwOp op) {
      if (op.getDpsInputs().size() == 2 && op.getDpsInits().size() == 1) {
        return false; // ILLEGAL -> Convert to NPU
      }
      return true;
    });

    target.addDynamicallyLegalOp<linalg::Conv2DNhwcHwcfOp>([](linalg::Conv2DNhwcHwcfOp op) {
      if (op.getDpsInputs().size() == 2 && op.getDpsInits().size() == 1) {
        return false; // ILLEGAL -> Convert to NPU
      }
      return true;
    });

    // Stage 2 Legality Rules: Intercept generic ops matching BatchNorm+ReLU or Residual Add+ReLU
    target.addDynamicallyLegalOp<linalg::GenericOp>([](linalg::GenericOp op) {
      Block &block = op.getBlock();
      auto &ops = block.getOperations();

      // Check for BatchNorm + ReLU sequence (subf -> mulf -> addf -> cmpf -> select -> yield)
      if (ops.size() >= 6) {
        auto it = ops.begin();
        if (isa<arith::SubFOp>(&*it++) && isa<arith::MulFOp>(&*it++) && isa<arith::AddFOp>(&*it++))
          return false; // ILLEGAL -> Convert to coral.bn_relu
      }

      // Check for Residual Addition + ReLU sequence (addf -> cmpf/select -> yield)
      if (ops.size() == 3 || ops.size() == 4) {
        auto it = ops.begin();
        if (isa<arith::AddFOp>(&*it++))
          return false; // ILLEGAL -> Convert to coral.add_relu
      }

      return true; // LEGAL -> Keep on CPU
    });

    // Stage 3 Legality Rules: Intercept tensor.pad operations
    target.addDynamicallyLegalOp<tensor::PadOp>([](tensor::PadOp op) {
      return false; // ILLEGAL -> Offload padding to coral.dma_pad
    });

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace

void registerLinalgToCoralPass() {
  PassRegistration<LinalgToCoralPass>();
}