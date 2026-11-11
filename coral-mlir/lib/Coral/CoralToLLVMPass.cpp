#include "Coral/CoralDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

// Helper function to extract the aligned data pointer from a MemRef descriptor
static Value extractAlignedPtr(PatternRewriter &rewriter, Location loc, Value memref, MLIRContext *ctx) {
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i64Ty = IntegerType::get(ctx, 64);
  auto arrayTy = LLVM::LLVMArrayType::get(i64Ty, 2);
  auto structTy = LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, ptrTy, i64Ty, arrayTy, arrayTy});

  auto cast = rewriter.create<UnrealizedConversionCastOp>(loc, structTy, memref).getResult(0);
  return rewriter.create<LLVM::ExtractValueOp>(loc, ptrTy, cast, ArrayRef<int64_t>{1});
}

// ============================================================================
// STAGE 1 REWRITE PATTERNS: Compute Kernels
// ============================================================================

// 1. Lower coral.matmul -> coral_runtime_exec_matmul(ptrA, ptrB, ptrC)
struct ConvertCoralMatmulToLLVM : public RewritePattern {
  ConvertCoralMatmulToLLVM(MLIRContext *context)
      : RewritePattern("coral.matmul", /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto ctx = getContext();

    if (op->getNumOperands() < 3)
      return failure();

    auto module = op->getParentOfType<ModuleOp>();
    auto runtimeFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("coral_runtime_exec_matmul");

    if (!runtimeFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, ptrTy});

      runtimeFunc = rewriter.create<LLVM::LLVMFuncOp>(
          loc, "coral_runtime_exec_matmul", funcTy);
    }

    Value ptrA = extractAlignedPtr(rewriter, loc, op->getOperand(0), ctx);
    Value ptrB = extractAlignedPtr(rewriter, loc, op->getOperand(1), ctx);
    Value ptrC = extractAlignedPtr(rewriter, loc, op->getOperand(2), ctx);

    rewriter.create<LLVM::CallOp>(
        loc,
        runtimeFunc,
        ValueRange{ptrA, ptrB, ptrC}
    );

    rewriter.eraseOp(op);
    return success();
  }
};

// 2. Lower coral.conv2d -> coral_runtime_exec_conv2d(ptrInput, ptrFilter, ptrOutput)
struct ConvertCoralConv2DToLLVM : public RewritePattern {
  ConvertCoralConv2DToLLVM(MLIRContext *context)
      : RewritePattern("coral.conv2d", /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto ctx = getContext();

    if (op->getNumOperands() < 3)
      return failure();

    auto module = op->getParentOfType<ModuleOp>();
    auto runtimeFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("coral_runtime_exec_conv2d");

    if (!runtimeFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, ptrTy});

      runtimeFunc = rewriter.create<LLVM::LLVMFuncOp>(
          loc, "coral_runtime_exec_conv2d", funcTy);
    }

    Value ptrInput  = extractAlignedPtr(rewriter, loc, op->getOperand(0), ctx);
    Value ptrFilter = extractAlignedPtr(rewriter, loc, op->getOperand(1), ctx);
    Value ptrOutput = extractAlignedPtr(rewriter, loc, op->getOperand(2), ctx);

    rewriter.create<LLVM::CallOp>(
        loc,
        runtimeFunc,
        ValueRange{ptrInput, ptrFilter, ptrOutput}
    );

    rewriter.eraseOp(op);
    return success();
  }
};

// 3. Lower coral.add -> coral_runtime_exec_add(ptrA, ptrB, ptrOut)
struct ConvertCoralAddToLLVM : public RewritePattern {
  ConvertCoralAddToLLVM(MLIRContext *context)
      : RewritePattern("coral.add", /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto ctx = getContext();

    if (op->getNumOperands() < 2)
      return failure();

    auto module = op->getParentOfType<ModuleOp>();
    auto runtimeFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("coral_runtime_exec_add");

    if (!runtimeFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);

      SmallVector<Type, 3> paramTypes;
      for (unsigned i = 0; i < op->getNumOperands(); ++i)
        paramTypes.push_back(ptrTy);

      auto funcTy = LLVM::LLVMFunctionType::get(voidTy, paramTypes);
      runtimeFunc = rewriter.create<LLVM::LLVMFuncOp>(
          loc, "coral_runtime_exec_add", funcTy);
    }

    SmallVector<Value, 3> ptrs;
    for (unsigned i = 0; i < op->getNumOperands(); ++i)
      ptrs.push_back(extractAlignedPtr(rewriter, loc, op->getOperand(i), ctx));

    rewriter.create<LLVM::CallOp>(loc, runtimeFunc, ptrs);
    rewriter.eraseOp(op);
    return success();
  }
};

// ============================================================================
// STAGE 2 REWRITE PATTERNS: Fused Elementwise Operations
// ============================================================================

// 4. Lower coral.bn_relu -> coral_runtime_exec_bn_relu(ptrX, ptrMean, ptrScale, ptrBias, ptrOut)
struct ConvertCoralBnReluToLLVM : public RewritePattern {
  ConvertCoralBnReluToLLVM(MLIRContext *context)
      : RewritePattern("coral.bn_relu", /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto ctx = getContext();

    if (op->getNumOperands() < 5)
      return failure();

    auto module = op->getParentOfType<ModuleOp>();
    auto runtimeFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("coral_runtime_exec_bn_relu");

    if (!runtimeFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto ptrTy  = LLVM::LLVMPointerType::get(ctx);
      // Signature: (ptr X, ptr mean, ptr scale, ptr bias, ptr out)
      auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, ptrTy, ptrTy, ptrTy});

      runtimeFunc = rewriter.create<LLVM::LLVMFuncOp>(
          loc, "coral_runtime_exec_bn_relu", funcTy);
    }

    SmallVector<Value, 5> ptrs;
    for (unsigned i = 0; i < op->getNumOperands(); ++i)
      ptrs.push_back(extractAlignedPtr(rewriter, loc, op->getOperand(i), ctx));

    rewriter.create<LLVM::CallOp>(loc, runtimeFunc, ptrs);
    rewriter.eraseOp(op);
    return success();
  }
};

// 5. Lower coral.add_relu -> coral_runtime_exec_add_relu(ptrX, ptrResidual, ptrOut)
struct ConvertCoralAddReluToLLVM : public RewritePattern {
  ConvertCoralAddReluToLLVM(MLIRContext *context)
      : RewritePattern("coral.add_relu", /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto ctx = getContext();

    if (op->getNumOperands() < 3)
      return failure();

    auto module = op->getParentOfType<ModuleOp>();
    auto runtimeFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("coral_runtime_exec_add_relu");

    if (!runtimeFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto ptrTy  = LLVM::LLVMPointerType::get(ctx);
      // Signature: (ptr X, ptr residual, ptr out)
      auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, ptrTy});

      runtimeFunc = rewriter.create<LLVM::LLVMFuncOp>(
          loc, "coral_runtime_exec_add_relu", funcTy);
    }

    SmallVector<Value, 3> ptrs;
    for (unsigned i = 0; i < op->getNumOperands(); ++i)
      ptrs.push_back(extractAlignedPtr(rewriter, loc, op->getOperand(i), ctx));

    rewriter.create<LLVM::CallOp>(loc, runtimeFunc, ptrs);
    rewriter.eraseOp(op);
    return success();
  }
};

// ============================================================================
// STAGE 3 REWRITE PATTERNS: Shape & Memory Operations
// ============================================================================

// 6. Lower coral.dma_pad -> coral_runtime_exec_dma_pad(ptrSrc, ptrDst, lowPadH, highPadH, lowPadW, highPadW)
struct ConvertCoralDmaPadToLLVM : public RewritePattern {
  ConvertCoralDmaPadToLLVM(MLIRContext *context)
      : RewritePattern("coral.dma_pad", /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto ctx = getContext();

    if (op->getNumOperands() < 2)
      return failure();

    auto module = op->getParentOfType<ModuleOp>();
    auto runtimeFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("coral_runtime_exec_dma_pad");

    if (!runtimeFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto ptrTy  = LLVM::LLVMPointerType::get(ctx);
      auto i64Ty  = IntegerType::get(ctx, 64);
      // Signature: (ptr src, ptr dst, i64 low_pad_h, i64 high_pad_h, i64 low_pad_w, i64 high_pad_w)
      auto funcTy = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, i64Ty, i64Ty, i64Ty, i64Ty});

      runtimeFunc = rewriter.create<LLVM::LLVMFuncOp>(
          loc, "coral_runtime_exec_dma_pad", funcTy);
    }

    Value ptrSrc = extractAlignedPtr(rewriter, loc, op->getOperand(0), ctx);
    Value ptrDst = extractAlignedPtr(rewriter, loc, op->getOperand(1), ctx);

    int64_t lowH = 0, highH = 0, lowW = 0, highW = 0;
    if (auto lowAttr = op->getAttrOfType<ArrayAttr>("low_pad")) {
      if (lowAttr.size() >= 4) {
        lowH = cast<IntegerAttr>(lowAttr[2]).getInt();
        lowW = cast<IntegerAttr>(lowAttr[3]).getInt();
      }
    }
    if (auto highAttr = op->getAttrOfType<ArrayAttr>("high_pad")) {
      if (highAttr.size() >= 4) {
        highH = cast<IntegerAttr>(highAttr[2]).getInt();
        highW = cast<IntegerAttr>(highAttr[3]).getInt();
      }
    }

    Value valLowH  = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(lowH));
    Value valHighH = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(highH));
    Value valLowW  = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(lowW));
    Value valHighW = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(highW));

    rewriter.create<LLVM::CallOp>(
        loc, runtimeFunc, ValueRange{ptrSrc, ptrDst, valLowH, valHighH, valLowW, valHighW});
    rewriter.eraseOp(op);
    return success();
  }
};

// ============================================================================
// PASS DEFINITION
// ============================================================================

struct CoralToLLVMPass : public PassWrapper<CoralToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CoralToLLVMPass)

  StringRef getArgument() const final { return "convert-coral-to-llvm"; }
  StringRef getDescription() const final {
    return "Lower Coral dialect ops (matmul, conv2d, add, bn_relu, add_relu, dma_pad) to host C runtime driver calls";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // Stage 1 patterns
    patterns.add<ConvertCoralMatmulToLLVM>(&getContext());
    patterns.add<ConvertCoralConv2DToLLVM>(&getContext());
    patterns.add<ConvertCoralAddToLLVM>(&getContext());
    // Stage 2 patterns
    patterns.add<ConvertCoralBnReluToLLVM>(&getContext());
    patterns.add<ConvertCoralAddReluToLLVM>(&getContext());
    // Stage 3 patterns
    patterns.add<ConvertCoralDmaPadToLLVM>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace

std::unique_ptr<Pass> createCoralToLLVMPass() {
  return std::make_unique<CoralToLLVMPass>();
}

void registerCoralToLLVMPass() {
  PassRegistration<CoralToLLVMPass>();
}