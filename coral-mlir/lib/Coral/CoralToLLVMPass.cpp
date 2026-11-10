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

    // Module-level runtime symbol lookup or creation
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
      auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy});

      runtimeFunc = rewriter.create<LLVM::LLVMFuncOp>(
          loc, "coral_runtime_exec_add", funcTy);
    }

    Value ptrA = extractAlignedPtr(rewriter, loc, op->getOperand(0), ctx);
    Value ptrB = extractAlignedPtr(rewriter, loc, op->getOperand(1), ctx);

    rewriter.create<LLVM::CallOp>(
        loc,
        runtimeFunc,
        ValueRange{ptrA, ptrB}
    );

    rewriter.eraseOp(op);
    return success();
  }
};

// Target ModuleOp for module-level symbol insertions
struct CoralToLLVMPass : public PassWrapper<CoralToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CoralToLLVMPass)

  StringRef getArgument() const final { return "convert-coral-to-llvm"; }
  StringRef getDescription() const final {
    return "Lower Coral dialect ops (matmul, conv2d, add) to host C runtime driver calls";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertCoralMatmulToLLVM>(&getContext());
    patterns.add<ConvertCoralConv2DToLLVM>(&getContext());
    patterns.add<ConvertCoralAddToLLVM>(&getContext());

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