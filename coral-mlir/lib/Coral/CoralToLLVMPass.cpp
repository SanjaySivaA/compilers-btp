#include "Coral/CoralDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct ConvertCoralMatmulToLLVM : public RewritePattern {
  ConvertCoralMatmulToLLVM(MLIRContext *context)
      : RewritePattern("coral.matmul", /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto matmulOp = cast<coral::MatMulOp>(op);
    auto ctx = getContext();

    // 1. Ensure global runtime symbol declaration exists in the parent ModuleOp
    auto module = op->getParentOfType<ModuleOp>();
    auto runtimeFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("coral_runtime_exec_matmul");

    if (!runtimeFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, ptrTy});

      runtimeFunc = rewriter.create<LLVM::LLVMFuncOp>(
          op->getLoc(), "coral_runtime_exec_matmul", funcTy);
    }

    // 2. Extract physical data pointers from MemRef descriptors
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto arrayTy = LLVM::LLVMArrayType::get(i64Ty, 2);
    auto structTy = LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, ptrTy, i64Ty, arrayTy, arrayTy});

    auto extractAlignedPtr = [&](Value memref) -> Value {
      auto cast = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), structTy, memref).getResult(0);
      return rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), ptrTy, cast, ArrayRef<int64_t>{1});
    };

    Value ptrA = extractAlignedPtr(matmulOp.getLhs());
    Value ptrB = extractAlignedPtr(matmulOp.getRhs());
    Value ptrC = extractAlignedPtr(matmulOp.getOut());

    // 3. Emit function call to host C driver
    rewriter.create<LLVM::CallOp>(
        op->getLoc(),
        runtimeFunc,
        ValueRange{ptrA, ptrB, ptrC}
    );

    rewriter.eraseOp(op);
    return success();
  }
};

// Target ModuleOp instead of func::FuncOp for module-level symbol insertions
struct CoralToLLVMPass : public PassWrapper<CoralToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CoralToLLVMPass)

  StringRef getArgument() const final { return "convert-coral-to-llvm"; }
  StringRef getDescription() const final {
    return "Lower Coral dialect ops to host C runtime driver calls";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertCoralMatmulToLLVM>(&getContext());

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