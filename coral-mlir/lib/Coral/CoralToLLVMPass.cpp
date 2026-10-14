#include "Coral/CoralDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace {

struct ConvertCoralMatmulToLLVM : public RewritePattern {
  ConvertCoralMatmulToLLVM(MLIRContext *context)
      : RewritePattern("coral.matmul", /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    
    auto llvmVoidTy = LLVM::LLVMVoidType::get(getContext());
    auto llvmPtrTy = LLVM::LLVMPointerType::get(getContext());

    StringRef asmStr = 
        "vsetvli zero, 16, e8, m1, ta, ma\n\t"
        "vle8.v v8, ($0)\n\t"
        "vle8.v v16, ($1)\n\t"
        "vwmacc.vv v24, v8, v16";
    StringRef constraints = "r,r";

    auto ptrA = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), llvmPtrTy, op->getOperand(0));
    auto ptrB = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), llvmPtrTy, op->getOperand(1));

    rewriter.create<LLVM::InlineAsmOp>(
        op->getLoc(),
        llvmVoidTy,
        ValueRange{ptrA.getResult(0), ptrB.getResult(0)}, 
        asmStr,
        constraints,
        true, 
        false, 
        mlir::LLVM::tailcallkind::TailCallKind::None,
        LLVM::AsmDialectAttr::get(getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr()
    );

    auto dummyCast = rewriter.create<UnrealizedConversionCastOp>(
        op->getLoc(),
        TypeRange{op->getResult(0).getType()},
        ValueRange{}
    );

    rewriter.replaceOp(op, dummyCast.getOutputs());

    return success();
  }
};

struct ConvertCoralAddToLLVM : public RewritePattern {
  ConvertCoralAddToLLVM(MLIRContext *context)
      : RewritePattern("coral.add", /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    
    auto llvmVoidTy = LLVM::LLVMVoidType::get(getContext());
    auto llvmPtrTy = LLVM::LLVMPointerType::get(getContext());

    StringRef asmStr = 
        "vsetvli zero, 16, e8, m1, ta, ma\n\t"
        "vle8.v v8, ($0)\n\t"
        "vle8.v v16, ($1)\n\t"
        "vadd.vv v24, v8, v16";
    StringRef constraints = "r,r";

    auto ptrA = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), llvmPtrTy, op->getOperand(0));
    auto ptrB = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), llvmPtrTy, op->getOperand(1));

    rewriter.create<LLVM::InlineAsmOp>(
        op->getLoc(),
        llvmVoidTy,
        ValueRange{ptrA.getResult(0), ptrB.getResult(0)}, 
        asmStr,
        constraints,
        true, 
        false, 
        mlir::LLVM::tailcallkind::TailCallKind::None,
        LLVM::AsmDialectAttr::get(getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr()
    );

    auto dummyCast = rewriter.create<UnrealizedConversionCastOp>(
        op->getLoc(),
        TypeRange{op->getResult(0).getType()},
        ValueRange{}
    );

    rewriter.replaceOp(op, dummyCast.getOutputs());

    return success();
  }
};

struct ConvertCoralConv2DToLLVM : public RewritePattern {
  ConvertCoralConv2DToLLVM(MLIRContext *context)
      : RewritePattern("coral.conv2d", /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    
    auto llvmVoidTy = LLVM::LLVMVoidType::get(getContext());
    auto llvmPtrTy = LLVM::LLVMPointerType::get(getContext());

    // Basic 2D convolution RISC-V vector snippet (simplified MAC sequence)
    StringRef asmStr = 
        "vsetvli zero, 16, e8, m1, ta, ma\n\t"
        "vle8.v v8, ($0)\n\t"        // Load input patch
        "vle8.v v16, ($1)\n\t"       // Load filter weights
        "vwmacc.vv v24, v8, v16";    // Vector widening multiply-accumulate
    StringRef constraints = "r,r";

    auto ptrInput = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), llvmPtrTy, op->getOperand(0));
    auto ptrFilter = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), llvmPtrTy, op->getOperand(1));

    rewriter.create<LLVM::InlineAsmOp>(
        op->getLoc(),
        llvmVoidTy,
        ValueRange{ptrInput.getResult(0), ptrFilter.getResult(0)}, 
        asmStr,
        constraints,
        true, 
        false, 
        mlir::LLVM::tailcallkind::TailCallKind::None,
        LLVM::AsmDialectAttr::get(getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr()
    );

    auto dummyCast = rewriter.create<UnrealizedConversionCastOp>(
        op->getLoc(),
        TypeRange{op->getResult(0).getType()},
        ValueRange{}
    );

    rewriter.replaceOp(op, dummyCast.getOutputs());

    return success();
  }
};

struct CoralToLLVMPass : public PassWrapper<CoralToLLVMPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CoralToLLVMPass)

  StringRef getArgument() const final { return "convert-coral-to-llvm"; }
  StringRef getDescription() const final { return "Lower coral ops to RISC-V LLVM Inline Assembly"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertCoralMatmulToLLVM>(&getContext());
    patterns.add<ConvertCoralAddToLLVM>(&getContext());
    patterns.add<ConvertCoralConv2DToLLVM>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace

void registerCoralToLLVMPass() {
  PassRegistration<CoralToLLVMPass>();
}