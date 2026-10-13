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
    
    // Get the LLVM Pointer Type (void or llvm.ptr)
    auto llvmVoidTy = LLVM::LLVMVoidType::get(getContext());
    auto llvmPtrTy = LLVM::LLVMPointerType::get(getContext());

    // Define the RISC-V assembly code and constraints
    StringRef asmStr = 
        "vsetvli zero, 16, e8, m1, ta, ma\n\t"
        "vle8.v v8, ($0)\n\t"
        "vle8.v v16, ($1)\n\t"
        "vwmacc.vv v24, v8, v16";

    StringRef constraints = "r,r";

    // Cast the abstract PyTorch tensors into bare-metal LLVM memory pointers
    auto ptrA = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), llvmPtrTy, op->getOperand(0));
    auto ptrB = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), llvmPtrTy, op->getOperand(1));

    // Create the Inline Assembly and feeding it the pointers
    rewriter.create<LLVM::InlineAsmOp>(
        op->getLoc(),
        llvmVoidTy,
        ValueRange{ptrA.getResult(0), ptrB.getResult(0)}, 
        asmStr,
        constraints,
        true, // has_side_effects = true
        false, // is_align_stack = false
        mlir::LLVM::tailcallkind::TailCallKind::None,
        LLVM::AsmDialectAttr::get(getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr()
    );

    // Create the dummy output tensor to satisfy the 'return' statement
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

  // Connects the pass to your terminal
  StringRef getArgument() const final { return "convert-coral-to-llvm"; }
  StringRef getDescription() const final { return "Lower coral.matmul to RISC-V LLVM Inline Assembly"; }

  // Forces the MLIR engine to load the LLVM infrastructure into memory before running the pass
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  // Runs the pattern matcher with ConvertCoralMatmulToLLVM logic on file
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertCoralMatmulToLLVM>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} 

// Registers the pass
void registerCoralToLLVMPass() {
  PassRegistration<CoralToLLVMPass>();
}