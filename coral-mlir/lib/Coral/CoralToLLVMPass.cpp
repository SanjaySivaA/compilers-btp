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
    
    // Cast the generic operation into our strongly-typed MatMulOp
    auto matmulOp = cast<coral::MatMulOp>(op);
    auto llvmPtrTy = LLVM::LLVMPointerType::get(getContext());

    // 1. Cast all 3 physical MemRefs into raw LLVM hardware pointers
    auto ptrA = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), llvmPtrTy, matmulOp.getLhs()).getResult(0);
    auto ptrB = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), llvmPtrTy, matmulOp.getRhs()).getResult(0);
    auto ptrC = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), llvmPtrTy, matmulOp.getOut()).getResult(0);

    // 2. The RISC-V Assembly String
    llvm::StringRef asmStr = 
        "vsetivli zero, 16, e8, m1, ta, ma\n\t"
        "vle8.v v8, ($0)\n\t"
        "vle8.v v16, ($1)\n\t"
        "vwmacc.vv v24, v8, v16\n\t"
        "vse32.v v24, ($2)";
    
    llvm::StringRef constraints = "r,r,r";

    // 3. Generate the inline assembly block in the IR
    rewriter.create<LLVM::InlineAsmOp>(
        op->getLoc(),
        LLVM::LLVMVoidType::get(getContext()), // Returns nothing
        ValueRange{ptrA, ptrB, ptrC},          // 3 pointers
        asmStr,
        constraints,
        /*has_side_effects=*/true,
        /*is_align_stack=*/false,
        LLVM::tailcallkind::TailCallKind::None, // <-- THE FIX FOR THE 11TH ARGUMENT
        LLVM::AsmDialectAttr::get(getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr()
    );

    // 4. Erase the original coral.matmul
    rewriter.eraseOp(op);

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