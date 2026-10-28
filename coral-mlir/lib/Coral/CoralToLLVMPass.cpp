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
    
    auto matmulOp = cast<coral::MatMulOp>(op);
    auto ctx = getContext();
    
    // Build the exact LLVM Struct type that MLIR uses for 2D MemRefs:
    // !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto arrayTy = LLVM::LLVMArrayType::get(i64Ty, 2);
    auto structTy = LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, ptrTy, i64Ty, arrayTy, arrayTy});

    // Helper function to cast the memref to the struct and extract the pointer at index 1
    auto extractAlignedPtr = [&](Value memref) -> Value {
        auto cast = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), structTy, memref).getResult(0);
        return rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), ptrTy, cast, ArrayRef<int64_t>{1});
    };

    // 1. Extract the physical hardware pointers from all 3 MemRefs
    Value ptrA = extractAlignedPtr(matmulOp.getLhs());
    Value ptrB = extractAlignedPtr(matmulOp.getRhs());
    Value ptrC = extractAlignedPtr(matmulOp.getOut());

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
        LLVM::LLVMVoidType::get(ctx), // Returns nothing
        ValueRange{ptrA, ptrB, ptrC}, // 3 pointers
        asmStr,
        constraints,
        /*has_side_effects=*/true,
        /*is_align_stack=*/false,
        LLVM::tailcallkind::TailCallKind::None,
        LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
        ArrayAttr()
    );

    // 4. Erase the original coral.matmul
    rewriter.eraseOp(op);

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

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace

void registerCoralToLLVMPass() {
  PassRegistration<CoralToLLVMPass>();
}