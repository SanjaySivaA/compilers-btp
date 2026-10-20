include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Coral_Dialect : Dialect {
  let name = "coral";
  let summary = "Custom dialect for the Coral NPU bare-metal accelerator.";
  let cppNamespace = "::mlir::coral";
}

class Coral_Op<string mnemonic, list<Trait> traits = []> :
    Op<Coral_Dialect, mnemonic, traits>;

def Coral_MatMulOp : Coral_Op<"matmul"> {
  let summary = "Hardware-aware matrix multiplication for Coral NPU";
  
  let arguments = (ins 
    AnyMemRef:$lhs,
    AnyMemRef:$rhs,
    AnyMemRef:$out
  );

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $out attr-dict `:` type($lhs) `,` type($rhs) `,` type($out)
  }];
}
