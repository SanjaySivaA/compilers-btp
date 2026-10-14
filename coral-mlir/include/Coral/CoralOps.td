include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Coral_Dialect : Dialect {
  let name = "coral";
  let summary = "Custom dialect for the Coral NPU bare-metal accelerator.";
  let cppNamespace = "::mlir::coral";
}

class Coral_Op<string mnemonic, list<Trait> traits = []> :
    Op<Coral_Dialect, mnemonic, traits>;

def Coral_MatMulOp : Coral_Op<"matmul", [Pure]> {
  let summary = "Hardware-aware matrix multiplication for Coral NPU";
  let arguments = (ins
     AnyTypeOf<[MemRefRankOf<[I8], [1, 2]>, TensorRankOf<[I8], [1, 2]>]>:$lhs,
     AnyTypeOf<[MemRefRankOf<[I8], [1, 2]>, TensorRankOf<[I8], [1, 2]>]>:$rhs
  );
  let results = (outs
     AnyTypeOf<[MemRefRankOf<[I32], [1, 2]>, TensorRankOf<[I32], [1, 2]>]>:$result
  );
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def Coral_AddOp : Coral_Op<"add", [Pure]> {
  let summary = "Hardware-aware element-wise addition for Coral NPU";
  let arguments = (ins
     AnyTypeOf<[MemRefRankOf<[I8, I32], [1, 2]>, TensorRankOf<[I8, I32], [1, 2]>]>:$lhs,
     AnyTypeOf<[MemRefRankOf<[I8, I32], [1, 2]>, TensorRankOf<[I8, I32], [1, 2]>]>:$rhs
  );
  let results = (outs
     AnyTypeOf<[MemRefRankOf<[I8, I32], [1, 2]>, TensorRankOf<[I8, I32], [1, 2]>]>:$result
  );
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def Coral_Conv2DOp : Coral_Op<"conv2d", [Pure]> {
  let summary = "2D convolution";
  let arguments = (ins
     AnyTypeOf<[MemRefRankOf<[I8], [4]>, TensorRankOf<[I8], [4]>]>:$input,
     AnyTypeOf<[MemRefRankOf<[I8], [4]>, TensorRankOf<[I8], [4]>]>:$filter
  );
  let results = (outs
     AnyTypeOf<[MemRefRankOf<[I32], [4]>, TensorRankOf<[I32], [4]>]>:$result
  );
  let assemblyFormat = [{
    $input `,` $filter attr-dict `:` type($input) `,` type($filter) `->` type($result)
  }];
}