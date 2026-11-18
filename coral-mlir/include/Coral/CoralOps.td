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
  let arguments = (ins AnyShaped:$lhs, AnyShaped:$rhs, AnyShaped:$out);
  let results = (outs Optional<AnyTensor>:$result);
}

def Coral_AddOp : Coral_Op<"add"> {
  let summary = "Hardware-aware element-wise addition for Coral NPU";
  let arguments = (ins AnyShaped:$lhs, AnyShaped:$rhs, Optional<AnyShaped>:$out);
  let results = (outs Optional<AnyTensor>:$result);
}

def Coral_Conv2DOp : Coral_Op<"conv2d"> {
  let summary = "2D convolution for Coral NPU";
  let arguments = (ins
    AnyShaped:$input,
    AnyShaped:$filter,
    AnyShaped:$out,
    OptionalAttr<AnyAttr>:$strides,
    OptionalAttr<AnyAttr>:$dilations
  );
  let results = (outs Optional<AnyTensor>:$result);
}

def Coral_BnReluOp : Coral_Op<"bn_relu"> {
  let summary = "Fused BatchNorm + ReLU for Coral NPU";
  let arguments = (ins
    AnyShaped:$input,
    AnyShaped:$mean,
    AnyShaped:$scale,
    AnyShaped:$bias,
    AnyShaped:$out
  );
  let results = (outs Optional<AnyTensor>:$result);
}

def Coral_AddReluOp : Coral_Op<"add_relu"> {
  let summary = "Fused residual Add + ReLU for Coral NPU";
  let arguments = (ins AnyShaped:$lhs, AnyShaped:$rhs, AnyShaped:$out);
  let results = (outs Optional<AnyTensor>:$result);
}

def Coral_DmaPadOp : Coral_Op<"dma_pad"> {
  let summary = "Hardware DMA tensor padding for Coral NPU";
  let arguments = (ins
    AnyShaped:$source,
    Optional<AnyShaped>:$out,
    OptionalAttr<I64ArrayAttr>:$low_pad,
    OptionalAttr<I64ArrayAttr>:$high_pad
  );
  let results = (outs Optional<AnyTensor>:$result);
}