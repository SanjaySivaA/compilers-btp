import torchvision.models as models
import torch_mlir
from torch_mlir.fx import OutputType

# Load a standard ResNet18 model
model = models.resnet18(weights=None)
model.eval()

# ResNet expects 3-channel RGB images of at least 224x224
example_input = torch.randn(1, 3, 224, 224)

module = torch_mlir.fx.export_and_import(
    model,
    example_input,
    output_type=OutputType.LINALG_ON_TENSORS
)

with open("resnet_linalg.mlir", "w") as f:
    f.write(str(module))

print("Successfully exported to resnet_linalg.mlir")

# The export_and_import call remains exactly the same