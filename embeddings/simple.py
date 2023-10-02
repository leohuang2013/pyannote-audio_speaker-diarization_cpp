import torch
import torch.onnx.verification
torch.manual_seed(0)
opset_version = 15

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 6),
        )
    def forward(self, x):
        return self.layers(x)

model = Model()
graph_info = torch.onnx.verification.find_mismatch(
    model,
    (torch.randn(2, 3),),
    opset_version=opset_version,
)

torch.onnx.export(
    model,
    (torch.randn(2, 3),),
    'simple.onnx',
    opset_version=opset_version,
    do_constant_folding = True,
    verbose = True
)

print("\n=============")
print("exported")
