import torch
from cs336_basics.network.multiplatform import ACCL_DEVICE, ACCL_TYPE
from torch import nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        for components in ("fc1", "ln", "fc2"):
            module = getattr(self, components)
            print(f"dtype of {components} weights: {module.weight.dtype}")
        x = self.fc1(x)
        print(f"dtype after fc1: {x.dtype}")
        x = self.relu(x)
        print(f"dtype after ReLU: {x.dtype}")
        x = self.ln(x)
        print(f"dtype after LayerNorm: {x.dtype}")
        x = self.fc2(x)
        print(f"dtype after fc2: {x.dtype}")
        return x


if __name__ == "__main__":
    model = ToyModel(20, 5).to(ACCL_DEVICE)
    with torch.autocast(device_type=ACCL_TYPE, dtype=torch.bfloat16):
        x = torch.randn(2, 20).to(ACCL_DEVICE)
        print(f"dtype of x: {x.dtype}")
        y = model(x)
        print(f"dtype of output y: {y.dtype}")
        y_hat = torch.randn(2, 5).to(ACCL_DEVICE)
        loss = nn.functional.mse_loss(y, y_hat)
        print(f"dtype of loss: {loss.dtype}")
        loss.backward()
        print(
            f"dtype of gradients for fc1 weights: {model.fc1.weight.grad.dtype if model.fc1.weight.grad is not None else 'None'}"
        )
