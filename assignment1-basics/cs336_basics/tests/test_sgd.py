from cs336_basics.optimize import optimizers
import torch

def main(lr: float = 1, epochs: int = 100):
    print(f"train with lr={lr}, epochs={epochs}")
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = optimizers.SGD([weights], lr=lr)
    for t in range(epochs):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(f"loss: {loss.cpu().item()}")
        loss.backward()
        opt.step()

if __name__ == "__main__":
    main(1, 10)
    main(1e1, 10)
    main(1e2, 10)
    main(1e3, 10)