import torch
from torch import nn
from torch.utils.data import DataLoader

from bearingtel.dataset import CLASSES, WindowDataset
from bearingtel.model import BearingCNN


def main():
    torch.manual_seed(0)
    ds = WindowDataset("train")
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    x, y = next(iter(loader))
    print("CLASSES in batch:", {c: int((y == i).sum()) for i, c in enumerate(CLASSES)})


    model = BearingCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(1, 201):
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            acc = (logits.argmax(dim=1) == y).float().mean()
            print(f"step {step:3d} loss {loss.item():.3f} {acc:.2f}")


if __name__ == "__main__":
    main() 