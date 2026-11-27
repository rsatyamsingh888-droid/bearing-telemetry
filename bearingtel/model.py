import torch
from torch import nn


def block(in_ch, out_ch, kernel, stride=1):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),
    )


class BearingCNN(nn.Module):
    def __init__(self, in_channels=2, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            block(in_channels, 16, kernel=64, stride=8),
            block(16, 32, kernel=3),
            block(32, 64, kernel=3),
            block(64, 64, kernel=3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(64, num_classes)

    def forward(self,x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)



if __name__ == "__main__":
    model = BearingCNN()
    x = torch.randn(8, 2, 4096)
    print("input: ", x.shape)
    for layer in model.features:
        x = layer(x)
        print("after block:", x.shape)
        print("output:", model(torch.randn(8, 2, 4096)).shape)
        print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
        


