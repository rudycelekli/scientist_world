"""ResNet-like CNN training on random image data.

Tests convolutional layers, batch norm, residual connections, and image pipeline.

Usage: python conv_resnet.py [--epochs 10] [--batch-size 32]
Submit: hive submit "python tests/workloads/conv_resnet.py"
"""
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=100)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    class ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU()

        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.relu(out + x)

    class MiniResNet(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(),
            )
            self.layer1 = nn.Sequential(ResBlock(64), ResBlock(64))
            self.down1 = nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(),
            )
            self.layer2 = nn.Sequential(ResBlock(128), ResBlock(128))
            self.down2 = nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(),
            )
            self.layer3 = nn.Sequential(ResBlock(256), ResBlock(256))
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x)
            x = self.down1(x)
            x = self.layer2(x)
            x = self.down2(x)
            x = self.layer3(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)

    model = MiniResNet(args.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MiniResNet: {n_params:,} params, {args.num_classes} classes")

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # Random 32x32 images (CIFAR-like)
    n_samples = 2000
    images = torch.randn(n_samples, 3, 32, 32, device=device)
    labels = torch.randint(0, args.num_classes, (n_samples,), device=device)

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for i in range(0, n_samples, args.batch_size):
            x = images[i:i + args.batch_size]
            y = labels[i:i + args.batch_size]
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()
        elapsed = time.perf_counter() - t0
        acc = 100.0 * correct / total
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={total_loss / (n_samples // args.batch_size):.4f}, "
              f"acc={acc:.1f}%, lr={lr:.4f}, {elapsed:.2f}s")

    print("ResNet training complete!")


if __name__ == "__main__":
    main()
