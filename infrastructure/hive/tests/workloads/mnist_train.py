"""Train a small CNN on random data — tests full training pipeline.

Usage: python mnist_train.py [--epochs 5] [--batch-size 64]
Submit: hive submit "python tests/workloads/mnist_train.py"
"""
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Simple CNN
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Random data (no download needed)
    n_samples = 1000
    images = torch.randn(n_samples, 1, 28, 28, device=device)
    labels = torch.randint(0, 10, (n_samples,), device=device)

    print(f"Training {sum(p.numel() for p in model.parameters())} params "
          f"for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n_samples, args.batch_size):
            x = images[i:i+args.batch_size]
            y = labels[i:i+args.batch_size]
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        elapsed = time.perf_counter() - t0
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"loss={total_loss/n_batches:.4f}, {elapsed:.2f}s")

    print("Training complete!")

if __name__ == "__main__":
    main()
