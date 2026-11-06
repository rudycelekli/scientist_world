"""Checkpoint save/load test — trains a model, saves checkpoint, reloads, continues.

Tests filesystem I/O from GPU jobs and verifies model state persistence.

Usage: python checkpoint_io.py [--epochs 5]
Submit: hive submit "python tests/workloads/checkpoint_io.py"
"""
import argparse
import os
import tempfile
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

    def make_model():
        return nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 10),
        ).to(device)

    model = make_model()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    data = torch.randn(1000, 256, device=device)
    labels = torch.randint(0, 10, (1000,), device=device)

    # Phase 1: Train for some epochs
    print(f"Phase 1: Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        total_loss = 0.0
        n = 0
        for i in range(0, len(data), args.batch_size):
            x, y = data[i:i + args.batch_size], labels[i:i + args.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        print(f"  Epoch {epoch + 1}: loss={total_loss / n:.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(tempfile.gettempdir(), "hive_test_checkpoint.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.epochs,
    }, ckpt_path)
    ckpt_size = os.path.getsize(ckpt_path) / 1e6
    print(f"Saved checkpoint: {ckpt_path} ({ckpt_size:.1f} MB)")

    # Phase 2: Load checkpoint into fresh model and continue
    model2 = make_model()
    optimizer2 = optim.SGD(model2.parameters(), lr=0.01)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model2.load_state_dict(checkpoint["model_state_dict"])
    optimizer2.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Verify: outputs should match
    with torch.no_grad():
        out1 = model(data[:10])
        out2 = model2(data[:10])
        max_diff = (out1 - out2).abs().max().item()
        print(f"Max output difference after reload: {max_diff:.2e}")
        assert max_diff < 1e-5, f"Checkpoint mismatch: {max_diff}"

    # Continue training
    print(f"Phase 2: Continue training for {args.epochs} more epochs...")
    for epoch in range(args.epochs):
        total_loss = 0.0
        n = 0
        for i in range(0, len(data), args.batch_size):
            x, y = data[i:i + args.batch_size], labels[i:i + args.batch_size]
            optimizer2.zero_grad()
            loss = criterion(model2(x), y)
            loss.backward()
            optimizer2.step()
            total_loss += loss.item()
            n += 1
        print(f"  Epoch {args.epochs + epoch + 1}: loss={total_loss / n:.4f}")

    # Cleanup
    os.remove(ckpt_path)
    print("Checkpoint I/O test complete!")


if __name__ == "__main__":
    main()
