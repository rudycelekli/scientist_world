"""Mixed-precision (AMP) training — tests FP16/BF16 automatic mixed precision.

Usage: python amp_training.py [--epochs 10] [--dtype float16|bfloat16]
Submit: hive submit "python tests/workloads/amp_training.py"
"""
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    print(f"Device: {device}, AMP dtype: {args.dtype}")

    # Deep MLP
    model = nn.Sequential(
        nn.Linear(512, args.hidden), nn.GELU(),
        nn.Linear(args.hidden, args.hidden), nn.GELU(),
        nn.Linear(args.hidden, args.hidden), nn.GELU(),
        nn.Linear(args.hidden, args.hidden), nn.GELU(),
        nn.Linear(args.hidden, 10),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, {n_params * 4 / 1e6:.1f} MB (FP32)")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(args.dtype == "float16"))

    n_samples = 5000
    data = torch.randn(n_samples, 512, device=device)
    labels = torch.randint(0, 10, (n_samples,), device=device)

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n_samples, args.batch_size):
            x = data[i:i + args.batch_size]
            y = labels[i:i + args.batch_size]
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1
        elapsed = time.perf_counter() - t0
        samples_per_sec = n_samples / elapsed
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={total_loss / n_batches:.4f}, "
              f"{elapsed:.3f}s, {samples_per_sec:.0f} samples/s")

    print("AMP training complete!")


if __name__ == "__main__":
    main()
