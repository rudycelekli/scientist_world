"""Gradient accumulation training — simulates large batch training on limited GPU memory.

Usage: python grad_accumulation.py [--epochs 10] [--accum-steps 8]
Submit: hive submit "python tests/workloads/grad_accumulation.py"
"""
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--accum-steps", type=int, default=8,
                        help="Gradient accumulation steps (effective batch = micro * accum)")
    parser.add_argument("--hidden", type=int, default=1024)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_batch = args.micro_batch * args.accum_steps
    print(f"Device: {device}")
    print(f"Micro-batch: {args.micro_batch}, Accum steps: {args.accum_steps}, "
          f"Effective batch: {effective_batch}")

    model = nn.Sequential(
        nn.Linear(512, args.hidden), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(args.hidden, args.hidden), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(args.hidden, args.hidden), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(args.hidden, 10),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    n_samples = 4000
    data = torch.randn(n_samples, 512, device=device)
    labels = torch.randint(0, 10, (n_samples,), device=device)

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        total_loss = 0.0
        optimizer_steps = 0
        model.train()
        optimizer.zero_grad()

        for i in range(0, n_samples, args.micro_batch):
            x = data[i:i + args.micro_batch]
            y = labels[i:i + args.micro_batch]
            loss = criterion(model(x), y) / args.accum_steps
            loss.backward()
            total_loss += loss.item() * args.accum_steps

            micro_step = (i // args.micro_batch) + 1
            if micro_step % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_steps += 1

        elapsed = time.perf_counter() - t0
        avg_loss = total_loss / (n_samples // args.micro_batch)
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f}, "
              f"optimizer_steps={optimizer_steps}, {elapsed:.3f}s")

    print("Gradient accumulation training complete!")


if __name__ == "__main__":
    main()
