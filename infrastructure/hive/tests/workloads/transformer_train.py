"""Train a small Transformer on random sequence data.

Usage: python transformer_train.py [--epochs 10] [--d-model 256] [--nhead 8]
Submit: hive submit "python tests/workloads/transformer_train.py"
"""
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=1000)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Transformer language model
    class TransformerLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(args.vocab_size, args.d_model)
            self.pos = nn.Embedding(args.seq_len, args.d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=args.d_model, nhead=args.nhead,
                dim_feedforward=args.d_model * 4, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers)
            self.head = nn.Linear(args.d_model, args.vocab_size)

        def forward(self, x):
            pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            x = self.embed(x) + self.pos(pos_ids)
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            x = self.transformer(x, mask=mask)
            return self.head(x)

    model = TransformerLM().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.num_layers}L, d={args.d_model}, heads={args.nhead}, "
          f"{n_params:,} params")

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    # Random token sequences
    n_samples = 2000
    data = torch.randint(0, args.vocab_size, (n_samples, args.seq_len), device=device)

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n_samples, args.batch_size):
            x = data[i:i + args.batch_size]
            logits = model(x[:, :-1])
            loss = criterion(logits.reshape(-1, args.vocab_size), x[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        elapsed = time.perf_counter() - t0
        tokens_per_sec = n_samples * args.seq_len / elapsed
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={total_loss / n_batches:.4f}, "
              f"{elapsed:.2f}s, {tokens_per_sec:.0f} tok/s")

    print("Transformer training complete!")


if __name__ == "__main__":
    main()
