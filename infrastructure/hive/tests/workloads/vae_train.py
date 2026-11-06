"""Variational Autoencoder (VAE) training — tests encoder-decoder with KL loss.

Usage: python vae_train.py [--epochs 15] [--latent-dim 64]
Submit: hive submit "python tests/workloads/vae_train.py"
"""
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=512)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    input_dim = 784  # 28x28 flattened

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, args.hidden), nn.ReLU(),
                nn.Linear(args.hidden, args.hidden), nn.ReLU(),
            )
            self.fc_mu = nn.Linear(args.hidden, args.latent_dim)
            self.fc_logvar = nn.Linear(args.hidden, args.latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(args.latent_dim, args.hidden), nn.ReLU(),
                nn.Linear(args.hidden, args.hidden), nn.ReLU(),
                nn.Linear(args.hidden, input_dim), nn.Sigmoid(),
            )

        def encode(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

    model = VAE().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"VAE: {n_params:,} params, latent_dim={args.latent_dim}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Random data (simulating MNIST-like input)
    n_samples = 5000
    data = torch.rand(n_samples, input_dim, device=device)

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        total_recon, total_kl = 0.0, 0.0
        n_batches = 0
        for i in range(0, n_samples, args.batch_size):
            x = data[i:i + args.batch_size]
            recon, mu, logvar = model(x)
            recon_loss = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            loss = recon_loss + kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1
        elapsed = time.perf_counter() - t0
        print(f"Epoch {epoch + 1}/{args.epochs}: "
              f"recon={total_recon / n_batches:.2f}, kl={total_kl / n_batches:.2f}, "
              f"{elapsed:.3f}s")

    # Test generation
    with torch.no_grad():
        z = torch.randn(16, args.latent_dim, device=device)
        generated = model.decode(z)
        print(f"Generated 16 samples, shape={generated.shape}, "
              f"range=[{generated.min():.3f}, {generated.max():.3f}]")

    print("VAE training complete!")


if __name__ == "__main__":
    main()
