"""Model inference benchmark — measures forward-pass throughput and latency.

Tests inference with various batch sizes to find optimal throughput.

Usage: python inference_bench.py [--model resnet|transformer|mlp]
Submit: hive submit "python tests/workloads/inference_bench.py"
"""
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet", "transformer", "mlp"], default="transformer")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    # Build model
    if args.model == "mlp":
        model = nn.Sequential(
            nn.Linear(512, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, 1000),
        )
        input_fn = lambda bs: torch.randn(bs, 512, device=device)
    elif args.model == "transformer":
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        model = nn.TransformerEncoder(encoder_layer, num_layers=6)
        input_fn = lambda bs: torch.randn(bs, 128, 512, device=device)
    else:  # resnet-like
        model = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.ReLU(), nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 1000),
        )
        input_fn = lambda bs: torch.randn(bs, 3, 224, 224, device=device)

    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print()

    # Benchmark various batch sizes
    batch_sizes = [1, 4, 16, 64, 128]
    print(f"{'Batch':>6} {'Latency (ms)':>14} {'Throughput':>12}")
    print("-" * 36)

    for bs in batch_sizes:
        try:
            x = input_fn(bs)
            # Warmup
            with torch.no_grad():
                for _ in range(args.warmup):
                    model(x)
            torch.cuda.synchronize()

            # Measure
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(args.iters):
                    model(x)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            latency_ms = elapsed / args.iters * 1000
            throughput = bs * args.iters / elapsed
            print(f"{bs:>6} {latency_ms:>12.2f}ms {throughput:>10.0f}/s")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{bs:>6} {'OOM':>14}")
                torch.cuda.empty_cache()
            else:
                raise

    print()
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("Inference benchmark complete!")


if __name__ == "__main__":
    main()
