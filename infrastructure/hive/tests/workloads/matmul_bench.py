"""GPU matmul benchmark — verifies CUDA works and measures throughput.

Usage: python matmul_bench.py [--size 4096] [--iters 100]
Submit: hive submit "python tests/workloads/matmul_bench.py"
"""
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("CUDA not available — skipping GPU benchmark")
        return

    device = torch.device("cuda")
    n = args.size
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Running {args.iters} matmuls of {n}x{n}...")

    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    # Warmup
    for _ in range(5):
        torch.mm(a, b)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(args.iters):
        torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tflops = (2 * n**3 * args.iters) / elapsed / 1e12
    print(f"Elapsed: {elapsed:.2f}s ({elapsed/args.iters*1000:.1f} ms/iter)")
    print(f"Throughput: {tflops:.2f} TFLOPS")

if __name__ == "__main__":
    main()
