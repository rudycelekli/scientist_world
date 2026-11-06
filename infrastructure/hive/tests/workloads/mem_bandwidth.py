"""GPU memory bandwidth test — measures host-to-device and device-to-device throughput.

Usage: python mem_bandwidth.py [--size-gb 2] [--iters 20]
Submit: hive submit "python tests/workloads/mem_bandwidth.py"
"""
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-gb", type=float, default=2.0, help="Tensor size in GB")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")
    n_elements = int(args.size_gb * 1e9 / 4)  # float32
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Tensor size: {args.size_gb:.1f} GB ({n_elements:,} float32 elements)")
    print(f"Iterations: {args.iters}")
    print()

    # Host-to-Device
    cpu_tensor = torch.randn(n_elements)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        gpu_tensor = cpu_tensor.to(device)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    bw = args.size_gb * args.iters / elapsed
    print(f"Host -> Device: {bw:.2f} GB/s ({elapsed / args.iters * 1000:.1f} ms/transfer)")

    # Device-to-Host
    gpu_tensor = torch.randn(n_elements, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        cpu_out = gpu_tensor.cpu()
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    bw = args.size_gb * args.iters / elapsed
    print(f"Device -> Host: {bw:.2f} GB/s ({elapsed / args.iters * 1000:.1f} ms/transfer)")

    # Device-to-Device copy
    src = torch.randn(n_elements, device=device)
    dst = torch.empty(n_elements, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        dst.copy_(src)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    bw = args.size_gb * args.iters / elapsed
    print(f"Device -> Device: {bw:.2f} GB/s ({elapsed / args.iters * 1000:.1f} ms/transfer)")

    # Allocation stress
    print()
    print("Allocation stress (10 rounds, allocate+free 4GB)...")
    for r in range(10):
        blocks = []
        for _ in range(4):
            blocks.append(torch.randn(int(1e9 / 4), device=device))
        del blocks
        torch.cuda.empty_cache()
    print(f"  Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print("Memory bandwidth test complete!")


if __name__ == "__main__":
    main()
