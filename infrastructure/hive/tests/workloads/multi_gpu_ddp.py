"""Multi-GPU DDP training — tests multi-GPU job submission.

Usage: torchrun --nproc_per_node=2 multi_gpu_ddp.py
Submit: hive submit "torchrun --nproc_per_node=2 tests/workloads/multi_gpu_ddp.py" --gpus 2
"""
import os
import time

def main():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"World size: {dist.get_world_size()}")
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # Simple model
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10),
    ).to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Random data
    n_samples = 2000
    data = torch.randn(n_samples, 512, device=device)
    labels = torch.randint(0, 10, (n_samples,), device=device)
    batch_size = 128

    for epoch in range(3):
        t0 = time.perf_counter()
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n_samples, batch_size):
            x = data[i:i+batch_size]
            y = labels[i:i+batch_size]
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if rank == 0:
            elapsed = time.perf_counter() - t0
            print(f"Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, {elapsed:.2f}s")

    if rank == 0:
        print("DDP training complete!")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
