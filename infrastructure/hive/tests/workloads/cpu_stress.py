"""CPU-only stress test — no GPU required.

Usage: python cpu_stress.py [--seconds 10] [--threads 4]
Submit: hive submit "python tests/workloads/cpu_stress.py --seconds 10" --gpus 0 --cpus 4
"""
import argparse
import math
import os
import threading
import time


def _cpu_load(duration):
    """Busy loop doing math to peg one core."""
    end = time.monotonic() + duration
    x = 1.0
    while time.monotonic() < end:
        for _ in range(1000):
            x = math.sin(x) + math.cos(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    print(f"CPU stress test started (pid={os.getpid()})")
    print(f"Running {args.threads} threads for {args.seconds}s")

    threads = []
    for i in range(args.threads):
        t = threading.Thread(target=_cpu_load, args=(args.seconds,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"CPU stress test completed after {args.seconds}s")


if __name__ == "__main__":
    main()
