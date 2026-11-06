"""Minimal sleep job — for testing async job submission and output capture.

Usage: python sleep_job.py [--seconds 10]
Submit: hive submit "python tests/workloads/sleep_job.py --seconds 5"
"""
import argparse
import sys
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--fail", action="store_true", help="Exit with error")
    args = parser.parse_args()

    print(f"Sleep job started (pid={__import__('os').getpid()})")
    print(f"Sleeping for {args.seconds} seconds...")
    time.sleep(args.seconds)

    if args.fail:
        print("Intentional failure!", file=sys.stderr)
        sys.exit(1)

    print(f"Sleep job completed after {args.seconds}s")

if __name__ == "__main__":
    main()
