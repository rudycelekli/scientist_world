"""Push the built parquet + dataset card to the Hugging Face Hub.

Usage:
    HF_TOKEN=... python -m dataset_builder.push_to_hf --repo <user-or-org>/<dataset-name>

The target repository comes from ``--repo`` or ``$SW_TRACES_REPO``; there
is no default. The token is read from ``$HF_TOKEN``, then
``$HUGGING_FACE_HUB_TOKEN``, then the cached ``huggingface-cli login``.
The repository is created if it does not exist; ``--private`` (the
default) and ``--no-private`` only apply at creation. The parquet is
uploaded as the dataset (one split per mode) and ``DATASET_CARD.md`` as
its ``README.md``.

Requires the ``datasets`` and ``huggingface_hub`` packages.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_PARQUET = _HERE / "out" / "scientist_world_traces.parquet"
_CARD = _HERE / "DATASET_CARD.md"


def _token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--repo", default=os.environ.get("SW_TRACES_REPO"),
        help="dataset repository id, <user-or-org>/<name> (default: $SW_TRACES_REPO)",
    )
    ap.add_argument("--parquet", default=str(_DEFAULT_PARQUET))
    ap.add_argument(
        "--private", action=argparse.BooleanOptionalAction, default=True,
        help="create the repository as private (default) or public; "
             "ignored when it already exists",
    )
    args = ap.parse_args(argv)
    if not args.repo:
        ap.error("no target repository: pass --repo or set SW_TRACES_REPO")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    token = _token()

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    # whoami sanity check (raises if token is missing/invalid)
    who = api.whoami()
    print(f"authenticated as: {who.get('name')}")

    parquet = Path(args.parquet)
    if not parquet.exists():
        sys.exit(f"parquet not found: {parquet} (run build_dataset first)")

    # Load, then split by mode so innovate / improve show as separate tabs
    # (splits) in the HF dataset viewer.
    from datasets import load_dataset, DatasetDict
    ds = load_dataset("parquet", data_files=str(parquet), split="train")
    dd = DatasetDict({
        "innovate": ds.filter(lambda r: r["mode"] == "innovate"),
        "improve": ds.filter(lambda r: r["mode"] == "improve"),
    })
    print(f"pushing -> {args.repo}: "
          f"innovate={len(dd['innovate'])} improve={len(dd['improve'])}")
    dd.push_to_hub(args.repo, token=token, private=args.private)

    # Upload the dataset card as README.md.
    if _CARD.exists():
        api.upload_file(
            path_or_fileobj=str(_CARD),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
            token=token,
        )
        print("uploaded dataset card (README.md)")

    print(f"DONE: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
