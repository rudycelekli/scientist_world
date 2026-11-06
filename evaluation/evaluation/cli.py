"""Command-line interface (``sw-eval`` or ``python -m evaluation.cli``).

A thin layer over the Python API in :mod:`evaluation`.
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sw-eval",
        description="Evaluation harness for Scientist World environments.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run one (paper, profile, mode) trial")
    p_run.add_argument("paper", help="paper id of a built environment")
    p_run.add_argument("model", help="profile name from the config")
    p_run.add_argument("--mode", default="innovate", choices=["innovate", "improve"])

    p_bench = sub.add_parser(
        "bench", help="run a matrix of papers x profiles x modes and wait",
    )
    p_bench.add_argument("--papers", required=True, help="comma-separated paper ids")
    p_bench.add_argument("--models", required=True, help="comma-separated profile names")
    p_bench.add_argument(
        "--modes", default="innovate",
        help="comma-separated modes (innovate, improve)",
    )

    p_serve = sub.add_parser("serve", help="run the dashboard and the run queue")
    p_serve.add_argument(
        "--host", default=None,
        help="bind address (default: config key `host`, 127.0.0.1)",
    )
    p_serve.add_argument(
        "--port", type=int, default=None,
        help="port (default: config key `port`)",
    )

    p_seed = sub.add_parser(
        "seed",
        help="write a config from the defaults plus the profiles and nexus "
             "block of a local Swarm config",
    )
    p_seed.add_argument(
        "--overwrite", action="store_true",
        help="replace existing keys (default: keep existing)",
    )
    p_seed.add_argument(
        "--swarm-config", default=None,
        help="path of the Swarm config.yaml to read (default: "
             "infrastructure/swarm/swarm/config/config.yaml of this repository)",
    )

    p_warm = sub.add_parser(
        "warm", help="populate the per-paper cache (data download + container build)"
    )
    p_warm.add_argument("paper")
    p_warm.add_argument(
        "--force", action="store_true",
        help="rebuild cache even if already populated",
    )

    p_norm = sub.add_parser(
        "renormalize",
        help="recompute the per-paper normalization coefficients and the global summary",
    )
    p_norm.add_argument(
        "--paper", default=None,
        help="recompute one paper only (default: all papers)",
    )

    p_pub = sub.add_parser(
        "publish",
        help="push completed runs to a branch of their per-paper repository "
             "(requires publish.enabled)",
    )
    p_pub.add_argument(
        "--paper", default=None,
        help="filter by paper id (default: all completed)",
    )
    p_pub.add_argument(
        "--profile", default=None,
        help="filter by profile name (default: all completed)",
    )
    p_pub.add_argument(
        "--mode", default=None, choices=["innovate", "improve"],
        help="filter by mode (default: all)",
    )
    p_pub.add_argument(
        "--dry-run", action="store_true",
        help="report what would be pushed without contacting GitHub",
    )
    p_pub.add_argument(
        "--limit", type=int, default=0,
        help="cap how many runs to publish (default: 0 = unlimited)",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from evaluation.config import ConfigError
    try:
        return _dispatch(args, parser)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.cmd == "run":
        from evaluation.runner import run_one
        r = run_one(args.paper, args.model, mode=args.mode)
        print(r)
        return 0 if r.status == "completed" else 1

    if args.cmd == "bench":
        from evaluation.orchestrator import bench, wait_all
        papers = [s.strip() for s in args.papers.split(",") if s.strip()]
        models = [s.strip() for s in args.models.split(",") if s.strip()]
        modes = [s.strip() for s in args.modes.split(",") if s.strip()]
        bench(papers, models, modes=modes)
        wait_all()
        return 0

    if args.cmd == "serve":
        from evaluation.dashboard import serve
        serve(port=args.port, host=args.host)
        return 0

    if args.cmd == "seed":
        from evaluation.seed_config import seed
        path = seed(overwrite=args.overwrite, swarm_config=args.swarm_config)
        print(f"wrote {path}")
        return 0

    if args.cmd == "warm":
        from evaluation.runner import warm
        warm(args.paper, force=args.force)
        return 0

    if args.cmd == "renormalize":
        from evaluation import normalization
        if args.paper:
            print(normalization.recompute_paper(args.paper))
            print(normalization.recompute_global_sd())
        else:
            r = normalization.recompute_all()
            print(r)
        return 0

    if args.cmd == "publish":
        from evaluation import config as _cfg
        from evaluation import git_publish
        if not git_publish.publish_enabled():
            parser.error(
                "publishing is disabled: set publish.enabled and "
                "publish.github_org in the config"
            )
        data_dir = _cfg.data_dir()
        git_publish._github_org()  # fail once, not per run, when unset

        ok = 0
        skipped = 0
        failed = 0
        for env_dir, meta in git_publish.iter_completed_runs(data_dir):
            paper_id = meta.get("paper_id")
            profile = meta.get("model")
            mode = meta.get("mode")
            if args.paper and paper_id != args.paper:
                continue
            if args.profile and profile != args.profile:
                continue
            if args.mode and mode != args.mode:
                continue
            try:
                res = git_publish.publish(
                    env_dir, paper_id, profile, mode, dry_run=args.dry_run,
                )
                if res.pushed:
                    print(f"pushed   {paper_id:14s}  {profile:22s}  {mode:8s}  → {res.repo_full_name}@{res.branch}  ({res.files_committed} files, {res.bytes_committed//1024} KB)")
                    ok += 1
                else:
                    print(f"skipped  {paper_id:14s}  {profile:22s}  {mode:8s}  → {res.skipped_reason}")
                    skipped += 1
            except Exception as e:
                print(f"FAILED   {paper_id}  {profile}  {mode}: {e}")
                failed += 1
            if args.limit and (ok + skipped + failed) >= args.limit:
                print(f"(stopped at --limit={args.limit})")
                break
        print(f"\n=== publish summary: {ok} pushed, {skipped} skipped, {failed} failed ===")
        return 0 if failed == 0 else 2

    parser.error(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
