"""Per-paper cache: pre-downloaded data + pre-built container.

Populated once by the ``warm`` command and reused across runs.

Layout:

    <data_dir>/cache/<paper_id>/
    ├── container.sif               ← copied into each run's workspace
    ├── overlay.img                 ← copied into each run's workspace
    ├── data/downloads/<paths>      ← RO bind-mounted into each run
    └── manifest.json               ← what's here, when, for validation

Per-run consumption (see :func:`apply_to_workspace`):

- ``container.sif`` and ``overlay.img`` are **copied** into
  ``<env_dir>/environment/``: each run owns its own files and can
  rebuild or delete them freely.
- Each cached download is **bind-mounted read-only at the file level**
  into ``<env_dir>/data/downloads/<rel>``. The containing directory
  stays writable, so the agent can add other files; only the specific
  cached file is immutable (kernel-enforced EROFS on write).
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from evaluation import config as _config
from evaluation.container import _copy_artifact

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
CONTAINER_SIF_NAME = "container.sif"
OVERLAY_IMG_NAME = "overlay.img"


def cache_root() -> Path:
    return _config.data_dir() / "cache"


def cache_dir(paper_id: str) -> Path:
    return cache_root() / paper_id


def manifest_path(paper_id: str) -> Path:
    return cache_dir(paper_id) / MANIFEST_NAME


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Lookup ────────────────────────────────────────────────────────────────


def load_manifest(paper_id: str) -> dict | None:
    mp = manifest_path(paper_id)
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def is_warm(paper_id: str) -> bool:
    """True if a complete cache exists for this paper."""
    return load_manifest(paper_id) is not None


# ── Population (called by ``runner.warm``) ────────────────────────────────


def populate_from_workspace(paper_id: str, env_dir: Path) -> dict:
    """Copy artifacts from a warmed workspace into the cache, write manifest.

    Assumes the workspace has been fully prepared (download.sh ran,
    container.sif built). Collects:
    - ``environment/container.sif``
    - ``environment/overlay.img``
    - every regular file under ``data/downloads/``

    Writes the manifest last, atomically, so a half-populated cache is
    never considered warm.
    """
    cache_path = cache_dir(paper_id)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Container images and overlays are copied with _copy_artifact, which
    # keeps the overlay sparse instead of writing out its full size.
    has_container = False
    src_sif = env_dir / "environment" / "container.sif"
    if src_sif.exists():
        _copy_artifact(src_sif, cache_path / CONTAINER_SIF_NAME)
        has_container = True
        log.info("Cached container.sif (%d bytes)", src_sif.stat().st_size)

    has_overlay = False
    src_overlay = env_dir / "environment" / "overlay.img"
    if src_overlay.exists():
        _copy_artifact(src_overlay, cache_path / OVERLAY_IMG_NAME)
        has_overlay = True
        log.info("Cached overlay.img (%d bytes)", src_overlay.stat().st_size)

    files: list[dict] = []
    downloads_src = env_dir / "data" / "downloads"
    downloads_dst = cache_path / "data" / "downloads"
    if downloads_src.exists():
        for src in sorted(downloads_src.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(downloads_src)
            dst = downloads_dst / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            files.append({
                "rel_path": f"data/downloads/{rel.as_posix()}",
                "size": src.stat().st_size,
            })
        log.info("Cached %d download artefacts", len(files))

    manifest = {
        "paper_id": paper_id,
        "warmed_at": _utcnow(),
        "has_container": has_container,
        "has_overlay": has_overlay,
        "files": files,
    }

    # Atomic write: tmp + rename. Any crash mid-populate leaves no
    # manifest, so ``is_warm`` correctly returns False.
    tmp = cache_path / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(manifest_path(paper_id))
    log.info("Wrote manifest %s", manifest_path(paper_id))

    return manifest


# ── Consumption (called by runner._scaffold) ──────────────────────────────


def apply_to_workspace(
    paper_id: str, env_dir: Path,
) -> list[tuple[Path, str]]:
    """Seed a freshly scaffolded workspace from the cache.

    Returns a list of ``(host_path, container_path)`` bind pairs that
    the caller must pass into the container command as RO file-level
    binds. The container is **copied** (not bound) and does not appear
    in the returned list.

    No-op if no cache for this paper.
    """
    manifest = load_manifest(paper_id)
    if manifest is None:
        return []

    cache_path = cache_dir(paper_id)

    # Container — real copy; agent owns it.
    if manifest.get("has_container"):
        src_sif = cache_path / CONTAINER_SIF_NAME
        dst_sif = env_dir / "environment" / "container.sif"
        if src_sif.exists() and not dst_sif.exists():
            _copy_artifact(src_sif, dst_sif)
            log.info("Copied cached container.sif → %s", dst_sif)

    # Overlay — real copy; each run owns its own (same reason as SIF).
    # Nexus mounts it read-only for GPU jobs, so per-run isolation is
    # enforced downstream too.
    if manifest.get("has_overlay"):
        src_ov = cache_path / OVERLAY_IMG_NAME
        dst_ov = env_dir / "environment" / "overlay.img"
        if src_ov.exists() and not dst_ov.exists():
            _copy_artifact(src_ov, dst_ov)
            log.info("Copied cached overlay.img → %s", dst_ov)

    # Data files — RO file-level binds. Pre-create empty placeholders
    # at the target paths so apptainer has something to mount over;
    # the bind then shadows the placeholder read-only.
    binds: list[tuple[Path, str]] = []
    for f in manifest.get("files", []):
        rel = f["rel_path"]
        src = cache_path / rel
        if not src.exists():
            log.warning("Manifest lists missing cache file: %s", src)
            continue
        target_host = env_dir / rel
        target_host.parent.mkdir(parents=True, exist_ok=True)
        if not target_host.exists():
            target_host.touch()
        container_path = f"/home/user/{rel}"
        binds.append((src, container_path))

    if binds:
        log.info("Applied cache for %s: %d RO file-binds", paper_id, len(binds))
    return binds
