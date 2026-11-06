"""Canonical model-id resolution for dataset rows.

A run records only its *profile* name in ``.evaluation.json["model"]``.
The model behind a profile is resolved from the ``profiles`` block of
the evaluation config: ``actual_model`` when the profile sets it, else
``model``. Set ``actual_model`` on profiles whose ``model`` value does
not identify the model (a gateway alias, or one value shared by several
models); otherwise their rows get the same model id.

resolve_model_id(profile) -> canonical model id string.
"""
from __future__ import annotations

import functools

from evaluation import config as _config


@functools.lru_cache(maxsize=1)
def _profiles() -> dict:
    return _config.load().get("profiles") or {}


@functools.lru_cache(maxsize=256)
def resolve_model_id(profile: str) -> str:
    """Return the canonical model id for a profile name.

    Order: config ``actual_model`` -> config ``model`` -> the profile
    name itself (fallback for profiles that are no longer in the config
    but still have runs on disk).
    """
    p = _profiles().get(profile)
    if not p:
        # Profile not in the current config. Best-effort: strip the
        # trailing effort tier so e.g. 'my-model-xhigh' -> 'my-model'.
        for tier in ("-xhigh", "-high", "-medium", "-low"):
            if profile.endswith(tier):
                return profile[: -len(tier)]
        return profile
    return p.get("actual_model") or p.get("model") or profile


_EFFORT_TIERS = ("xhigh", "high", "medium", "low")


@functools.lru_cache(maxsize=256)
def combined_model_id(profile: str) -> str:
    """Canonical model id with the effort tier folded in, e.g.
    profile ``my-model-xhigh`` (model ``my-model-1``) -> ``my-model-1-xhigh``.

    The tier is the trailing ``-xhigh/-high/-medium/-low`` of the profile
    name. Profiles with no such tier return the bare canonical id.
    """
    base = resolve_model_id(profile)
    tier = next((t for t in _EFFORT_TIERS if profile.endswith("-" + t)), None)
    if tier and not base.endswith("-" + tier):
        return f"{base}-{tier}"
    return base


def scaffold_for(profile: str) -> str | None:
    p = _profiles().get(profile)
    return p.get("scaffold") if p else None
