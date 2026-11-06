# Environment {phase_cap} Failed

Your `container.def` or `setup.sh` failed to build. Your job was not submitted.

## Scheduler hints

- Failure class: `{failure_class}`
- Failure signature: `{failure_signature}`
- Hint: {failure_hint}
- Env comparison: {env_change_note}
- Repeat signal: {retry_note}

These hints are advisory, not authoritative. The scheduler cannot perfectly
diagnose every failure. If you believe the issue is transient, you may still
submit again without changing the environment files. If you believe the issue
is real and local to the workspace, edit the environment files first.

## Recommended base image

If your current base image is failing repeatedly (Docker Hub rate limits, missing CUDA, broken pip under fakeroot), the proven-working pattern is:

```
Bootstrap: docker
From: ubuntu:22.04

%post
    apt-get update
    apt-get install -y wget gnupg
    # Install CUDA from NVIDIA's apt repo (avoids Docker Hub rate limits)
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update
    apt-get install -y cuda-toolkit-12-4 python3 python3-pip python3-dev
```

## Do NOT give up after repeated build failures

Container build errors are nearly always **transient or fixable**. Common transients: Docker Hub rate limits, fakeroot/overlay timing, network blips. Common fixables: missing apt packages, wrong base image. Iterate on the environment, switch base images, retry — but do not write `action: done` because of build failures alone. Mark `done` only when the verifier signs off on completed reproduction work, not because the environment refuses to build.

## Common issues

- **`pip install` or `uv pip install` in `container.def` `%post`** — this is the #1 cause of build failures. Python packages belong in `setup.sh`, not `container.def`. Only `apt-get` packages go in `%post`.
- **`uv: command not found` or `unrecognized subcommand 'uv'`** — `uv` is not available during container build. If you need pip in `%post` for some reason, use `python3 -m pip`, but strongly prefer moving it to `setup.sh`.
- **`pip3: Too many levels of symbolic links`** — pip3 is broken under fakeroot (used during container builds). Use `python3 -m pip` instead, or better yet, move pip installs to `setup.sh`.
- **CUDA base image missing Python** — if you use `nvidia/cuda:*` as your base, you must `apt-get install python3 python3-pip python3-dev` in `%post`. The CUDA images don't include Python.
- **SSL certificate errors in `setup.sh`** — ignore the warnings about `SSL_CERT_FILE` / `SSL_CERT_DIR`; they are harmless. If downloads actually fail, check that the package name and version exist.
- **Overlay locked by another process** — a previous job may still be running. Wait and retry.

## Build log (last 40 lines)

```
{log_tail}
```

Full log: `{build_log_path}`

Fix the issue and submit again.
