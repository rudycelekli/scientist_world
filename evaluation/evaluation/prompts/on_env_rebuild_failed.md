## Environment rebuild failed

Your edit to `environment/container.def` or `environment/setup.sh`
could not be built. The job was **not** submitted.

Error:

```
{error}
```

Logs:

- SIF builds → `logs/container_build.log`
- Overlay / setup.sh runs → `logs/setup.log`

Open the relevant log, find the first error, and fix the root cause
in `container.def` or `setup.sh`. Then write `action: submit` again —
the pipeline will try to rebuild. Do not mark `done` because of a
build failure; environment errors are fixable.
