# Clean Workspace Setup

Target branch:

```text
research/joint-dependency-v2-clean
```

Base commit:

```text
f9d4fefce5738f4e85942ce4fe8d9dc754d1f637
```

## Preferred: fresh clone

```bash
git clone https://github.com/WuYanXingege/RSJG.git RSJG_joint_dependency_v2
cd RSJG_joint_dependency_v2
git fetch origin
git checkout -b research/joint-dependency-v2-clean \
  f9d4fefce5738f4e85942ce4fe8d9dc754d1f637

git status --short
git rev-parse HEAD
```

Expected `git status --short`: no output.

## Alternative: worktree

```bash
cd /path/to/existing/RSJG
git fetch origin
git worktree add ../RSJG_joint_dependency_v2 \
  -b research/joint-dependency-v2-clean \
  f9d4fefce5738f4e85942ce4fe8d9dc754d1f637

cd ../RSJG_joint_dependency_v2
git status --short
```

Never clean/reset the old research workspace just to make the new one clean.

Use separate runtime roots:

```bash
export RSJG_CACHE_ROOT=/path/to/cache_joint_dependency_v2
export RSJG_OUTPUT_ROOT=/path/to/output_joint_dependency_v2
```

Reuse only immutable assets with matching manifests, such as raw datasets and frozen baseline checkpoints.
