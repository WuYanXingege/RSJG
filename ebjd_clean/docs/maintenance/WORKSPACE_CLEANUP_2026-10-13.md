---
date: 2026-10-13
system: RSJG/EBJD
type: workspace_cleanup
status: COMPLETED_SAFE_SCOPE
---

# Workspace cleanup receipt

The cleanup was performed only after confirming that no RSJG, EBJD, training,
evaluation, or supervisor process was running.

## Removed regenerable state

- Workspace-local pip/download caches (`.pip-cache`, `.cache`).
- 73 Python/test cache directories (`__pycache__`, `.pytest_cache`, and related
  cache names), totalling approximately 13.3 MiB.
- 23 stale PID files whose recorded processes no longer existed.
- Six clean, completed duplicate Git worktrees. Their branches and commits were
  retained and can be materialized again with `git worktree add`.

The removed worktrees were:

- `RSJG_EBJD_reports`
- `RSJG_EBJD_speed_worktree`
- `RSJG_EBJD_current_speedup_worktree`
- `RSJG_EBJD_epoch45_diagnostic_worktree`
- `RSJG_EBJD_goalweight_pilot_worktree`
- `RSJG_EBJD_rollout005_pilot_worktree`

## Reversible quarantine

Seven failed, empty, or superseded sub-megabyte EBJD attempt directories were
moved out of the production output tree to:

`/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.cleanup_quarantine/20261013/ebjd_outputs`

The quarantine occupies 472 KiB and was not permanently deleted.

## Preserved evidence

- Version-controlled tests and production scripts were retained.
- Canonical training logs, manifests, reports, evaluation banks, and
  checkpoints were retained.
- Large generated batch caches were retained because deleting them would force
  expensive regeneration; they are not classified as meaningless logs.
- Historical Git branches were retained at their original commit IDs.

Critical artifact verification after cleanup:

| Artifact | SHA256 |
|---|---|
| HOTEL EBJD epoch 45 | `425988097793b639eb868fcc9ff28e41a3022c08598d13f396383da89268e1b1` |
| rollout-weight 0.05 epoch 50 | `26e7551e1afe868ca0e6faf203cae3a0d1a440e51d0c4bb87d5e1bebb0663497` |
| rollout-weight 0.05 final JSON | `1e13f20092ee657ef17cf4969f4a8c7f4e4c2f79778ae24ffec0ee6c0aa15aad` |

The safe cleanup reclaimed approximately 2.1 GiB. No checkpoint, formal result,
raw dataset, or active experiment was deleted.
