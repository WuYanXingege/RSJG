# Local assets

`assets/data`, `assets/checkpoints`, and `assets/references` are ignored local
links into the persistent sibling store `RSJG_EBJD_assets`.  They are never
committed.  Run `scripts/link_assets.sh` to create or verify the links.

The expected hashes and readiness status are recorded in
`results/ASSET_MANIFEST.json` and `docs/EXPERIMENT_REGISTRY.md`.
