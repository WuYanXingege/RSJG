#!/usr/bin/env bash
set -euo pipefail

paper_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_root="${EBJD_ASSET_ROOT:-$(cd "$paper_root/.." && pwd)/RSJG_EBJD_assets}"

for name in data checkpoints references; do
  test -d "$asset_root/$name" || {
    printf 'missing asset directory: %s\n' "$asset_root/$name" >&2
    exit 2
  }
  if test -L "$paper_root/assets/$name"; then
    test "$(readlink -f "$paper_root/assets/$name")" = "$(readlink -f "$asset_root/$name")" || {
      printf 'existing link targets another asset store: %s\n' "$paper_root/assets/$name" >&2
      exit 3
    }
  elif test -e "$paper_root/assets/$name"; then
    printf 'refusing to replace existing asset path: %s\n' "$paper_root/assets/$name" >&2
    exit 4
  else
    ln -s "$asset_root/$name" "$paper_root/assets/$name"
  fi
done

printf 'asset store: %s\n' "$asset_root"
