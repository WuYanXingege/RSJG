#!/bin/bash
set -euo pipefail

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v unzip >/dev/null || { echo "unzip is required" >&2; exit 1; }

if [[ -e data || -e ../data ]]; then
  echo "Refusing to download: ./data or ../data already exists." >&2
  exit 1
fi

archive_path=$(mktemp "${TMPDIR:-/tmp}/gdts-data.XXXXXX.zip")
trap 'rm -f "$archive_path"' EXIT

curl --fail --location --output "$archive_path" \
  'https://www.dropbox.com/s/luu2t6c6d24xvrb/dataset.zip?dl=0'
unzip "$archive_path"

mv data ..
