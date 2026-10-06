#!/usr/bin/env python3
"""Launch the fixed upstream GDTS HOTEL reproduction after bound checks."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


OFFICIAL_COMMIT = "297d508558c10831983ea4b19c2b3e657459a449"
EXPECTED_CODE = {
    "main.py": "a82919313036d0a0caa55b005f03ec3def820a5c895072055047ab39724e0aef",
    "train.sh": "24d34631ec8152ba1a9ab34800ae02341cd66c46d51af4fe06aa931aa2672b5a",
    "src/parser.py": "41e426ad1e8f2e49d9225f7ecbfd83d0357225749612216456a56c7e68b9ac7a",
    "src/trainer.py": "b8643fc91f11ec39499ca58b6ed53d83489f40e191c87c03376ed2c59ac0bbad",
    "src/data_pre_process.py": "8f27ced2aeff3ff3475320eb54df4489a582d468b29829da96188c3aa52b200d",
    "src/data_loader.py": "c6c66e761455c126dda173d107033dd30e60d4777acafeb285bd671632376952",
    "src/models/model.py": "6f671a4b38db5c9e92dcf55425dbf89b6ebe0f517a158645234d45a317f8af48",
}
EXPECTED_PATCH = "b8a09ae42f0c1ce3136a4c0a6e40d33f437defff3b87e014fa1c2a8c6b51f5e1"
EXPECTED_DATA = {
    "data/eth5/hotel/train/biwi_eth.txt": "cd75b1008b82b7f442b2e03967b0f4aac36da2e73d440df1f605bb197d23fb33",
    "data/eth5/hotel/train/crowds_zara01.txt": "1147a1962a09abfb86f28c6cddcac862e095a0cf129b3016385b69eacdd09d85",
    "data/eth5/hotel/train/crowds_zara02.txt": "8a649d0f8c9ae75c87c4d23a85f892786b0aa30266e996c7be03e69dafff22ff",
    "data/eth5/hotel/train/crowds_zara03.txt": "16b3e899932c4baacd07f45013d5b921f90bc5a29eb2b0fe42f4d7c904ac3108",
    "data/eth5/hotel/train/students001.txt": "a6d87f278d94136fe39b8be91555487a29ac77259ae403b9dba2d5c18caf7b5b",
    "data/eth5/hotel/train/students003.txt": "e25798b660634330aa89f8bb259425de720e84d0873902726c1d1f4ccff21d6c",
    "data/eth5/hotel/val/biwi_hotel.txt": "9caa771bb9153d6b809dd0916b6f86761b641e6bbb15e766c1de3133fbbb7fcf",
    "data/eth5/hotel/test/biwi_hotel.txt": "9caa771bb9153d6b809dd0916b6f86761b641e6bbb15e766c1de3133fbbb7fcf",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, destination)


def verify(root, smoke):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != OFFICIAL_COMMIT:
        raise RuntimeError("official commit mismatch")
    actual_code = {
        name: sha256(root / name) for name in EXPECTED_CODE
    }
    if actual_code != EXPECTED_CODE:
        raise RuntimeError("official/compatibility source mismatch")
    patch = subprocess.check_output(
        ["git", "diff", "--", "src/data_loader.py"], cwd=root)
    if hashlib.sha256(patch).hexdigest() != EXPECTED_PATCH:
        raise RuntimeError("compatibility patch mismatch")
    data_root = root.parent / "data"
    actual_data = {
        name: sha256(data_root / Path(name).relative_to("data"))
        for name in EXPECTED_DATA
    }
    if actual_data != EXPECTED_DATA:
        raise RuntimeError("official HOTEL data mismatch")
    if len(list((root / "output/hotel/data_batches/train_batches").glob(
            "*.pkl"))) != 591:
        raise RuntimeError("train cache incomplete")
    if len(list((root / "output/hotel/data_batches/valid_batches").glob(
            "*.pkl"))) != 19:
        raise RuntimeError("valid cache incomplete")
    receipt = json.loads(smoke.read_text())
    if (receipt.get("status") != "PASS" or
            receipt.get("official_commit") != head or
            receipt.get("code_sha256") != actual_code or
            receipt.get("cache") != {
                "train_batches": 591, "valid_batches": 19}):
        raise RuntimeError("stale CUDA smoke receipt")
    for relative in (
            "output/hotel/config_train_test.yaml",
            "output/hotel/log_curve.txt",
            "output/hotel/saved_models"):
        if (root / relative).exists():
            raise RuntimeError("formal HOTEL training artifact already exists")
    return head, actual_code, actual_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.official_root.resolve()
    smoke = args.smoke.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    request = output / "LAUNCH_REQUEST.json"
    if request.exists():
        raise RuntimeError("duplicate formal HOTEL launch")
    head, code, data = verify(root, smoke)
    if args.check_only:
        print(json.dumps({
            "status": "OFFICIAL_GDTS_HOTEL_READY",
            "official_commit": head,
            "compatibility_patch_sha256": EXPECTED_PATCH,
            "code_sha256": code,
            "data_sha256": data,
        }))
        return
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True, capture_output=True)
    if query.returncode or query.stdout.strip():
        raise RuntimeError("GPU busy or query failed")
    python = (root / "../.conda/rsjg/bin/python").resolve()
    command = [
        "setsid", "nohup", str(python), "-u", "-B", "main.py",
        "--dataset", "eth5", "--test_set", "hotel",
        "--reproducibility", "True", "--phase", "train_test",
        "--num_epochs", "250", "--batch_size", "64",
        "--start_validation", "1", "--validate_every", "10",
        "--learning_rate", "0.001", "--skip_ts_window", "1",
        "--down_factor", "8", "--num_workers", "2",
        "--use_wandb", "False", "--seed", "2025",
    ]
    log = output / "official_gdts_hotel.log"
    environment = dict(
        os.environ,
        MPLCONFIGDIR="/tmp/rsjg_mpl_official",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        NUMEXPR_NUM_THREADS="4",
    )
    with log.open("x") as handle:
        child = subprocess.Popen(
            command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, close_fds=True)
    payload = {
        "status": "LAUNCH_REQUESTED",
        "pid": child.pid,
        "official_commit": head,
        "compatibility_patch_sha256": EXPECTED_PATCH,
        "code_sha256": code,
        "data_sha256": data,
        "smoke_sha256": sha256(smoke),
        "command": command,
        "cwd": str(root),
        "log": str(log),
        "boundary": {
            "literal_official_code_reproduction": True,
            "logging_disabled_only": True,
            "goal_pretrain_loaded": False,
            "validation_and_test_source_bytes_identical": True,
        },
    }
    atomic_json(payload, request)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
