"""Documentation-only QA; standard library, no model import or forward."""
import ast
import hashlib
import json
import pathlib
import re
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
result = json.loads((HERE / "RESULTS.json").read_text())
static = json.loads((HERE / "STATIC_CHECKS.json").read_text())
checks = []
for file in HERE.glob("*.json"):
    json.loads(file.read_text())
    checks.append("json:" + file.name)
for file in HERE.glob("*.py"):
    ast.parse(file.read_text())
    checks.append("python-syntax:" + file.name)
for file in HERE.glob("*.md"):
    text = file.read_text()
    assert text.count("~~~") % 2 == 0, file.name
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if "://" not in target and not target.startswith("#"):
            assert (file.parent / target.split("#")[0]).exists(), (file.name, target)
    checks.append("markdown-links-and-fences:" + file.name)
totals = {}
for arch in result["current_architectures"]:
    ids = [layer["id"] for layer in arch["layers"]]
    assert len(ids) == len(set(ids)), arch["id"]
    totals[arch["id"]] = sum(layer["parameters"] for layer in arch["layers"])
assert totals == {"GDTS": 7826588, "canonical_Stage_A": 331845,
                  "Stage_B_V2A": 30851, "V3_V4": 390579}, totals
for arch in result["proposed_architectures"]:
    total = sum(layer["parameters"] for layer in arch["layers"])
    assert total == arch["trainable_parameters"], (arch["id"], total)
    if arch["id"] in ("A", "B"):
        active = sum(layer["parameters"] for layer in arch["layers"]
                     if not layer.get("dormant", False))
        assert active == arch["active_parameters"], (arch["id"], active)
assert static["base_registered_parameters"] == totals["GDTS"]
assert static["stage_a_trainable"] == totals["canonical_Stage_A"]
assert result["actual_execution"]["gpu_calls"] == 0
assert result["actual_execution"]["exact_payload_margin_audit_executed"] is False
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
tree = subprocess.check_output(["git", "rev-parse", "HEAD:src"], cwd=ROOT, text=True).strip()
assert tree == result["source_tree"], tree
source_diff = subprocess.check_output(["git", "diff", "--", "src", "configs"], cwd=ROOT)
assert not source_diff, "production/config diff"
checks.extend(["architecture-parameter-sums", "proposal-active-parameter-sums",
               "source-tree-unchanged", "no-production-config-diff"])
files = sorted(HERE.iterdir())
fingerprints = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in files if p.is_file() and p.name not in
                {"DOCUMENT_QA.json", "ARTIFACT_HASHES.json"}]
print(json.dumps({"status": "PASS", "head_at_check": head, "source_tree": tree,
                  "checks": checks, "layer_parameter_sums": totals,
                  "files": fingerprints, "model_calls": 0}, indent=2))
