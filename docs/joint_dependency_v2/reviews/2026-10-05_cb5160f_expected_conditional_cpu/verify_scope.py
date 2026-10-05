"""Read-only AST/source comparison against the authorized design commit."""
import ast
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[4]
SOURCE = "cb5160fa8783cd7c90f0f0453809bf3e84665ed6"
FILE = "src/joint_goal_loss.py"
before = subprocess.check_output(["git","show",SOURCE+":"+FILE],cwd=ROOT,text=True)
after = (ROOT/FILE).read_text()
old, new = ast.parse(before), ast.parse(after)
old_defs = {n.name:n for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
new_defs = {n.name:n for n in new.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
unchanged = []
for name,node in old_defs.items():
    assert ast.get_source_segment(before,node)==ast.get_source_segment(after,new_defs[name]),name
    unchanged.append(name)
assert set(new_defs)-set(old_defs)=={"expected_conditional_composite"}
callers = []
for path in (ROOT/"src").rglob("*.py"):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and (
            isinstance(node.func,ast.Name) and node.func.id=="expected_conditional_composite" or
            isinstance(node.func,ast.Attribute) and node.func.attr=="expected_conditional_composite"):
            callers.append(str(path.relative_to(ROOT)))
assert not callers,callers
function=new_defs["expected_conditional_composite"]
for node in ast.walk(function):
    if isinstance(node,ast.Call):
        name=node.func.id if isinstance(node.func,ast.Name) else getattr(node.func,"attr","")
        assert name not in {"Parameter","register_buffer","rand","randn","randint",
                            "multinomial","Generator","manual_seed","load","save"},name
tracked=subprocess.check_output(["git","diff",SOURCE,"--name-only"],cwd=ROOT,text=True).splitlines()
allowed={"src/joint_goal_loss.py","docs/joint_dependency_v2/reviews/README.md",
         "tests/test_jdv2_expected_conditional.py"}
prefix="docs/joint_dependency_v2/reviews/2026-10-05_cb5160f_expected_conditional_cpu/"
assert all(p in allowed or p.startswith(prefix) for p in tracked),tracked
protected=["src/models/model.py","src/parser.py","src/trainer.py",
           "src/models","configs",
           "docs/joint_dependency_v2/reviews/2026-10-04_79ea3fa_innovation_theory_architecture"]
assert not subprocess.check_output(["git","diff",SOURCE,"--",*protected],cwd=ROOT)
print(json.dumps({"status":"PASS","source_commit":SOURCE,
    "old_definitions_byte_identical":unchanged,"new_definitions":["expected_conditional_composite"],
    "production_callers":callers,"protected_tracked_paths_unchanged":protected,
    "tracked_diff_at_check":tracked,
    "added_parameters":0,"added_buffers":0,"model_strict_load_executed":False,
    "semantics_note":"new standalone library function added; existing definitions unchanged",
    "library_sha256":hashlib.sha256(after.encode()).hexdigest()},indent=2))
