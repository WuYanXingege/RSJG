"""Read-only AST and byte checks against the input commit; no torch import."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[4]
BASE='c9dbfecf64eb95504664c2901d3c2bb26f41aea5'
paths=['src/jdv2_objective_state.py','src/joint_goal_loss.py','src/models/model.py','src/parser.py','src/trainer.py']
def definitions(source):
    answer={}
    def walk(nodes,prefix=''):
        for n in nodes:
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
                answer[prefix+n.name]=ast.dump(n,include_attributes=False)
            elif isinstance(n,ast.ClassDef):walk(n.body,prefix+n.name+'.')
    walk(ast.parse(source).body)
    return answer
result={'input_commit':BASE,'files':{}}
for path in paths:
    old=subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)
    new=(ROOT/path).read_bytes()
    a,b=definitions(old),definitions(new)
    result['files'][path]={'old_sha256':hashlib.sha256(old).hexdigest(),'new_sha256':hashlib.sha256(new).hexdigest(),
        'unchanged_definitions':sorted(k for k in a if k in b and a[k]==b[k]),
        'changed_definitions':sorted(k for k in a if k in b and a[k]!=b[k]),
        'added_definitions':sorted(set(b)-set(a)),'removed_definitions':sorted(set(a)-set(b))}
assert result['files']['src/models/model.py']['changed_definitions']==['GDTS._jdv2_mc_goal_losses']
assert result['files']['src/joint_goal_loss.py']['changed_definitions']==['expected_conditional_composite']
print(json.dumps(result,indent=2))
