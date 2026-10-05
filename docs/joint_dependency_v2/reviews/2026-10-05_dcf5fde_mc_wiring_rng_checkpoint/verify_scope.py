"""Static source comparison, no tensors/models/checkpoints."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
BASE = 'dcf5fde1aa77f62a25a9209fd91e4f854a21510a'


def definitions(source, name):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)
    return {n.name: '\n'.join(source.splitlines()[n.lineno-1:n.end_lineno])
            for n in cls.body if isinstance(n, ast.FunctionDef)}


old = subprocess.check_output(['git', 'show', BASE+':src/models/model.py'], cwd=ROOT, text=True)
new = (ROOT/'src/models/model.py').read_text()
a, b = definitions(old, 'GDTS'), definitions(new, 'GDTS')
changed = [name for name in a if a[name] != b[name]]
assert changed == ['__init__', '_jdv2_no_z_goal_losses']
dispatch = "        if objective_mode(self.args) == MC:\n            return self._jdv2_mc_goal_losses(inputs)\n"
assert b['_jdv2_no_z_goal_losses'].replace(dispatch, '') == a['_jdv2_no_z_goal_losses']
assert b['__init__'].replace('        validate_mode(args, device)\n', '') == a['__init__']
old_trainer = subprocess.check_output(['git', 'show', BASE+':src/trainer.py'], cwd=ROOT, text=True)
new_trainer = (ROOT/'src/trainer.py').read_text()
c, d = definitions(old_trainer, 'trainer'), definitions(new_trainer, 'trainer')
for name in ('_jdv2_architecture_config', '_jdv2_ablation_config', '_jdv2_inference_sampler_config', '_set_scheduler'):
    assert c[name] == d[name]
loop = d['_train_loop']
assert loop.index('self._save_checkpoint(epoch)') < loop.index('self.scheduler.step(')
assert loop.index('self._save_checkpoint(epoch, best_epoch=True)') < loop.index('self.scheduler.step(')
assert loop.index('self._save_checkpoint(epoch, last_epoch=True)') > loop.rindex('self.scheduler.step(')
assert "if self.args.training_stage == 'multiway_coupling' else 1" in new_trainer
assert not subprocess.check_output(['git', 'diff', BASE, '--', 'configs', 'src/joint_goal_loss.py'], cwd=ROOT)
result = dict(status='PASS', source_commit=BASE, changed_existing_GDTS_methods=changed,
              old_no_z_body_exact_after_dispatch_removal=True,
              unchanged_GDTS_methods=[name for name in a if name not in changed],
              objective_not_in_architecture=True, scheduler_definition_unchanged=True,
              best_numbered_before_scheduler_last_after=True, stage_a_accumulation=1,
              pure_loss_and_configs_unchanged=True,
              scope='static source identity; not real-model/GPU output parity')
result['files'] = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in (
    'src/models/model.py', 'src/parser.py', 'src/trainer.py', 'src/jdv2_objective_state.py',
    'tests/test_jdv2_mc_wiring.py', 'tests/mc_wiring_fixture.py')}
print(json.dumps(result, indent=2))
