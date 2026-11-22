"""CPU-only adapter and synthetic complete-state writer checks; no research fit."""
from pathlib import Path
import math

import torch
import yaml

from bcep.contracts import canonical_hash, digest
from bcep_joint.common import read
from bcep_v2.provider import GEOMETRY
from bcep_v2.state import run_stage
from joint_flow_fourfold.inputs import Inputs, SOURCE
from joint_flow_fourfold.train import RECIPE, _retention_check


def test_existing_zara2_assets_strict_load_on_cpu(tmp_path):
    formal = SOURCE.parent/'RSJG_EBJD_assets/experiments/bcep_ebjd_v2_integrated'
    run = read(formal/'FORMAL_RUN.json')
    plan = read(run['input_plan'])
    split_path = Path(plan['folds']['zara2']['split']['path'])
    split = read(split_path)
    config_path = formal/'AUTHORIZED.yaml'
    config = yaml.safe_load(config_path.read_text())
    geometry = dict(layout=GEOMETRY, split=config['folds']['zara2']['split_sha256'],
        crop_sources=config['geometry_source'], records=canonical_hash([row['sha256'] for row in split['records']]))
    manifest = dict(fold='zara2', split_path=str(split_path), split_sha256=digest(split_path),
        config_path=str(config_path), config_sha256=digest(config_path),
        identity=run['folds']['zara2']['identity'], geometry=geometry, banks={})
    for stage in ('density', 'head', 'joint'):
        folder = formal/'zara2'/stage
        frozen = read(folder/'FROZEN.json')
        checkpoint = folder/f"epoch_{frozen['epoch']:03d}.pt"
        manifest[stage] = dict(checkpoint=str(checkpoint), sha256=digest(checkpoint), frozen=frozen,
                               stage_directory=str(folder))
    for purpose in ('train', 'selection_dev'):
        folder = formal/'zara2/refined_banks'/f'{purpose}_2035'
        manifest['banks'][purpose] = dict(path=str(folder), identity=read(folder/'IDENTITY.json'),
                                         sha256=canonical_hash(read(folder/'COMPLETE.json')))
    torch.set_num_threads(1)
    inputs = Inputs(manifest, tmp_path, 'cpu')
    model, provider = inputs.load_frozen('cpu')
    assert inputs.config['scales'] == split['bridge_scales']
    assert len(model.bridge.denoiser.blocks) == 6
    assert math.isclose(float(model.bridge.representation.goal_scale), split['bridge_scales']['goal'], rel_tol=1e-6)
    assert math.isclose(float(model.bridge.representation.bridge_scale), split['bridge_scales']['bridge'], rel_tol=1e-6)
    assert not any(p.requires_grad for net in (model, provider) for p in net.parameters())
    assert not any(m.training for net in (model, provider) for m in net.modules())
    assert all(p.device.type == 'cpu' for net in (model, provider) for p in net.parameters())
    # Verify the barrier before any heldout array is requested.
    try:
        inputs.data.item(split['heldout_confirmation_indices'][0], 'heldout_confirmation', truth=True)
    except ValueError:
        pass
    else:
        raise AssertionError('heldout access unexpectedly allowed')
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def _toy_run(folder, until, resume):
    torch.manual_seed(403)
    model = torch.nn.Linear(2, 1)
    recipe = dict(RECIPE, max_epochs=6, minimum=6, patience=10, logical_scenes=2)
    completed = [torch.load(folder/'last.pt', map_location='cpu', weights_only=False)['completed_epoch'] if resume else 0]
    scores = [6., 3., 3., 2., 7., 1.]
    def backward(group, order_rng, view_rng):
        del order_rng
        for index in group:
            x = torch.tensor([float(index), 1.])+torch.rand(2, generator=view_rng)*.01+torch.randn(2)*.001
            (model(x).square().mean()/len(group)).backward()
        return {'toy_loss_window_count':len(group)}
    def validate():
        value = scores[completed[0]]
        completed[0] += 1
        validate.metrics = {'all':{'JADE':value}}
        return value
    return run_stage(folder, {'stage':'joint','fold':'synthetic-only'}, recipe, model,
                     list(range(6)), backward, validate, until, resume=resume)


def test_native_writer_six_epochs_partial_resume_and_top5(tmp_path):
    torch.set_num_threads(1)
    direct = _toy_run(tmp_path/'direct', 18, False)
    partial = _toy_run(tmp_path/'resumed', 4, False)
    assert partial['completed_epoch'] == 1 and partial['cursor'] == 2
    assert partial['status'] == 'SESSION_BUDGET_PAUSED'
    resumed = _toy_run(tmp_path/'resumed', 18, True)
    assert resumed['status'] == direct['status'] == 'COMPLETE'
    assert resumed['completed_epoch'] == 6 and resumed['update'] == 18
    for name, weight in direct['model'].items():
        assert torch.equal(weight, resumed['model'][name])
    for name in ('cpu','order','view'):
        assert torch.equal(direct['rng'][name], resumed['rng'][name])
    assert resumed['history'] == direct['history']
    receipt = _retention_check(tmp_path/'resumed', resumed)
    assert receipt['all_completed_epoch_snapshots'] == 6
    assert receipt['top5_distinct_epochs'] == [6, 4, 2, 3, 1]
    assert RECIPE['max_epochs'] == RECIPE['scheduler_horizon'] == 100
    assert RECIPE['minimum'] == 30 and RECIPE['patience'] == 15 and RECIPE['logical_scenes'] == 16
