from types import SimpleNamespace

import pytest
import torch

from src.models.goal_pretrain import (
    Goal_Pretrain,
    goal_architecture_config,
    goal_pretrainer,
)
from src.parser import get_parser
from src.trainer import trainer


class _ToyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.goal_module = torch.nn.Linear(3, 2)
        self.diffnet = torch.nn.Linear(3, 2)


def _args(tmp_path):
    return SimpleNamespace(
        pretrain_path=None,
        goal_model_type='independent',
        training_stage='baseline',
        dataset='sdd',
        test_set='sdd',
        obs_length=8,
        pred_length=12,
        goal_pretrain_checkpoint=None,
        goal_pretrain_checkpoint_sha256=None,
        goal_pretrain_checkpoint_epoch=None,
        model_dir=str(tmp_path),
    )


def _checkpoint(path, args, goal_module, **overrides):
    payload = {
        'checkpoint_type': 'gdts_goal_pretrain',
        'format_version': 1,
        'epoch': 140,
        'dataset': 'sdd',
        'test_set': 'sdd',
        'goal_architecture': goal_architecture_config(args),
        'training_epochs_planned': 150,
        'model_state_dict': {
            f'goal_module.{key}': value.detach().clone()
            for key, value in goal_module.state_dict().items()
        },
    }
    payload.update(overrides)
    torch.save(payload, path)
    return payload


def _runner(tmp_path):
    instance = object.__new__(trainer)
    instance.args = _args(tmp_path)
    instance.device = torch.device('cpu')
    instance.net = _ToyNet()
    return instance


def test_goal_pretrain_metrics_are_bce_only():
    model = object.__new__(Goal_Pretrain)
    assert model.init_test_metrics() == {'goal_BCE': []}
    assert model.init_best_metrics() == {'goal_BCE': 1e9}
    assert model.best_valid_metric() == 'goal_BCE'


def test_parser_default_does_not_enable_goal_initialization():
    args = get_parser().parse_args([])
    assert args.goal_pretrain_checkpoint is None


def test_strict_goal_checkpoint_loads_only_goal_module(tmp_path):
    instance = _runner(tmp_path)
    source = torch.nn.Linear(3, 2)
    with torch.no_grad():
        source.weight.fill_(2.5)
        source.bias.fill_(-0.75)
    checkpoint = tmp_path / 'goal.pt'
    _checkpoint(checkpoint, instance.args, source)

    diffusion_before = {
        key: value.detach().clone()
        for key, value in instance.net.diffnet.state_dict().items()
    }
    instance._load_goal_pretrain_checkpoint(str(checkpoint))

    for key, value in source.state_dict().items():
        assert torch.equal(instance.net.goal_module.state_dict()[key], value)
    for key, value in diffusion_before.items():
        assert torch.equal(instance.net.diffnet.state_dict()[key], value)
    assert instance.args.goal_pretrain_checkpoint_epoch == 140
    assert len(instance.args.goal_pretrain_checkpoint_sha256) == 64
    assert instance.goal_pretrain_initialization[
        'training_epochs_planned'] == 150


@pytest.mark.parametrize(
    'override, message',
    [
        ({'checkpoint_type': 'full_model'}, 'typed goal-pretrain'),
        ({'dataset': 'eth5'}, 'dataset mismatch'),
        ({'format_version': 2}, 'Unsupported'),
        ({'goal_architecture': {'wrong': True}}, 'architecture mismatch'),
    ],
)
def test_goal_checkpoint_rejects_incompatible_metadata(
        tmp_path, override, message):
    instance = _runner(tmp_path)
    checkpoint = tmp_path / 'bad.pt'
    _checkpoint(checkpoint, instance.args, torch.nn.Linear(3, 2), **override)
    with pytest.raises(RuntimeError, match=message):
        instance._load_goal_pretrain_checkpoint(str(checkpoint))


def test_goal_checkpoint_rejects_non_goal_parameters(tmp_path):
    instance = _runner(tmp_path)
    checkpoint = tmp_path / 'bad_keys.pt'
    payload = _checkpoint(
        checkpoint, instance.args, torch.nn.Linear(3, 2))
    payload['model_state_dict']['diffnet.weight'] = torch.zeros(1)
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match='non-goal parameters'):
        instance._load_goal_pretrain_checkpoint(str(checkpoint))


def test_goal_checkpoint_and_full_pretrain_are_mutually_exclusive(tmp_path):
    instance = _runner(tmp_path)
    instance.args.pretrain_path = 'full_model.pt'
    checkpoint = tmp_path / 'goal.pt'
    _checkpoint(checkpoint, instance.args, torch.nn.Linear(3, 2))
    with pytest.raises(RuntimeError, match='mutually exclusive'):
        instance._load_goal_pretrain_checkpoint(str(checkpoint))


def test_goal_pretrainer_loads_explicit_last_checkpoint(tmp_path):
    instance = object.__new__(goal_pretrainer)
    instance.args = SimpleNamespace(
        model_dir=str(tmp_path), model_name='goal_pretrain')
    instance.device = torch.device('cpu')
    instance.net = torch.nn.Linear(3, 2)
    checkpoint_dir = tmp_path / 'saved_models'
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / 'goal_pretrain_last_model.pt'
    torch.save({
        'epoch': 17,
        'checkpoint_type': 'gdts_goal_pretrain',
        'model_state_dict': instance.net.state_dict(),
    }, checkpoint)

    assert instance._load_checkpoint('last') == 17
    assert instance._pending_training_state['epoch'] == 17


def test_joint_trainer_resolves_explicit_last_checkpoint(tmp_path):
    instance = object.__new__(trainer)
    instance.args = SimpleNamespace(model_dir=str(tmp_path))
    checkpoint_dir = tmp_path / 'saved_models'
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / 'last_model.pt'
    checkpoint.touch()
    loaded = []

    def load_state_file(path, weights_only=False):
        assert weights_only is False
        loaded.append(path)
        return 23

    instance._load_state_file = load_state_file
    assert instance._load_checkpoint('last') == 23
    assert loaded == [str(checkpoint)]


def test_goal_validation_bce_uses_raw_logits():
    model = object.__new__(Goal_Pretrain)
    torch.nn.Module.__init__(model)
    model.args = SimpleNamespace(obs_length=1, down_factor=8)
    model.device = torch.device('cpu')
    model.compute_loss_mask = lambda seq_list, obs_length: torch.ones(1)
    logits = torch.tensor([[[[[0.25]]]]])
    inputs = {
        'input_traj_maps': torch.tensor([[[[0.0]], [[1.0]]]])}
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(0), inputs['input_traj_maps'][:, 1:])

    actual = model.compute_model_metrics(
        'goal_BCE', logits, torch.empty(0), torch.empty(0),
        torch.empty(0), inputs)

    assert actual == pytest.approx([expected.item()])


def test_goal_pretrain_real_loss_path_backpropagates():
    model = object.__new__(Goal_Pretrain)
    torch.nn.Module.__init__(model)
    model.args = SimpleNamespace(obs_length=2)
    model.device = torch.device('cpu')
    logits = torch.zeros(1, 2, 3, 2, 2, requires_grad=True)
    model.goal_logit_map_prediction = lambda inputs: logits
    inputs = {
        'input_traj_maps': torch.zeros(2, 5, 2, 2),
    }
    seq_list = torch.tensor([
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])

    losses = model.get_loss(inputs, seq_list)
    losses['goal_BCE_loss'].backward()

    assert torch.isfinite(losses['goal_BCE_loss'])
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
