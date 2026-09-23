import importlib
import json
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from src import parser as parser_module
from src.models.joint_dependency_v2 import DependencyCorrector


ROOT = Path(__file__).resolve().parents[1]
STAGE_A_CONFIG = ROOT / "configs/joint_dependency_v2/jdv2_stage_a_frozen_eth.yaml"
STAGE_B_CONFIG = ROOT / "configs/joint_dependency_v2/jdv2_stage_b_v1_eth.yaml"
FREEZE_MANIFEST = ROOT / (
    "outputs/joint_dependency_v2/eth/joint_dependency_v2/"
    "stage_a_freeze/manifest.json")


class _IdentityScene:
    @staticmethod
    def make_world_coord_torch(coordinates):
        return coordinates.float()

    @staticmethod
    def make_pixel_coord_torch(coordinates):
        return coordinates.float()


class _TinyGoalModule(nn.Module):
    def __init__(self, enc_chs, dec_chs, out_chs):
        super().__init__()
        del dec_chs
        self.projection = nn.Conv2d(enc_chs[0], out_chs, 1)

    def forward(self, inputs):
        return self.projection(inputs)


class _TinyDiffusionNet(nn.Module):
    def __init__(self, context_dim, tf_layer):
        super().__init__()
        del context_dim, tf_layer
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs, beta, context):
        del beta, context
        return self.scale * inputs


@pytest.fixture
def model_module(monkeypatch):
    scene = _IdentityScene()
    dataset = SimpleNamespace(scenes={"dummy": scene})
    factory = ModuleType("src.data_src.dataset_src.dataset_create")
    factory.create_dataset = lambda _name, **_kwargs: dataset
    monkeypatch.setitem(
        sys.modules, "src.data_src.dataset_src.dataset_create", factory)
    module = importlib.import_module("src.models.model")
    monkeypatch.setattr(module, "create_dataset",
                        lambda _name, **_kwargs: dataset)
    monkeypatch.setattr(module, "UNet", _TinyGoalModule)
    monkeypatch.setattr(module, "TransformerConcatLinear", _TinyDiffusionNet)

    def candidates(probability_map, num_candidates, device, use_ttst):
        del use_ttst
        n = probability_map.shape[0]
        value = torch.linspace(0.1, 3.1, num_candidates, device=device)
        goals = torch.stack((value, value.flip(0)), dim=-1)
        goals = goals[None].expand(n, -1, -1).contiguous()
        probability = torch.softmax(
            torch.linspace(-1, 1, num_candidates, device=device), dim=0)
        return goals, probability[None].expand(n, -1).contiguous()

    monkeypatch.setattr(module, "generate_goal_candidates", candidates)
    return module


def _args(tmp_path, stage="joint_trajectory", exact=True):
    values = [
        "--dataset", "sdd", "--device", "cpu",
        "--goal_model_type", "joint_dependency_v2",
        "--training_stage", stage, "--data_augmentation", "False",
        "--use_scene_latent", "False",
        "--jdv2_latent_objective", "strict_no_z",
        "--use_wandb", "False", "--fast_debug", "True",
        "--e_dim", "8", "--ddpm_step", "4", "--ddim_step", "2",
        "--trunk_stage_step", "2", "--graph_type", "full",
        "--down_factor", "1", "--use_ttst", "False",
    ]
    if exact:
        values.extend([
            "--jdv2_refinement_policy",
            "exact_lexicographic_persistent_tie"])
    args = parser_module.get_parser().parse_args(values)
    args = parser_module.check_and_add_additional_args(args)
    args.model_dir = str(tmp_path / stage)
    return args


def _inputs(n=3, size=8, one_scene=False):
    time = torch.arange(20).float()
    agent = torch.arange(n).float()
    absolute = torch.zeros(20, n, 2)
    absolute[..., 0] = 0.1 * time[:, None] + 0.2 * agent[None]
    absolute[..., 1] = 0.2 + 0.3 * agent[None]
    augmented = torch.zeros(20, n, 8)
    augmented[..., 6:8] = absolute
    augmented[1:, :, 2:4] = absolute[1:] - absolute[:-1]
    augmented[0, :, 2:4] = augmented[1, :, 2:4]
    scene_index = torch.zeros(n, dtype=torch.long)
    if n == 3 and not one_scene:
        scene_index = torch.tensor([4, 4, 9])
    return {
        "x_augmented": augmented,
        "tensor_image": torch.zeros(6, size, size),
        "input_traj_maps": torch.zeros(n, 20, size, size),
        "world_coord": absolute,
        "obs_traj_world": absolute[:8].permute(1, 0, 2).contiguous(),
        "scene_index": scene_index,
        "scene": _IdentityScene(),
    }, torch.ones(20, n)


def _set_context(model, seed=2035, epoch=1, batch=0, batches=3):
    return model.set_jdv2_training_sampling_context(
        seed, epoch, batch, batches)


def test_dependency_corrector_architecture_count_and_initial_zero():
    model = DependencyCorrector()
    assert sum(parameter.numel() for parameter in model.parameters()) == 30851
    assert model.state_encoder[0].in_features == 21
    assert model.timestep_network[-1].out_features == 128
    assert model.output[-1].out_features == 2
    velocity = torch.randn(3, 12, 2)
    edge = torch.tensor([[0, 1], [1, 2]])
    result = model(
        velocity, torch.randn(3, 2), edge, torch.randn(2, 16),
        torch.tensor([30, 30, 30]))
    assert torch.equal(result, torch.zeros_like(result))


def test_dependency_corrector_e0_is_tensor_exact_zero():
    model = DependencyCorrector()
    velocity = torch.randn(2, 12, 2)
    result = model(
        velocity, torch.randn(2, 2), torch.empty(2, 0, dtype=torch.long),
        torch.empty(0, 16), torch.tensor([30, 25]))
    assert torch.equal(result, torch.zeros_like(result))


def test_parser_accepts_strict_no_z_stage_b_and_rejects_finetune(tmp_path):
    args = _args(tmp_path)
    assert args.training_stage == "joint_trajectory"
    with pytest.raises(ValueError, match="strict_no_z authorizes only"):
        _args(tmp_path, stage="joint_finetune")


def test_stage_b_config_and_frozen_stage_a_config_contract():
    before = yaml.safe_load(STAGE_A_CONFIG.read_text())
    stage_b = yaml.safe_load(STAGE_B_CONFIG.read_text())
    assert before["training_stage"] == "joint_goal"
    assert stage_b["training_stage"] == "joint_trajectory"
    assert stage_b["stage_b_architecture_version"] == "jdv2-stage-b-v1"
    assert stage_b["jdv2_refinement_policy"] == \
        "exact_lexicographic_persistent_tie"
    assert stage_b["lambda_diff"] == 1.0
    assert stage_b["lambda_relative"] == 0.05
    manifest = json.loads(FREEZE_MANIFEST.read_text())
    assert stage_b["stage_a_parent_checkpoint_sha256"] == \
        manifest["checkpoint"]["sha256"]
    parser = parser_module.get_parser()
    known = set(vars(parser_module.check_and_add_additional_args(
        parser.parse_args(["--device", "cpu", "--data_augmentation", "False"]))))
    assert not (set(stage_b) - known)
    parser.set_defaults(**stage_b)
    loaded = parser_module.check_and_add_additional_args(parser.parse_args([]))
    assert loaded.training_stage == "joint_trajectory"
    assert loaded.branch_stage_step == 14


def test_active_corrector_timestep_derivation(model_module):
    assert model_module.jdv2_active_corrector_timesteps(100, 20, 14) == \
        (30, 25, 20, 15, 10, 5)


def test_scene_consistent_timesteps_and_invalid_rejection(model_module):
    scene = torch.tensor([4, 4, 9, 9, 9])
    active = (30, 25, 20, 15, 10, 5)
    timestep = model_module.jdv2_scene_timesteps(
        scene, active, supplied_t=torch.tensor([25, 10]))
    assert timestep.tolist() == [25, 25, 10, 10, 10]
    with pytest.raises(ValueError, match="share one timestep"):
        model_module.jdv2_scene_timesteps(
            scene, active, supplied_t=torch.tensor([25, 20, 10, 10, 10]))
    with pytest.raises(ValueError, match="not inference-active"):
        model_module.jdv2_scene_timesteps(scene, active, supplied_t=99)


def test_scene_oracle_branch_and_relation_gather_are_scene_shared(model_module):
    goals = torch.tensor([
        [[0., 0.], [5., 0.]],
        [[0., 0.], [5., 0.]],
        [[9., 0.], [2., 0.]],
        [[9., 0.], [2., 0.]],
    ])
    target = torch.tensor([[4., 0.], [4., 0.], [2., 0.], [2., 0.]])
    scene = torch.tensor([7, 7, 11, 11])
    edge = torch.tensor([[0, 2], [1, 3]])
    relation = torch.arange(2 * 2 * 16).reshape(2, 2, 16).float()
    selected = model_module.jdv2_select_scene_oracle_branch(
        goals, target, scene, edge, relation)
    assert selected["scene_branch"].tolist() == [1, 1]
    assert selected["agent_branch"].tolist() == [1, 1, 1, 1]
    assert selected["edge_branch"].tolist() == [1, 1]
    assert torch.equal(
        selected["selected_relation_embedding"], relation[:, 1])


def test_scene_oracle_rejects_cross_scene_edge(model_module):
    with pytest.raises(ValueError, match="cross scene"):
        model_module.jdv2_select_scene_oracle_branch(
            torch.zeros(2, 20, 2), torch.zeros(2, 2), torch.tensor([0, 1]),
            torch.tensor([[0], [1]]), torch.zeros(1, 20, 16))


def test_only_corrector_trainable_and_frozen_modules_eval(tmp_path, model_module):
    model = model_module.GDTS(_args(tmp_path), torch.device("cpu")).train()
    trainable = [name for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    assert trainable
    assert all(name.startswith("jdv2_corrector.") for name in trainable)
    assert sum(dict(model.named_parameters())[name].numel()
               for name in trainable) == 30851
    assert model.jdv2_corrector.training
    assert all(not module.training for module in model._baseline_modules())
    assert all(not getattr(model, name).training
               for name in model._jdv2_goal_module_names())


def test_training_context_is_resume_safe_and_does_not_consume_global_rng(
        tmp_path, model_module):
    model = model_module.GDTS(_args(tmp_path), torch.device("cpu"))
    torch.manual_seed(17)
    before = torch.get_rng_state().clone()
    serial = _set_context(model, epoch=3, batch=2, batches=7)
    after = torch.get_rng_state()
    assert serial == 16
    assert torch.equal(before, after)
    assert model.jdv2_sampler._sampling_context == (2035, 16)


def test_exact_stage_a_sampling_is_independent_of_global_rng(
        tmp_path, model_module):
    model = model_module.GDTS(_args(tmp_path), torch.device("cpu")).eval()
    inputs, _ = _inputs(one_scene=True)
    goal_probability = torch.full((3, 1, 8, 8), 0.5)

    def sample(global_seed):
        random.seed(global_seed)
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        model.jdv2_sampler.set_sampling_context(2035, 7)
        with torch.no_grad():
            result = model._jdv2_goal_outputs(
                inputs, goal_probability, sample=True,
                include_teacher=False)["sampled"]["candidate_index"]
        assert random.getstate() == python_state
        after_numpy = np.random.get_state()
        assert after_numpy[0] == numpy_state[0]
        assert np.array_equal(after_numpy[1], numpy_state[1])
        assert after_numpy[2:] == numpy_state[2:]
        assert torch.equal(torch.get_rng_state(), torch_state)
        return result

    assert torch.equal(sample(3), sample(999))


def test_stage_b_e_gt0_forward_backward_and_gradient_freeze(
        tmp_path, model_module):
    torch.manual_seed(31)
    model = model_module.GDTS(_args(tmp_path), torch.device("cpu")).train()
    _set_context(model)
    losses = model.get_loss(*_inputs(one_scene=True), t=2)
    assert set(losses) == {"jdv2_diffusion_loss", "jdv2_relative_loss"}
    total = sum(model.set_losses_coeffs()[name] * value
                for name, value in losses.items())
    total.backward()
    gradients = [parameter.grad for parameter in model.jdv2_corrector.parameters()
                 if parameter.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert any(torch.count_nonzero(value) for value in gradients)
    assert all(parameter.grad is None
               for name, parameter in model.named_parameters()
               if not name.startswith("jdv2_corrector."))
    assert model.last_joint_diagnostics["degree_positive_agent_count"] == 3
    assert model.last_joint_diagnostics["active_timestep_count_2"] == 3


def test_stage_b_e0_losses_are_differentiable_zero(tmp_path, model_module):
    model = model_module.GDTS(_args(tmp_path), torch.device("cpu")).train()
    _set_context(model)
    losses = model.get_loss(*_inputs(n=1), t=2)
    total = sum(losses.values())
    assert total.requires_grad
    assert total.item() == 0.0
    total.backward()
    assert all(parameter.grad is not None
               for parameter in model.jdv2_corrector.parameters())
    assert all(torch.count_nonzero(parameter.grad) == 0
               for parameter in model.jdv2_corrector.parameters())
    assert model.last_joint_diagnostics["e0_skipped_count"] == 1


def test_optimizer_smoke_changes_only_corrector(tmp_path, model_module):
    torch.manual_seed(41)
    model = model_module.GDTS(_args(tmp_path), torch.device("cpu")).train()
    before = {name: parameter.detach().clone()
              for name, parameter in model.named_parameters()}
    _set_context(model)
    losses = model.get_loss(*_inputs(one_scene=True), t=2)
    total = sum(model.set_losses_coeffs()[name] * value
                for name, value in losses.items())
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad], lr=1e-4)
    total.backward()
    optimizer.step()
    changed = {name for name, parameter in model.named_parameters()
               if not torch.equal(before[name], parameter.detach())}
    assert changed
    assert all(name.startswith("jdv2_corrector.") for name in changed)


def test_untrained_corrector_is_exact_no_harm_in_forward(
        tmp_path, model_module):
    model = model_module.GDTS(_args(tmp_path), torch.device("cpu")).eval()
    inputs, _ = _inputs(one_scene=True)
    random.seed(53)
    np.random.seed(53)
    torch.manual_seed(53)
    model.jdv2_sampler.set_sampling_context(2035, 0)
    with_corrector, _ = model(inputs, if_test=True)
    model.args.use_dependency_corrector = False
    random.seed(53)
    np.random.seed(53)
    torch.manual_seed(53)
    model.jdv2_sampler.set_sampling_context(2035, 0)
    without_corrector, _ = model(inputs, if_test=True)
    assert torch.equal(with_corrector, without_corrector)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_stage_b_cuda_bf16_forward_backward(tmp_path, model_module):
    device = torch.device("cuda")
    model = model_module.GDTS(_args(tmp_path), device).to(device).train()
    inputs, sequence = _inputs(one_scene=True)
    inputs = {name: value.to(device) if torch.is_tensor(value) else value
              for name, value in inputs.items()}
    sequence = sequence.to(device)
    _set_context(model)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        losses = model.get_loss(inputs, sequence, t=2)
        total = sum(model.set_losses_coeffs()[name] * value
                    for name, value in losses.items())
    total.backward()
    assert torch.isfinite(total)
    assert any(parameter.grad is not None and
               torch.isfinite(parameter.grad).all()
               for parameter in model.jdv2_corrector.parameters())
