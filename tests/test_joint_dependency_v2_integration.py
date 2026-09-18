import importlib
import random
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src import parser as parser_module
from src.jdv2_audit import summarize_repeated_metrics
from src.utils import isolated_random_seed


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
    monkeypatch.setattr(
        module, "TTST_test_time_sampling_trick",
        lambda heatmap, num_goals, device: torch.zeros(
            num_goals + 1, heatmap.shape[0], 1, 2, device=device))
    return module


def _args(tmp_path, stage="joint_goal", all_off=False):
    values = [
        "--dataset", "sdd", "--device", "cpu",
        "--goal_model_type", "joint_dependency_v2",
        "--training_stage", stage, "--data_augmentation", "False",
        "--use_wandb", "False", "--fast_debug", "True",
        "--e_dim", "8", "--ddpm_step", "4", "--ddim_step", "2",
        "--trunk_stage_step", "2", "--graph_type", "full",
        "--down_factor", "1", "--use_ttst", "False",
    ]
    if all_off:
        values.extend([
            "--use_scene_latent", "False",
            "--use_dynamic_relation", "False",
            "--use_joint_energy", "False",
            "--use_dependency_corrector", "False",
        ])
    args = parser_module.get_parser().parse_args(values)
    args = parser_module.check_and_add_additional_args(args)
    args.model_dir = str(tmp_path / stage)
    return args


def _legacy_args(tmp_path):
    args = parser_module.get_parser().parse_args([
        "--dataset", "sdd", "--device", "cpu",
        "--goal_model_type", "independent", "--training_stage", "baseline",
        "--data_augmentation", "False", "--use_wandb", "False",
        "--e_dim", "8", "--ddpm_step", "4", "--ddim_step", "2",
        "--trunk_stage_step", "2", "--num_samples", "20",
    ])
    args = parser_module.check_and_add_additional_args(args)
    args.model_dir = str(tmp_path / "legacy")
    return args


def _inputs(n=3, size=8):
    time = torch.arange(20).float()
    agent = torch.arange(n).float()
    absolute = torch.zeros(20, n, 2)
    absolute[..., 0] = 0.1 * time[:, None] + 0.2 * agent[None]
    absolute[..., 1] = 0.2 + 0.3 * agent[None]
    augmented = torch.zeros(20, n, 8)
    augmented[..., 6:8] = absolute
    augmented[1:, :, 2:4] = absolute[1:] - absolute[:-1]
    augmented[0, :, 2:4] = augmented[1, :, 2:4]
    trajectory_maps = torch.zeros(n, 20, size, size)
    scene_index = torch.tensor([0, 0, 1])
    return {
        "x_augmented": augmented,
        "tensor_image": torch.zeros(6, size, size),
        "input_traj_maps": trajectory_maps,
        "world_coord": absolute,
        "obs_traj_world": absolute[:8].permute(1, 0, 2).contiguous(),
        "scene_index": scene_index,
        "scene": _IdentityScene(),
    }, torch.ones(20, n)


def test_stage_a_integrated_loss_and_gradient(tmp_path, model_module):
    torch.manual_seed(11)
    model = model_module.GDTS(
        _args(tmp_path, "joint_goal"), torch.device("cpu"))
    inputs, sequence = _inputs()
    losses = model.get_loss(inputs, sequence)
    assert set(losses) == {
        "jdv2_mixture_pl", "jdv2_posterior_distill",
        "jdv2_relation_kl"}
    total = sum(model.set_losses_coeffs()[name] * value
                for name, value in losses.items())
    assert torch.isfinite(total)
    total.backward()
    assert any(parameter.grad is not None
               for parameter in model.jdv2_joint_energy.parameters())
    assert not any(parameter.requires_grad
                   for module in model._baseline_modules()
                   for parameter in module.parameters())
    diagnostics = model.last_joint_diagnostics
    required = {
        "L_mix", "L_q", "L_PL_post", "L_PL_prior", "L_r", "L_JG",
        "mean_KL_z", "mean_KL_r", "energy_mean", "energy_std",
        "energy_min", "energy_max", "q_z_prior_l1", "KL_gamma_q",
        "between_z_log_score_variance_e_gt0",
    }
    assert required <= diagnostics.keys()
    assert all(torch.isfinite(torch.tensor(diagnostics[name]))
               for name in required)
    assert sum(diagnostics[f"p_z_mean_{index}"]
               for index in range(4)) == pytest.approx(1.0, abs=1e-5)
    assert sum(diagnostics[f"q_z_mean_{index}"]
               for index in range(4)) == pytest.approx(1.0, abs=1e-5)
    assert sum(diagnostics[f"gamma_z_mean_{index}"]
               for index in range(4)) == pytest.approx(1.0, abs=1e-5)
    assert sum(diagnostics[f"predicted_relation_usage_{index}"]
               for index in range(4)) == pytest.approx(1.0, abs=1e-5)
    assert sum(diagnostics[f"teacher_relation_usage_{index}"]
               for index in range(4)) == pytest.approx(1.0, abs=1e-5)


def test_stage_b_corrector_gradient_and_branch_forward(tmp_path, model_module):
    torch.manual_seed(12)
    model = model_module.GDTS(
        _args(tmp_path, "joint_trajectory"), torch.device("cpu"))
    inputs, sequence = _inputs()
    losses = model.get_loss(inputs, sequence, t=torch.tensor([2, 2, 2]))
    total = sum(model.set_losses_coeffs()[name] * value
                for name, value in losses.items())
    total.backward()
    assert model.jdv2_corrector.output[-1].weight.grad is not None
    model.eval()
    prediction, auxiliary = model(inputs, if_test=True)
    assert prediction.shape == (20, 20, 3, 2)
    assert auxiliary["sampled_scene_mode"].shape == (2, 20)
    assert auxiliary["dependency_state"]["relation_embedding"].shape == (
        1, 20, 16)
    diagnostics = model.last_joint_diagnostics
    assert sum(diagnostics[f"sampled_scene_mode_usage_{index}"]
               for index in range(4)) == pytest.approx(1.0, abs=1e-5)
    assert sum(diagnostics[f"sampled_relation_usage_{index}"]
               for index in range(4)) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("amp_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_stage_a_full_loss_gpu_autocast_keeps_fp32_reductions(
        amp_dtype, tmp_path, model_module):
    """Regress mixed-dtype sparse index_add/einsum failures under AMP."""
    args = _args(tmp_path, "joint_goal")
    device = torch.device("cuda")
    model = model_module.GDTS(args, device).to(device).train()
    inputs, sequence = _inputs()
    inputs = {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in inputs.items()
    }
    sequence = sequence.to(device)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        losses = model.get_loss(inputs, sequence)
        total = sum(model.set_losses_coeffs()[name] * value
                    for name, value in losses.items())
    assert all(value.dtype == torch.float32 for value in losses.values())
    assert torch.isfinite(total)
    total.backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("disabled,zero_loss", [
    ("use_scene_latent", "jdv2_posterior_distill"),
    ("use_dynamic_relation", "jdv2_relation_kl"),
    ("use_joint_energy", None),
])
def test_stage_a_partial_ablation_contracts(
        disabled, zero_loss, tmp_path, model_module):
    args = _args(tmp_path, "joint_goal")
    setattr(args, disabled, False)
    model = model_module.GDTS(args, torch.device("cpu"))
    losses = model.get_loss(*_inputs())
    assert all(torch.isfinite(value) for value in losses.values())
    if zero_loss is not None:
        assert losses[zero_loss].item() == pytest.approx(0.0, abs=1e-6)
    assert "jdv2_mixture_pl" in losses


def test_stage_c_corrector_off_keeps_explicit_goal_objective(
        tmp_path, model_module):
    args = _args(tmp_path, "joint_finetune")
    args.use_dependency_corrector = False
    model = model_module.GDTS(args, torch.device("cpu"))
    losses = model.get_loss(*_inputs())
    assert set(losses) == {
        "jdv2_mixture_pl", "jdv2_posterior_distill",
        "jdv2_relation_kl"}


def test_all_off_is_exact_legacy_passthrough(tmp_path, model_module):
    torch.manual_seed(101)
    legacy = model_module.GDTS(_legacy_args(tmp_path), torch.device("cpu"))
    torch.manual_seed(101)
    all_off = model_module.GDTS(
        _args(tmp_path, "baseline", all_off=True), torch.device("cpu"))
    assert all_off.active_goal_model_type == "independent"
    assert not all_off.jdv2_active
    assert all_off.init_losses() == legacy.init_losses()
    assert all_off.set_losses_coeffs() == legacy.set_losses_coeffs()
    assert all_off.init_test_metrics() == legacy.init_test_metrics()
    assert all_off.best_valid_metric() == legacy.best_valid_metric()
    assert not any(key.startswith("jdv2_")
                   for key in all_off.state_dict())
    all_off.load_state_dict(legacy.state_dict(), strict=True)
    inputs, _ = _inputs()
    torch.manual_seed(202)
    legacy_output, legacy_aux = legacy(inputs, if_test=True)
    torch.manual_seed(202)
    all_off_output, all_off_aux = all_off(inputs, if_test=True)
    assert torch.equal(legacy_output, all_off_output)
    assert legacy_aux.keys() == all_off_aux.keys()
    for key in legacy_aux:
        assert torch.equal(legacy_aux[key], all_off_aux[key])


def test_legacy_checkpoint_allowlist_and_v2_resume_validation(
        tmp_path, model_module):
    trainer_module = importlib.import_module("src.trainer")
    legacy = model_module.GDTS(_legacy_args(tmp_path), torch.device("cpu"))
    args = _args(tmp_path, "joint_goal")
    args.jdv2_cache_manifest_hash = "cache-hash"
    args.jdv2_source_checkpoint_hash = "source-hash"
    v2 = model_module.GDTS(args, torch.device("cpu"))
    shell = trainer_module.trainer.__new__(trainer_module.trainer)
    shell.args = args
    shell.device = torch.device("cpu")
    shell.net = v2
    shell._pending_training_state = None
    shell._best_selection = (float("inf"), float("inf"))

    legacy_path = tmp_path / "legacy.pt"
    torch.save({"epoch": 2, "model_state_dict": legacy.state_dict()},
               legacy_path)
    assert shell._load_state_file(
        str(legacy_path), baseline_initialization=True) == 2

    v2_path = tmp_path / "v2.pt"
    torch.save({
        "epoch": 3,
        "model_state_dict": v2.state_dict(),
        "architecture_config": shell._jdv2_architecture_config(),
        "ablation_config": shell._jdv2_ablation_config(),
        "training_stage": "joint_goal",
        "latent_objective": args.jdv2_latent_objective,
        "cache_manifest_hash": "cache-hash",
        "source_checkpoint_hash": "source-hash",
    }, v2_path)
    assert shell._load_state_file(str(v2_path)) == 3
    assert shell._pending_training_state is not None

    incompatible = torch.load(v2_path)
    incompatible["architecture_config"] = dict(
        incompatible["architecture_config"], goal_candidates=20)
    bad_path = tmp_path / "bad.pt"
    torch.save(incompatible, bad_path)
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        shell._load_state_file(str(bad_path))


def test_isolated_validation_rng_is_reproducible_and_restores_all_cpu_rngs():
    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    with isolated_random_seed(2035, use_cuda=False):
        first = (random.random(), np.random.rand(), torch.rand(4))
    after = (random.random(), np.random.rand(), torch.rand(4))

    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    expected_after = (random.random(), np.random.rand(), torch.rand(4))
    assert after[0] == expected_after[0]
    assert after[1] == expected_after[1]
    assert torch.equal(after[2], expected_after[2])

    with isolated_random_seed(2035, use_cuda=False):
        second = (random.random(), np.random.rand(), torch.rand(4))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_isolated_validation_rng_restores_all_cuda_rngs():
    before = torch.cuda.get_rng_state_all()
    with isolated_random_seed(2035, use_cuda=True):
        first = torch.rand(4, device="cuda")
    after = torch.cuda.get_rng_state_all()
    assert all(torch.equal(left, right) for left, right in zip(before, after))
    with isolated_random_seed(2035, use_cuda=True):
        second = torch.rand(4, device="cuda")
    assert torch.equal(first, second)


def test_repeated_validation_summary_uses_population_mean_and_std():
    summary = summarize_repeated_metrics([
        {"JFDE": 1.0, "JADE": 2.0},
        {"JFDE": 3.0, "JADE": 4.0},
    ])
    assert summary["JFDE"] == {"mean": 2.0, "std": 1.0}
    assert summary["JADE"] == {"mean": 3.0, "std": 1.0}


def test_checkpoint_persists_selection_tables_and_explicit_last(
        tmp_path, model_module):
    trainer_module = importlib.import_module("src.trainer")
    args = _args(tmp_path, "joint_goal")
    args.jdv2_cache_manifest_hash = "cache-hash"
    args.jdv2_source_checkpoint_hash = "source-hash"
    shell = trainer_module.trainer.__new__(trainer_module.trainer)
    shell.args = args
    shell.net = model_module.GDTS(args, torch.device("cpu"))
    shell.scaler = torch.amp.GradScaler("cuda", enabled=False)
    shell._best_selection = (0.7, 0.8)
    shell._stage_optimizer_steps_completed = 123
    shell.best_metrics = {"JADE": 0.4, "JFDE": 0.7}
    shell.best_metrics_epochs = {"JADE": 10, "JFDE": 11}
    shell._save_checkpoint(12, last_epoch=True)
    checkpoint = torch.load(
        tmp_path / "joint_goal" / "saved_models" / "last_model.pt",
        map_location="cpu")
    assert checkpoint["best_selection"] == {
        "primary": 0.7, "tie_break": 0.8}
    assert checkpoint["best_metrics"] == {"JADE": 0.4, "JFDE": 0.7}
    assert checkpoint["best_metrics_epochs"] == {"JADE": 10, "JFDE": 11}
    assert checkpoint["stage_optimizer_steps_completed"] == 123
    assert checkpoint["latent_objective"] == "v2_marginal_responsibility"


def test_early_collapse_gate_requires_joint_same_mode_signature():
    trainer_module = importlib.import_module("src.trainer")
    diagnostics = {
        "p_entropy_e_gt0": 1e-5,
        "q_entropy_e_gt0": 1e-5,
        "gamma_entropy_e_gt0": 1e-5,
        "future_shuffle_l1_e_gt0": 1e-9,
    }
    for family in ("p", "q", "gamma"):
        for mode in range(4):
            diagnostics[f"{family}_usage_{mode}_e_gt0"] = (
                0.997 if mode == 1 else 0.001)
    for mode in range(4):
        diagnostics[f"gamma_hard_usage_{mode}_e_gt0"] = (
            0.97 if mode == 1 else 0.01)
    collapsed, evidence = \
        trainer_module.trainer._jdv2_early_collapse_signature(diagnostics)
    assert collapsed
    assert evidence["dominant_modes"] == {"p": 1, "q": 1, "gamma": 1}
    diagnostics["future_shuffle_l1_e_gt0"] = 1e-3
    collapsed, _ = \
        trainer_module.trainer._jdv2_early_collapse_signature(diagnostics)
    assert not collapsed


def test_old_checkpoint_restores_selection_and_reconstructs_metric_tables(
        tmp_path):
    trainer_module = importlib.import_module("src.trainer")
    curve = tmp_path / "log_curve.txt"
    curve.write_text(
        "epoch,valid_JADE,valid_JFDE\n"
        "1,0.6,0.9\n"
        "11,0.4,0.7\n"
        "12,0.5,0.8\n")
    shell = trainer_module.trainer.__new__(trainer_module.trainer)
    shell.log_curve_file = str(curve)
    shell.best_metrics = {"JADE": 1e9, "JFDE": 1e9}
    shell.best_metrics_epochs = {"JADE": -1, "JFDE": -1}
    shell._best_selection = (float("inf"), float("inf"))
    shell._restore_best_state({
        "epoch": 11,
        "best_metric": {"primary": 0.7, "tie_break": 0.4},
    })
    assert shell._best_selection == (0.7, 0.4)
    assert shell.best_metrics == {"JADE": 0.4, "JFDE": 0.7}
    assert shell.best_metrics_epochs == {"JADE": 11, "JFDE": 11}
    assert not shell._selection_is_better((0.8, 0.3), shell._best_selection)


def test_cross_stage_load_resets_epoch_and_progress_while_same_stage_resumes():
    trainer_module = importlib.import_module("src.trainer")
    shell = trainer_module.trainer.__new__(trainer_module.trainer)
    shell.args = SimpleNamespace(
        load_checkpoint="best", training_stage="joint_trajectory",
        jdv2_stage_progress=0.75)
    shell._pending_training_state = {
        "training_stage": "joint_goal", "stage_progress": 0.25}
    shell._load_checkpoint = lambda _: 11
    shell.curve_metric_names = []
    shell.curve_loss_names = []
    assert shell._load_or_restart() == 1
    assert shell.args.jdv2_stage_progress == 0.0

    shell.args.training_stage = "joint_goal"
    shell.args.jdv2_stage_progress = 0.25
    shell._pending_training_state = {
        "training_stage": "joint_goal", "stage_progress": 0.25}
    assert shell._load_or_restart() == 12
    assert shell.args.jdv2_stage_progress == 0.25


def test_validation_seed_defaults_to_training_seed(tmp_path):
    args = _args(tmp_path)
    assert args.validation_seed == args.seed
