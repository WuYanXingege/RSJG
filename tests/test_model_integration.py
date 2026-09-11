"""CPU integration checks for the five goal-model ablations.

The original U-Net and diffusion transformer are intentionally replaced by
small shape-compatible modules.  The graph, social encoder, latent-mode,
relation-energy, sampler, and joint-goal loss implementations remain real so
this file exercises their wiring through :class:`GDTS`.
"""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from src import parser as parser_module


GOAL_MODEL_TYPES = ("independent", "social", "lowrank", "energy", "joint")

EXPECTED_LOSS_KEYS = {
    "independent": {"diffusion_loss", "goal_BCE_loss"},
    "social": {"diffusion_loss", "goal_BCE_loss", "mode_loss"},
    "lowrank": {
        "diffusion_loss", "goal_BCE_loss", "mode_loss",
        "mode_balance_loss",
    },
    "energy": {
        "diffusion_loss", "goal_BCE_loss", "pseudo_likelihood_loss",
        "relation_entropy_loss", "relation_balance_loss",
    },
    "joint": {
        "diffusion_loss", "goal_BCE_loss", "mode_loss",
        "pseudo_likelihood_loss", "relation_entropy_loss",
        "relation_balance_loss", "mode_balance_loss",
    },
}


class _IdentityScene:
    """Minimal scene transform: map pixels and world metres coincide."""

    @staticmethod
    def make_world_coord_torch(coordinates):
        return coordinates.to(dtype=torch.float32)


class _TinyGoalModule(nn.Module):
    def __init__(self, enc_chs, dec_chs, out_chs):
        super().__init__()
        del dec_chs
        self.projection = nn.Conv2d(enc_chs[0], out_chs, kernel_size=1)

    def forward(self, inputs):
        return self.projection(inputs)


class _TinyDiffusionNet(nn.Module):
    def __init__(self, context_dim, tf_layer):
        super().__init__()
        del context_dim, tf_layer
        self.input_scale = nn.Parameter(torch.tensor(0.1))
        self.context_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, inputs, beta, context):
        del beta
        context_scalar = context.mean(dim=(1, 2), keepdim=True)
        return self.input_scale * inputs + self.context_scale * context_scalar


@pytest.fixture
def model_module(monkeypatch):
    """Remove heavyweight/data-dependent baseline pieces from these tests."""

    scene = _IdentityScene()
    dataset = SimpleNamespace(scenes={"dummy": scene})
    # Importing the production model normally imports every dataset backend.
    # Substitute only the factory module before that import so this integration
    # test has no pandas/dataset dependency at collection time.
    factory_module = ModuleType(
        "src.data_src.dataset_src.dataset_create")
    factory_module.create_dataset = lambda _name, **_kwargs: dataset
    monkeypatch.setitem(
        sys.modules, "src.data_src.dataset_src.dataset_create", factory_module)
    current_model_module = importlib.import_module("src.models.model")

    monkeypatch.setattr(
        current_model_module, "create_dataset",
        lambda _name, **_kwargs: dataset)
    monkeypatch.setattr(current_model_module, "UNet", _TinyGoalModule)
    monkeypatch.setattr(
        current_model_module, "TransformerConcatLinear", _TinyDiffusionNet)
    return current_model_module


def _make_args(goal_model_type, tmp_path):
    parser = parser_module.get_parser()
    args = parser.parse_args([
        "--dataset", "sdd",
        "--device", "cpu",
        "--goal_model_type", goal_model_type,
        "--training_stage", "finetune",
        "--e_dim", "8",
        "--ddpm_step", "4",
        "--ddim_step", "2",
        "--trunk_stage_step", "2",
        "--num_samples", "2",
        "--num_joint_samples", "2",
        "--num_goal_candidates", "3",
        "--num_social_modes", "2",
        "--num_relation_modes", "3",
        "--goal_pair_rank", "2",
        "--social_feature_dim", "8",
        "--social_attention_layers", "1",
        "--num_refinement_steps", "1",
        "--graph_type", "full",
        "--down_factor", "1",
        "--use_ttst", "False",
        "--data_augmentation", "False",
        "--use_wandb", "False",
        # Non-legacy values make the independent coefficient assertion prove
        # that the baseline still uses its historical constants 1 and 20.
        "--lambda_diff", "0.5",
        "--lambda_goal", "7.0",
        "--lambda_mode", "0.6",
        "--lambda_PL", "0.7",
        "--lambda_relation_entropy", "0.1",
        "--lambda_relation_balance", "0.2",
        "--lambda_mode_balance", "0.3",
    ])
    args = parser_module.check_and_add_additional_args(args)
    args.model_dir = str(tmp_path / goal_model_type)
    return args


def _make_inputs(num_agents=3, scene_index=None, spatial_size=5):
    """Create a synchronized 20-step scene batch without reading a dataset."""

    num_steps, height, width = 20, spatial_size, spatial_size
    time = torch.arange(num_steps, dtype=torch.float32)
    agent = torch.arange(num_agents, dtype=torch.float32)

    absolute = torch.zeros(num_steps, num_agents, 2)
    absolute[..., 0] = 0.10 * time[:, None] + 0.30 * agent[None, :]
    absolute[..., 1] = 0.20 + 0.35 * agent[None, :]

    augmented = torch.zeros(num_steps, num_agents, 8)
    augmented[..., 6:8] = absolute
    augmented[1:, :, 2:4] = absolute[1:] - absolute[:-1]
    augmented[0, :, 2:4] = augmented[1, :, 2:4]

    trajectory_maps = torch.zeros(num_agents, num_steps, height, width)
    x_index = absolute[..., 0].round().long().clamp(0, width - 1)
    y_index = absolute[..., 1].round().long().clamp(0, height - 1)
    for step in range(num_steps):
        trajectory_maps[
            torch.arange(num_agents), step, y_index[step], x_index[step]
        ] = 1.0

    if scene_index is None:
        # A two-agent scene plus a singleton scene exercises compact scene-id
        # mapping and prevents accidental cross-scene edges.
        if num_agents == 3:
            scene_index = torch.tensor([4, 4, 9], dtype=torch.long)
        else:
            scene_index = torch.zeros(num_agents, dtype=torch.long)

    return {
        "x_augmented": augmented,
        "tensor_image": torch.zeros(6, height, width),
        "input_traj_maps": trajectory_maps,
        "world_coord": absolute,
        "obs_traj_world": absolute[:8].permute(1, 0, 2).contiguous(),
        "scene_index": scene_index,
        "scene": _IdentityScene(),
    }, torch.ones(num_steps, num_agents)


def _assert_finite_losses(losses, expected_keys):
    assert set(losses) == expected_keys
    for value in losses.values():
        assert value.ndim == 0
        assert torch.isfinite(value)


def _make_v4_args(tmp_path):
    args = _make_args('joint', tmp_path)
    args.training_stage = 'multiway_coupling'
    args.trajectory_coupling = 'multiway_v4'
    args.upstream_generator = 'stage0_independent'
    args.freeze_upstream_generator = True
    args.trajectory_dim = 16
    args.trajectory_pair_rank = 4
    args.sync_iterations = 2
    args.sinkhorn_iterations = 4
    args.hard_projection = 'hungarian'
    args.keep_threshold = 0.0
    args.trajectory_alignment_top_k_edges = 0
    args.lambda_gate_reg = 0.0
    return args


def test_v4_integrated_loss_gradients_and_hard_set_preservation(
        tmp_path, model_module, monkeypatch):
    """The formal stage uses real samples, trains V4, and gathers only."""
    torch.manual_seed(101)
    args = _make_v4_args(tmp_path)
    # Avoid invoking clustering in this small wiring test while retaining the
    # exact legacy Stage0 goal/context path.
    monkeypatch.setattr(
        model_module, 'TTST_test_time_sampling_trick',
        lambda heatmap, num_goals, device: torch.zeros(
            num_goals + 1, heatmap.shape[0], 1, 2, device=device))
    model = model_module.GDTS(args, torch.device('cpu'))
    inputs, seq_list = _make_inputs()

    losses = model.get_loss(inputs, seq_list)
    assert set(losses) == {
        'loss_alignment', 'loss_pair_score', 'loss_pair_assignment',
        'loss_no_harm', 'loss_perm_entropy', 'loss_relation_prior',
        'loss_gate_reg'}
    assert all(torch.isfinite(value) for value in losses.values())
    weighted = sum(
        model.set_losses_coeffs()[name] * value
        for name, value in losses.items())
    weighted.backward()
    for module in (
            model.multiway_coupler.trajectory_encoder,
            model.multiway_coupler.trajectory_relation,
            model.multiway_coupler.trajectory_energy,
            model.multiway_coupler.permutation_synchronizer.keep_head):
        gradients = [
            parameter.grad for parameter in module.parameters()
            if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.abs().sum() > 0 for gradient in gradients)
    assert not any(
        parameter.requires_grad for module in model._baseline_modules()
        for parameter in module.parameters())

    model.eval()
    predictions, auxiliary = model(inputs, if_test=True)
    permutation = auxiliary['multiway_coupling_permutation']
    expected = torch.arange(args.num_samples).expand(
        inputs['obs_traj_world'].shape[0], -1)
    assert torch.equal(permutation.sort(dim=-1).values, expected)
    # Both marginal metric pairs are explicitly asserted in the diagnostics.
    diagnostics = model.multiway_batch_diagnostics(
        auxiliary, inputs,
        torch.ones(inputs['obs_traj_world'].shape[0], dtype=torch.bool))
    assert diagnostics['scalar']['marginal_ADE_max_abs_error'] == 0
    assert diagnostics['scalar']['marginal_FDE_max_abs_error'] == 0


@pytest.mark.parametrize("goal_model_type", GOAL_MODEL_TYPES)
def test_all_goal_model_types_encode_and_get_loss(
        goal_model_type, tmp_path, model_module):
    """Every paper ablation runs its actual integrated CPU loss path."""

    torch.manual_seed(17)
    args = _make_args(goal_model_type, tmp_path)
    model = model_module.GDTS(args, torch.device("cpu"))
    inputs, seq_list = _make_inputs()

    context, auxiliary = model.encode(
        inputs, if_test=False,
        for_loss=goal_model_type != "independent")
    assert context.shape == (1, 3, 1, args.e_dim)
    assert auxiliary["goal_point"].shape == (1, 3, 2)
    assert torch.isfinite(context).all()

    losses = model.get_loss(inputs, seq_list, t=1)
    _assert_finite_losses(losses, EXPECTED_LOSS_KEYS[goal_model_type])

    if goal_model_type == "joint":
        sum(losses.values()).backward()
        for joint_module in (
                model.joint_goal_model, model.relation_inference,
                model.goal_energy):
            gradients = [
                parameter.grad for parameter in joint_module.parameters()
                if parameter.grad is not None
            ]
            assert gradients
            assert all(torch.isfinite(gradient).all() for gradient in gradients)
            assert any(gradient.abs().sum() > 0 for gradient in gradients)


def test_joint_test_time_goal_layout_preserves_scene_sample_semantics(
        tmp_path, model_module):
    """Axis 0 is a complete joint scene sample, followed by one MAP trunk."""

    torch.manual_seed(23)
    args = _make_args("joint", tmp_path)
    model = model_module.GDTS(args, torch.device("cpu"))
    inputs, _ = _make_inputs()

    context, auxiliary = model.encode(inputs, if_test=True)
    num_iterations = args.num_samples + 1
    assert context.shape == (num_iterations, 3, 1, args.e_dim)
    assert auxiliary["goal_point"].shape == (num_iterations, 3, 2)
    assert auxiliary["joint_goal_points_map"].shape == (
        3, num_iterations, 2)
    assert torch.equal(
        auxiliary["goal_point"].permute(1, 0, 2),
        auxiliary["joint_goal_points_map"],
    )
    # Coverage sampling avoids duplicate branch candidates for every agent.
    for row in auxiliary["joint_candidate_index"][:, :args.num_samples]:
        assert row.unique().numel() == args.num_samples
    # Without TTST, the shared tree trunk uses the heatmap-MAP candidate.
    torch.testing.assert_close(
        auxiliary["joint_candidate_index"][:, -1],
        auxiliary["candidate_heatmap_prob"].argmax(dim=-1),
    )

    selected = auxiliary["goal_candidates_map"].gather(
        1,
        auxiliary["joint_candidate_index"].unsqueeze(-1).expand(-1, -1, 2),
    )
    assert torch.equal(selected, auxiliary["joint_goal_points_map"])
    # One social-mode draw per scene/sample (not one unrelated draw per agent).
    assert auxiliary["sampled_social_mode"].shape == (2, num_iterations)


def test_full_joint_complete_diffusion_tree_forward_is_finite(
        tmp_path, model_module):
    """The joint endpoints drive the unchanged trunk/branch forward API."""
    torch.manual_seed(27)
    args = _make_args('joint', tmp_path)
    model = model_module.GDTS(args, torch.device('cpu')).eval()
    inputs, _ = _make_inputs()

    with torch.no_grad():
        predictions, auxiliary = model(inputs, if_test=True)

    assert predictions.shape == (
        args.num_samples, args.seq_length, 3, 2)
    assert torch.isfinite(predictions).all()
    assert auxiliary['goal_point'].shape == (
        args.num_samples + 1, 3, 2)
    torch.testing.assert_close(
        predictions[:, :args.obs_length],
        inputs['x_augmented'][:args.obs_length, :, 6:8].unsqueeze(0).expand(
            args.num_samples, -1, -1, -1))


def test_full_joint_single_agent_empty_graph_is_finite(tmp_path, model_module):
    """N=1 / E=0 works through sampling and all loss terms."""

    torch.manual_seed(29)
    args = _make_args("joint", tmp_path)
    model = model_module.GDTS(args, torch.device("cpu"))
    inputs, seq_list = _make_inputs(
        num_agents=1, scene_index=torch.tensor([37]))

    _, auxiliary = model.encode(inputs, if_test=True, for_loss=True)
    assert auxiliary["edge_index"].shape == (2, 0)
    assert auxiliary["relation_prob"].shape == (0, args.num_relation_modes)
    assert auxiliary["energy_output"]["effective_energy"].shape == (
        0, args.num_goal_candidates, args.num_goal_candidates)
    assert auxiliary["goal_point"].shape == (args.num_samples + 1, 1, 2)

    losses = model.get_loss(inputs, seq_list, t=torch.tensor([1]))
    _assert_finite_losses(losses, EXPECTED_LOSS_KEYS["joint"])


def test_independent_state_dict_and_baseline_loss_contract_are_unchanged(
        tmp_path, model_module):
    """GDTS-Base keeps its legacy namespace, keys, and objective weights."""

    args = _make_args("independent", tmp_path)
    model = model_module.GDTS(args, torch.device("cpu"))
    forbidden_prefixes = (
        "interaction_graph.", "social_encoder.", "joint_goal_model.",
        "relation_inference.", "goal_energy.", "joint_sampler.",
    )
    state_keys = tuple(model.state_dict())
    assert not any(
        key.startswith(prefix)
        for key in state_keys for prefix in forbidden_prefixes
    )
    assert set(model.init_losses()) == {"diffusion_loss", "goal_BCE_loss"}
    assert model.set_losses_coeffs() == {
        "diffusion_loss": 1,
        "goal_BCE_loss": 20,
    }
    assert model.best_valid_metric() == 'ADE'


def test_structured_auto_checkpoint_metric_is_world_joint(
        tmp_path, model_module):
    args = _make_args('joint', tmp_path)
    model = model_module.GDTS(args, torch.device('cpu'))
    assert model.best_valid_metric() == 'JADE'


def test_usage_balance_uses_other_scene_history_not_per_scene_uniformity(
        tmp_path, model_module):
    args = _make_args('joint', tmp_path)
    model = model_module.GDTS(args, torch.device('cpu')).train()

    def loss_after_mode_zero_history(current):
        model._mode_usage_history.zero_()
        model._mode_usage_history[0] = torch.tensor([1.0, 0.0])
        model._mode_usage_history_count.fill_(1)
        model._mode_usage_history_pointer.fill_(1)
        probability = torch.tensor([current], requires_grad=True)
        loss = model._usage_balance_loss(probability, family='mode')
        loss.backward()
        assert torch.isfinite(probability.grad).all()
        return float(loss.detach())

    complementary = loss_after_mode_zero_history([0.0, 1.0])
    repeated = loss_after_mode_zero_history([1.0, 0.0])
    assert complementary < repeated


def test_goal_recall_uses_k_candidates_not_s_selected_branches(
        tmp_path, model_module):
    args = _make_args('joint', tmp_path)
    args.goal_recall_threshold_meter = 1e-4
    model = model_module.GDTS(args, torch.device('cpu'))
    inputs, _ = _make_inputs()
    ground_truth_goal = inputs['world_coord'][-1]
    auxiliary = {
        # All S branch goals deliberately miss.
        'goal_point': torch.full(
            (args.num_samples + 1, 3, 2), 100.0),
        # Candidate K-1 is exact for every agent.
        'goal_candidates_world': torch.cat((
            torch.full((3, args.num_goal_candidates - 1, 2), 100.0),
            ground_truth_goal[:, None, :]), dim=1),
    }
    predictions = torch.zeros(
        args.num_samples, args.seq_length, 3, 2)
    recalled = model.compute_model_metrics(
        'Goal_Recall@K', predictions, torch.ones(3, dtype=torch.bool),
        auxiliary, inputs, obs_length=args.obs_length)
    assert recalled == [1.0, 1.0, 1.0]


def test_adaptive_rank_and_continuous_refinement_train_together(
        tmp_path, model_module):
    """The proposed modules add finite losses and receive gradients."""
    torch.manual_seed(47)
    args = _make_args('joint', tmp_path)
    args.adaptive_graph = True
    args.adaptive_graph_hidden_dim = 8
    args.adaptive_graph_top_k = 2
    args.lambda_graph_density = 0.01
    args.lambda_joint_rank = 0.1
    args.continuous_refinement = True
    args.continuous_refinement_steps = 1
    args.lambda_continuous_refinement = 0.5
    args.lambda_refinement_delta = 0.01
    args.baseline_unfreeze_epoch = 2
    args.baseline_lr_scale = 0.2
    model = model_module.GDTS(args, torch.device('cpu'))
    inputs, seq_list = _make_inputs()

    assert not any(
        parameter.requires_grad
        for module in model._baseline_modules()
        for parameter in module.parameters())
    assert any(
        parameter.requires_grad
        for parameter in model.continuous_goal_refiner.parameters())
    families = model.optimizer_parameter_families()
    assert families['baseline'] and families['structured']

    losses = model.get_loss(inputs, seq_list, t=1)
    expected_new = {
        'joint_rank_loss', 'graph_density_loss',
        'continuous_refinement_loss', 'refinement_delta_loss',
    }
    assert expected_new.issubset(losses)
    _assert_finite_losses(losses, set(model.init_losses()))
    sum(losses.values()).backward()
    for module in (model.interaction_graph.gate_mlp,
                   model.continuous_goal_refiner):
        gradients = [
            parameter.grad for parameter in module.parameters()
            if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)

    model.configure_training_epoch(2)
    assert all(
        parameter.requires_grad
        for module in model._baseline_modules()
        for parameter in module.parameters())


def test_structured_checkpoint_resume_rejects_missing_module_keys(
        tmp_path, model_module):
    trainer_module = importlib.import_module('src.trainer')
    args = _make_args('joint', tmp_path)
    model = model_module.GDTS(args, torch.device('cpu'))
    checkpoint_state = dict(model.state_dict())
    checkpoint_state.pop(next(
        key for key in checkpoint_state if key.startswith('goal_energy.')))
    checkpoint_path = tmp_path / 'incomplete_joint.pt'
    torch.save({'epoch': 2, 'model_state_dict': checkpoint_state},
               checkpoint_path)

    runner = object.__new__(trainer_module.trainer)
    runner.args = args
    runner.device = torch.device('cpu')
    runner.net = model
    with pytest.raises(RuntimeError):
        runner._load_state_file(
            str(checkpoint_path), baseline_initialization=False)


def test_alignment_stage_trains_only_post_diffusion_permutation_module(
        tmp_path, model_module):
    torch.manual_seed(53)
    args = _make_args('joint', tmp_path)
    args.training_stage = 'alignment'
    args.trajectory_alignment = True
    args.trajectory_alignment_hidden_dim = 16
    args.trajectory_alignment_sinkhorn_iters = 4
    args.trajectory_alignment_steps = 1
    args.trajectory_alignment_temperature = 0.2
    args.trajectory_alignment_target_temperature = 0.5
    args.trajectory_alignment_fde_weight = 1.0
    args.trajectory_alignment_keep_threshold = 0.6
    args.trajectory_alignment_top_k_edges = 2
    args.lambda_trajectory_alignment = 1.0
    args.lambda_trajectory_pair = 0.5
    args.lambda_alignment_no_harm = 1.0
    args.lambda_alignment_entropy = 0.02
    model = model_module.GDTS(args, torch.device('cpu'))
    inputs, seq_list = _make_inputs()

    assert all(
        not parameter.requires_grad
        for module in model._baseline_modules()
        for parameter in module.parameters())
    assert all(
        not parameter.requires_grad
        for module in (model.social_encoder, model.joint_goal_model,
                       model.relation_inference, model.goal_energy)
        for parameter in module.parameters())
    assert all(
        parameter.requires_grad
        for parameter in model.trajectory_aligner.parameters())

    model.train()
    losses = model.get_loss(inputs, seq_list)
    assert set(losses) == {
        'trajectory_alignment_loss', 'trajectory_pair_loss',
        'alignment_no_harm_loss', 'alignment_entropy_loss'}
    weighted = sum(
        model.set_losses_coeffs()[name] * value
        for name, value in losses.items())
    assert torch.isfinite(weighted)
    weighted.backward()
    gradients = [
        parameter.grad for parameter in model.trajectory_aligner.parameters()
        if parameter.grad is not None]
    assert gradients and any(gradient.abs().sum() > 0 for gradient in gradients)


def test_real_backbones_full_joint_forward_backward_is_finite(
        tmp_path, monkeypatch):
    """Exercise the production U-Net, history LSTM and diffusion denoiser."""
    from src.models.diffusion import TransformerConcatLinear as RealDiffusion
    from src.models.model_utils.U_net_CNN import UNet as RealUNet

    scene = _IdentityScene()
    dataset = SimpleNamespace(scenes={"dummy": scene})
    factory_module = ModuleType(
        "src.data_src.dataset_src.dataset_create")
    factory_module.create_dataset = lambda _name, **_kwargs: dataset
    monkeypatch.setitem(
        sys.modules, "src.data_src.dataset_src.dataset_create", factory_module)
    current_model_module = importlib.import_module("src.models.model")
    monkeypatch.setattr(
        current_model_module, "create_dataset",
        lambda _name, **_kwargs: dataset)
    # Earlier parametrized tests temporarily install lightweight doubles in
    # the cached module. Explicitly restore the production implementations.
    monkeypatch.setattr(current_model_module, "UNet", RealUNet)
    monkeypatch.setattr(
        current_model_module, "TransformerConcatLinear", RealDiffusion)

    torch.manual_seed(31)
    args = _make_args("joint", tmp_path)
    model = current_model_module.GDTS(args, torch.device("cpu"))
    inputs, seq_list = _make_inputs(
        num_agents=2, scene_index=torch.zeros(2, dtype=torch.long),
        spatial_size=64)

    losses = model.get_loss(inputs, seq_list, t=torch.tensor([1, 2]))
    _assert_finite_losses(losses, EXPECTED_LOSS_KEYS["joint"])
    weighted_total = sum(
        model.set_losses_coeffs()[name] * value
        for name, value in losses.items())
    weighted_total.backward()

    for module in (
            model.goal_module, model.registrar, model.diffnet,
            model.social_encoder, model.joint_goal_model,
            model.relation_inference, model.goal_energy):
        gradients = [
            parameter.grad for parameter in module.parameters()
            if parameter.grad is not None]
        assert gradients, type(module).__name__
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.abs().sum() > 0 for gradient in gradients)
