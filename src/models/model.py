import torch
import torch.nn.functional as F
import numpy as np
from contextlib import nullcontext

from src.data_src.dataset_src.dataset_create import create_dataset
from src.models.model_utils.U_net_CNN import UNet
from src.losses import Goal_BCE_loss
from src.metrics import (
    ADE_best_of,
    FDE_best_of,
    JADE,
    JFDE,
    collision_rate,
    goal_minFDE,
    goal_recall_at_K,
    joint_goal_compatibility,
    joint_goal_endpoint_error,
    minADE_at_K,
    minFDE_at_K,
)
from src.models.model_utils.sampling_2D_map import (
    TTST_test_time_sampling_trick,
    generate_goal_candidates,
)
from src.models.diffusion import VarianceSchedule, TransformerConcatLinear, linear_beta_schedule
from src.models.model_utils.hist_traj_rnn_encoder import Encoding
from src.models.model_utils.model_registrar import ModelRegistrar
from src.models.common import derivative_of
from src.models.interaction_graph import EDGE_FEATURE_DIM, SparseInteractionGraph
from src.models.social_encoder import SocialMotionEncoder
from src.models.relation_inference import RelationInference
from src.models.joint_goal import LowRankJointGoal, canonicalize_scene_index
from src.models.goal_energy import LowRankGoalEnergy
from src.models.joint_sampler import JointGoalSampler, estimate_joint_goal_complexity
from src.models.continuous_goal_refiner import ContinuousJointGoalRefiner
from src.models.trajectory_aligner import RelationAwareTrajectoryAligner
from src.models.multiway_trajectory_coupler import (
    RelationAwareMultiwayCoupler,
    gather_trajectory_bank,
)
from src.models.joint_dependency_v2 import (
    DependencyCorrector,
    DynamicHypothesisRelation,
    ParallelConditionalSampler,
    RelationSpecificJointEnergy,
    SceneFutureTeacher,
    SceneLatentPrior,
    UnaryGoalResidual,
)
from src.models.joint_dependency_v2.future_teacher import (
    future_pair_descriptor,
)
from src.multiway_coupling_loss import (
    assert_marginal_preservation,
    compute_multiway_coupling_loss,
    compute_scene_balanced_multiway_coupling_loss,
    hard_alignment_diagnostics,
)
from src.joint_goal_loss import (
    best_joint_continuous_refinement_loss,
    build_soft_goal_target,
    low_rank_social_mode_loss,
    marginal_goal_log_prob,
    pseudo_likelihood_loss,
    scene_joint_ranking_loss,
    jdv2_mixture_composite_loss,
    jdv2_no_z_pseudo_likelihood_from_local,
    jdv2_no_z_relation_kl_per_edge,
    jdv2_posterior_distillation,
    jdv2_relation_kl_per_edge_mode,
    jdv2_scene_mode_log_score,
    jdv2_teacher_probability,
    jdv2_warmup_beta,
    scene_balanced_mean,
    sparse_relative_motion_loss,
)


def jdv2_active_corrector_timesteps(
        ddpm_steps: int, ddim_steps: int,
        branch_stage_step: int) -> tuple[int, ...]:
    """Return the exact branch timesteps at which ``ts_sample`` corrects."""
    if ddpm_steps < 1 or ddim_steps < 1:
        raise ValueError("diffusion schedule lengths must be positive")
    if not 0 <= int(branch_stage_step) < int(ddim_steps):
        raise ValueError("branch_stage_step must lie in [0, ddim_steps)")
    schedule = np.linspace(int(ddpm_steps), 0, int(ddim_steps) + 1)
    return tuple(
        int(schedule[index - 1])
        for index in range(int(branch_stage_step) + 1,
                           int(ddim_steps) + 1))


def jdv2_select_scene_oracle_branch(
        goals: torch.Tensor,
        target_goal: torch.Tensor,
        scene_index: torch.Tensor,
        edge_index: torch.Tensor,
        relation_embedding: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Select one shared Stage-A world per scene and gather its edge state."""
    if goals.ndim != 3 or goals.shape[-1] != 2:
        raise ValueError("goals must have shape [N,P,2]")
    num_agents, num_worlds, _ = goals.shape
    if target_goal.shape != (num_agents, 2):
        raise ValueError("target_goal must have shape [N,2]")
    scene_ids, compact = canonicalize_scene_index(
        scene_index, num_agents, goals.device)
    squared_error = (goals.float() - target_goal.float()[:, None]).square()
    squared_error = squared_error.sum(dim=-1)
    scene_error = squared_error.new_zeros((scene_ids.numel(), num_worlds))
    scene_error.index_add_(0, compact, squared_error)
    counts = torch.bincount(
        compact, minlength=scene_ids.numel()).to(scene_error.dtype)
    scene_error = scene_error / counts[:, None].clamp_min(1)
    scene_branch = scene_error.argmin(dim=-1)
    agent_branch = scene_branch[compact]
    selected_goal = goals.gather(
        1, agent_branch[:, None, None].expand(-1, 1, 2))

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    edge_count = edge_index.shape[1]
    if relation_embedding.shape != (edge_count, num_worlds, 16):
        raise ValueError("relation_embedding must have shape [E,P,16]")
    if edge_count:
        src, dst = edge_index.long()
        if not torch.equal(compact[src], compact[dst]):
            raise ValueError("JDV2 edges may not cross scene boundaries")
        edge_branch = scene_branch[compact[src]]
        selected_relation = relation_embedding.gather(
            1, edge_branch[:, None, None].expand(-1, 1, 16)).squeeze(1)
    else:
        edge_branch = scene_branch.new_empty((0,))
        selected_relation = relation_embedding.new_empty((0, 16))
    return {
        'scene_ids': scene_ids,
        'compact_scene_index': compact,
        'scene_error': scene_error,
        'scene_branch': scene_branch,
        'agent_branch': agent_branch,
        'edge_branch': edge_branch,
        'selected_goal': selected_goal,
        'selected_relation_embedding': selected_relation,
    }


def jdv2_scene_timesteps(
        scene_index: torch.Tensor,
        active_timesteps: tuple[int, ...],
        *,
        supplied_t=None,
) -> torch.Tensor:
    """Choose one inference-active timestep per scene and broadcast to agents."""
    scene_ids, compact = canonicalize_scene_index(
        scene_index, scene_index.numel(), scene_index.device)
    active = torch.as_tensor(
        active_timesteps, dtype=torch.long, device=scene_index.device)
    if active.numel() == 0:
        raise ValueError("Stage B has no inference-active timesteps")
    if supplied_t is None:
        choice = torch.randint(
            active.numel(), (scene_ids.numel(),), device=scene_index.device)
        scene_timestep = active[choice]
    else:
        supplied = torch.as_tensor(
            supplied_t, dtype=torch.long, device=scene_index.device)
        if supplied.ndim == 0:
            scene_timestep = supplied.expand(scene_ids.numel())
        elif supplied.shape == (scene_ids.numel(),):
            scene_timestep = supplied
        elif supplied.shape == (scene_index.numel(),):
            scene_timestep = supplied.new_empty((scene_ids.numel(),))
            for scene in range(scene_ids.numel()):
                values = torch.unique(supplied[compact == scene])
                if values.numel() != 1:
                    raise ValueError(
                        "all agents in a scene must share one timestep")
                scene_timestep[scene] = values[0]
        else:
            raise ValueError("t must be scalar, [C], or [N]")
    if not torch.isin(scene_timestep, active).all():
        raise ValueError("Stage-B timestep is not inference-active")
    return scene_timestep[compact]


class GDTS(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.is_trainable = True
        self.args = args
        self.device = device
        # Cached batches already contain the downsampled visual tensors. Keep
        # only scene geometry in the model, avoiding ~10 GiB of redundant SDD
        # RGB/semantic arrays during training and evaluation.
        self.dataset = create_dataset(
            self.args.dataset, load_visual_data=False)

        ##################
        # MODEL PARAMETERS
        ##################
        # Goal Module
        self.num_image_channels = 6
        self.enc_chs = (self.num_image_channels + self.args.obs_length, 32, 32, 64, 64, 64)
        self.dec_chs = (64, 64, 64, 32, 32)
        self.goal_module = UNet(enc_chs=self.enc_chs, dec_chs=self.dec_chs, out_chs=self.args.pred_length).to(self.device)

        # Augmented History Encoder
        # Keep the legacy registrar/LSTM names exactly unchanged so an
        # independent GDTS checkpoint remains strict-load compatible.  The
        # original hard-coded CUDA device prevented CPU execution.
        self.registrar = ModelRegistrar(self.args.model_dir, self.device)
        self.encoder = Encoding(self.registrar, self.device, self.args.e_dim)

        # Diffusion Network
        self.var_sched = VarianceSchedule(num_steps=self.args.ddpm_step, beta_T=5e-2, mode='linear')
        self.diffnet = TransformerConcatLinear(context_dim=self.args.e_dim, tf_layer=2).to(self.device)

        # ddim
        self.betas = linear_beta_schedule(self.args.ddpm_step) #cosine_beta_schedule(self.args.ddpm_step)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        ###############################
        # STRUCTURED JOINT-GOAL MODULES
        ###############################
        # These modules are created only for a social ablation.  Therefore
        # goal_model_type=independent has the same parameter namespace as the
        # original GDTS model and can still load legacy checkpoints strictly.
        self.jdv2_requested = (
            self.args.goal_model_type == 'joint_dependency_v2')
        self.jdv2_active = bool(
            self.jdv2_requested and getattr(self.args, 'jdv2_active', False))
        self.strict_no_z = bool(
            self.jdv2_active and
            self.args.jdv2_latent_objective == 'strict_no_z')
        self.active_goal_model_type = self.args.goal_model_type
        if self.jdv2_requested and not self.jdv2_active:
            # Exact all-off gate: this branch is decided before constructing
            # any V2 module, so initialization consumes no additional RNG.
            self.active_goal_model_type = 'independent'
        if (self.args.goal_model_type != 'independent' and
                self.args.training_stage == 'baseline'):
            self.active_goal_model_type = 'independent'

        if (self.args.goal_model_type != 'independent' and
                not self.jdv2_requested):
            graph_type = {
                'full': 'full',
                'radius': 'radius-only',
                'radius_only': 'radius-only',
                'radius_ttc': 'radius+TTC',
            }[self.args.graph_type]
            self.interaction_graph = SparseInteractionGraph(
                graph_type=graph_type,
                radius=self.args.graph_radius,
                ttc_threshold=self.args.ttc_threshold,
                adaptive=self.args.adaptive_graph,
                gate_hidden_dim=self.args.adaptive_graph_hidden_dim,
                top_k=self.args.adaptive_graph_top_k,
                gate_floor=self.args.adaptive_graph_gate_floor,
            )
            message_layers = (self.args.social_attention_layers
                              if self.args.use_social_encoder else 0)
            self.social_encoder = SocialMotionEncoder(
                hidden_dim=self.args.social_feature_dim,
                output_dim=self.args.social_feature_dim,
                edge_dim=EDGE_FEATURE_DIM,
                num_message_layers=message_layers,
            )

            # Social-GDTS is the R=1 ablation: its marginal sees neighbours,
            # while agent candidates are still sampled independently. LR-Goal
            # and Full-Joint use the configured scene-level latent modes.
            if self.args.goal_model_type in {'social', 'lowrank', 'joint'}:
                num_modes = (1 if self.args.goal_model_type == 'social'
                             else self.args.num_social_modes)
                self.joint_goal_model = LowRankJointGoal(
                    agent_dim=self.args.social_feature_dim,
                    num_social_modes=num_modes,
                    hidden_dim=self.args.social_feature_dim,
                    prior_temperature=self.args.goal_candidate_temperature,
                )

            if self.args.goal_model_type in {'energy', 'joint'}:
                self.relation_inference = RelationInference(
                    agent_dim=self.args.social_feature_dim,
                    num_relation_modes=self.args.num_relation_modes,
                    edge_dim=EDGE_FEATURE_DIM,
                    hidden_dim=self.args.social_feature_dim,
                    hard=self.args.hard_relation,
                )
                self.goal_energy = LowRankGoalEnergy(
                    agent_dim=self.args.social_feature_dim,
                    edge_dim=EDGE_FEATURE_DIM,
                    num_relation_modes=self.args.num_relation_modes,
                    rank=self.args.goal_pair_rank,
                    hidden_dim=self.args.social_feature_dim,
                    use_group_relative_feature=(
                        self.args.use_group_relative_feature),
                )

            if self.args.continuous_refinement:
                relation_dim = (
                    self.args.num_relation_modes
                    if self.args.goal_model_type in {'energy', 'joint'} else 0)
                self.continuous_goal_refiner = ContinuousJointGoalRefiner(
                    agent_dim=self.args.social_feature_dim,
                    edge_dim=EDGE_FEATURE_DIM,
                    relation_dim=relation_dim,
                    hidden_dim=self.args.social_feature_dim,
                    num_steps=self.args.continuous_refinement_steps,
                    max_delta=self.args.continuous_refinement_max_delta,
                )

            if self.args.trajectory_alignment:
                relation_dim = (
                    self.args.num_relation_modes
                    if self.args.goal_model_type in {'energy', 'joint'} else 0)
                self.trajectory_aligner = RelationAwareTrajectoryAligner(
                    agent_dim=self.args.social_feature_dim,
                    edge_dim=EDGE_FEATURE_DIM,
                    relation_dim=relation_dim,
                    hidden_dim=self.args.trajectory_alignment_hidden_dim,
                    sinkhorn_iterations=(
                        self.args.trajectory_alignment_sinkhorn_iters),
                    alignment_iterations=self.args.trajectory_alignment_steps,
                    temperature=self.args.trajectory_alignment_temperature,
                    keep_threshold=(
                        self.args.trajectory_alignment_keep_threshold),
                    top_k_edges=self.args.trajectory_alignment_top_k_edges,
                )

            if self.args.trajectory_coupling == 'multiway_v4':
                self.multiway_coupler = RelationAwareMultiwayCoupler(
                    trajectory_dim=self.args.trajectory_dim,
                    edge_dim=EDGE_FEATURE_DIM,
                    num_relation_modes=self.args.num_relation_modes,
                    trajectory_pair_rank=self.args.trajectory_pair_rank,
                    trajectory_conditioned_relation=(
                        self.args.trajectory_conditioned_relation),
                    pair_specific_gate=self.args.pair_specific_gate,
                    trajectory_energy_type=self.args.trajectory_energy_type,
                    sync_iterations=self.args.sync_iterations,
                    sinkhorn_iterations=self.args.sinkhorn_iterations,
                    sync_temperature=self.args.sync_temperature,
                    hard_projection=self.args.hard_projection,
                    use_identity_bypass=self.args.use_identity_bypass,
                    keep_threshold=self.args.keep_threshold,
                    trajectory_dt=self.args.trajectory_dt,
                    inference_edge_topk=(
                        self.args.sync_inference_top_k_edges),
                )

            self.joint_sampler = JointGoalSampler(
                num_samples=self.args.num_joint_samples,
                num_refinement_steps=self.args.num_refinement_steps,
                temperature=self.args.joint_sampling_temperature,
                energy_weight=self.args.energy_weight,
                sampling_mode=self.args.joint_sampling_mode,
                sampling_strategy=self.args.joint_sampling_strategy,
                normalize_pair_energy=(
                    self.args.pair_energy_normalization == 'mean'),
            )
            active_mode_count = (
                self.args.num_social_modes
                if self.args.goal_model_type in {'lowrank', 'joint'} else 1)
            # Each cache item is one variable-N scene window, so a conventional
            # within-minibatch utilization KL would force every single scene's
            # p(z|X) toward uniform.  Short detached histories instead estimate
            # utilization across recent scene windows while leaving individual
            # distributions free to be sharp.
            history_size = 64
            self.register_buffer(
                '_mode_usage_history',
                torch.zeros(history_size, active_mode_count),
                persistent=False)
            self.register_buffer(
                '_mode_usage_history_count', torch.zeros((), dtype=torch.long),
                persistent=False)
            self.register_buffer(
                '_mode_usage_history_pointer', torch.zeros((), dtype=torch.long),
                persistent=False)
            self.register_buffer(
                '_relation_usage_history',
                torch.zeros(history_size, self.args.num_relation_modes),
                persistent=False)
            self.register_buffer(
                '_relation_usage_history_count',
                torch.zeros((), dtype=torch.long), persistent=False)
            self.register_buffer(
                '_relation_usage_history_pointer',
                torch.zeros((), dtype=torch.long), persistent=False)
            self.last_joint_diagnostics = {}

        if self.jdv2_active:
            graph_type = {
                'full': 'full',
                'radius': 'radius-only',
                'radius_only': 'radius-only',
                'radius_ttc': 'radius+TTC',
            }[self.args.graph_type]
            # V2 caches and runs use the parameter-free sparse proposal graph.
            self.interaction_graph = SparseInteractionGraph(
                graph_type=graph_type,
                radius=self.args.graph_radius,
                ttc_threshold=self.args.ttc_threshold,
                dt=self.args.trajectory_dt,
                adaptive=False,
            )
            self.social_encoder = SocialMotionEncoder(
                hidden_dim=128, output_dim=128, edge_dim=EDGE_FEATURE_DIM,
                num_message_layers=self.args.social_attention_layers)
            self.relation_inference = RelationInference(
                agent_dim=128, num_relation_modes=4,
                edge_dim=EDGE_FEATURE_DIM, hidden_dim=128, hard=False)
            self.jdv2_scene_prior = (
                None if self.strict_no_z else SceneLatentPrior())
            self.jdv2_future_teacher = SceneFutureTeacher(
                use_scene_latent=not self.strict_no_z)
            self.jdv2_unary = UnaryGoalResidual(
                prior_temperature=self.args.goal_candidate_temperature,
                use_scene_latent=not self.strict_no_z)
            self.jdv2_dynamic_relation = DynamicHypothesisRelation(
                graph_radius=self.args.graph_radius,
                use_scene_latent=not self.strict_no_z)
            self.jdv2_joint_energy = RelationSpecificJointEnergy(
                use_scene_latent=not self.strict_no_z)
            self.jdv2_sampler = ParallelConditionalSampler(
                num_samples=20, num_refinement_steps=2,
                temperature=self.args.joint_sampling_temperature,
                minimum_active_mode=self.args.jdv2_minimum_active_mode,
                strict_no_z=self.strict_no_z,
                refinement_policy=self.args.jdv2_refinement_policy)
            self.jdv2_corrector = DependencyCorrector(
                dt=self.args.trajectory_dt)
            self.last_joint_diagnostics = {}

        self._configure_training_stage(epoch=1)


        # saved_model_name = os.path.join(
        #             self.args.model_dir, 'saved_models',
        #             'goal_pretrain_best_model.pt')
        # if os.path.isfile(saved_model_name):
        #     print('Loading pretrained goal module...')
        #     checkpoint = torch.load(saved_model_name,
        #                             map_location=self.device)
        #     model_state_dict = checkpoint['model_state_dict']
        #     new_model_state_dict = {}
        #     for k in list(model_state_dict.keys()):
        #         new_k = k.replace('goal_module.', '')
        #         new_model_state_dict[new_k] = model_state_dict[k]
        #     self.goal_module.load_state_dict(new_model_state_dict)

    def _baseline_modules(self):
        """Return the original GDTS modules as one optimizer/freeze family."""
        return (self.goal_module, self.registrar, self.diffnet)

    def _jdv2_goal_module_names(self):
        names = [
            'social_encoder', 'relation_inference', 'jdv2_future_teacher',
            'jdv2_unary', 'jdv2_dynamic_relation', 'jdv2_joint_energy']
        if not self.strict_no_z:
            names.insert(2, 'jdv2_scene_prior')
        return tuple(names)

    def _configure_training_stage(self, epoch=1):
        """Apply the staged-training freeze policy without changing modules.

        ``joint`` freezes the original U-Net, history encoder and diffusion
        denoiser. ``finetune`` trains every component. ``baseline`` leaves the
        GDTS components trainable and freezes the newly added modules; the
        exact historical baseline is still obtained with
        ``goal_model_type=independent``.
        """
        stage = self.args.training_stage
        if self.jdv2_active:
            if stage not in {
                    'joint_goal', 'joint_trajectory', 'joint_finetune'}:
                raise ValueError(f'Unsupported JDV2 training_stage={stage!r}')
            for module in self._baseline_modules():
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            goal_names = self._jdv2_goal_module_names()
            goal_trainable = stage in {'joint_goal', 'joint_finetune'}
            for name in goal_names:
                for parameter in getattr(self, name).parameters():
                    parameter.requires_grad_(goal_trainable)
            corrector_trainable = stage in {
                'joint_trajectory', 'joint_finetune'} and \
                self.args.use_dependency_corrector
            for parameter in self.jdv2_corrector.parameters():
                parameter.requires_grad_(corrector_trainable)
            return
        legacy_structured_names = (
            'interaction_graph', 'social_encoder', 'joint_goal_model',
            'relation_inference', 'goal_energy', 'continuous_goal_refiner')
        if stage not in {
                'joint', 'baseline', 'finetune', 'alignment',
                'multiway_coupling'}:
            raise ValueError(f'Unsupported training_stage={stage!r}')

        baseline_trainable = (
            stage == 'baseline' or
            (stage == 'finetune' and
             int(epoch) >= self.args.baseline_unfreeze_epoch) or
            (stage == 'multiway_coupling' and
             not self.args.freeze_upstream_generator))
        structured_trainable = stage in {'joint', 'finetune'} or (
            stage == 'multiway_coupling' and
            not self.args.freeze_upstream_generator and
            self.args.upstream_generator == 'stage1_joint')
        for module in self._baseline_modules():
            for parameter in module.parameters():
                parameter.requires_grad_(baseline_trainable)
        for name in legacy_structured_names:
            module = getattr(self, name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(structured_trainable)
        trajectory_aligner = getattr(self, 'trajectory_aligner', None)
        if trajectory_aligner is not None:
            aligner_trainable = stage in {'joint', 'finetune', 'alignment'}
            for parameter in trajectory_aligner.parameters():
                parameter.requires_grad_(aligner_trainable)
        multiway_coupler = getattr(self, 'multiway_coupler', None)
        if multiway_coupler is not None:
            for parameter in multiway_coupler.parameters():
                parameter.requires_grad_(stage == 'multiway_coupling')

    def train(self, mode=True):
        """Keep a frozen generator deterministic in alignment-only training."""
        super().train(mode)
        if mode and self.jdv2_active:
            # Frozen modules must be both gradient-frozen and state-frozen.
            for module in self._baseline_modules():
                module.eval()
            if self.args.training_stage == 'joint_trajectory':
                for name in self._jdv2_goal_module_names():
                    getattr(self, name).eval()
            self.jdv2_corrector.train(
                self.args.training_stage in {
                    'joint_trajectory', 'joint_finetune'} and
                self.args.use_dependency_corrector)
            return self
        if mode and getattr(self.args, 'training_stage', None) == 'alignment':
            frozen_modules = self._baseline_modules() + tuple(
                getattr(self, name) for name in (
                    'interaction_graph', 'social_encoder', 'joint_goal_model',
                    'relation_inference', 'goal_energy',
                    'continuous_goal_refiner')
                if getattr(self, name, None) is not None)
            for module in frozen_modules:
                module.eval()
            self.trajectory_aligner.train(True)
        if (mode and
                getattr(self.args, 'training_stage', None) ==
                'multiway_coupling'):
            if self.args.freeze_upstream_generator:
                for module in self._baseline_modules():
                    module.eval()
                for name in (
                        'interaction_graph', 'social_encoder',
                        'joint_goal_model', 'relation_inference',
                        'goal_energy', 'continuous_goal_refiner',
                        'trajectory_aligner'):
                    module = getattr(self, name, None)
                    if module is not None:
                        module.eval()
            self.multiway_coupler.train(True)
        return self

    def configure_training_epoch(self, epoch):
        """Apply run-local progressive unfreezing before a training epoch."""
        self._configure_training_stage(epoch=epoch)

    def optimizer_parameter_families(self):
        """Return disjoint baseline/structured parameter lists.

        Finetuning includes baseline tensors in the optimizer even while they
        are temporarily frozen, so delayed unfreezing does not rebuild the
        optimizer or reset its scheduler state.
        """
        if self.jdv2_active:
            baseline_parameters = [
                parameter for module in self._baseline_modules()
                for parameter in module.parameters()]
            goal_names = self._jdv2_goal_module_names()
            goal_parameters = [
                parameter for name in goal_names
                for parameter in getattr(self, name).parameters()
                if parameter.requires_grad]
            corrector_parameters = [
                parameter for parameter in self.jdv2_corrector.parameters()
                if parameter.requires_grad]
            all_ids = [id(parameter) for parameter in (
                baseline_parameters + goal_parameters + corrector_parameters)]
            if len(all_ids) != len(set(all_ids)):
                raise RuntimeError('JDV2 optimizer parameter families overlap')
            if any(parameter.requires_grad for parameter in baseline_parameters):
                raise RuntimeError('Base GDTS must remain frozen in JDV2')
            return {
                'gdts_frozen_backbone': [],
                'jdv2_goal_dependency': goal_parameters,
                'jdv2_corrector': corrector_parameters,
            }
        baseline_ids = {
            id(parameter)
            for module in self._baseline_modules()
            for parameter in module.parameters()
        }
        families = {'baseline': [], 'structured': []}
        for parameter in self.parameters():
            family = 'baseline' if id(parameter) in baseline_ids \
                else 'structured'
            if self.args.training_stage == 'finetune' or \
                    parameter.requires_grad:
                families[family].append(parameter)
        return families

    def _map_goals_to_world(self, goal_points, scene):
        """Convert downsampled heatmap coordinates ``[...,2]`` to metres."""
        return scene.make_world_coord_torch(
            goal_points * float(self.args.down_factor))

    def _map_goals_to_map(self, goal_points, scene):
        """Convert world-metre endpoints to downsampled heatmap coordinates."""
        return scene.make_pixel_coord_torch(goal_points) / float(
            self.args.down_factor)

    @staticmethod
    def _categorical_entropy(probability):
        """Mean categorical entropy with an empty-tensor safe value."""
        if probability.numel() == 0:
            return probability.new_zeros(())
        probability = probability.clamp_min(1e-8)
        return -(probability * probability.log()).sum(dim=-1).mean()

    def _usage_balance_loss(self, probability, family):
        """KL over recent-window utilization, not a per-scene uniform target."""
        if probability.numel() == 0:
            return probability.new_zeros(())
        if family not in {'mode', 'relation'}:
            raise ValueError("family must be 'mode' or 'relation'")
        history = getattr(self, f'_{family}_usage_history')
        count_buffer = getattr(self, f'_{family}_usage_history_count')
        pointer_buffer = getattr(self, f'_{family}_usage_history_pointer')
        if probability.ndim != 2 or probability.shape[1] != history.shape[1]:
            raise ValueError(f'{family} probability/history shape mismatch')

        current_usage = probability.mean(dim=0)
        history_count = int(count_buffer.item())
        if history_count == 0:
            # With no other scene observed yet there is no batch-utilization
            # evidence. Return a differentiable zero instead of regularizing
            # this first scene toward a uniform posterior.
            loss = current_usage.sum() * 0.0
        else:
            previous_usage = history[:history_count].sum(
                dim=0).detach().clone()
            historical_usage = (
                previous_usage + current_usage.detach()) / (
                    history_count + 1)
            historical_usage = historical_usage.clamp_min(1e-8)
            historical_usage = historical_usage / historical_usage.sum()
            # Keep the forward value history-smoothed, but do not dilute the
            # useful gradient by 1/(history_count+1). The current window is the
            # only trainable term and should be pushed toward under-used modes,
            # rather than receiving an effectively zero signal after 64 steps.
            usage = (
                historical_usage + current_usage - current_usage.detach())
            uniform_log_prob = -np.log(float(usage.numel()))
            loss = (usage * (
                usage.clamp_min(1e-8).log() - uniform_log_prob)).sum()

        if self.training:
            with torch.no_grad():
                pointer = int(pointer_buffer.item())
                history[pointer].copy_(current_usage.detach())
                pointer_buffer.fill_((pointer + 1) % history.shape[0])
                count_buffer.fill_(min(history_count + 1, history.shape[0]))
        return loss

    def _record_joint_diagnostics(self, structured):
        """Store compact finite scalars for the trainer progress display."""
        mode_prob = structured['mode_prob'].detach()
        relation_prob = structured.get('relation_prob')
        candidate_prob = structured['candidate_prob'].detach()
        edge_count = int(structured['edge_index'].shape[1])
        agent_count = int(structured['goal_candidates_world'].shape[0])
        diagnostics = {
            'num_agents': float(agent_count),
            'num_edges': float(edge_count),
            'avg_degree': (2.0 * edge_count / max(agent_count, 1)),
            'mode_entropy': float(
                self._categorical_entropy(mode_prob).cpu()),
            'goal_candidate_entropy': float(
                self._categorical_entropy(candidate_prob).cpu()),
        }
        graph_diagnostics = getattr(
            self.interaction_graph, 'last_diagnostics', {})
        diagnostics.update({
            f'graph_{key}': float(value)
            for key, value in graph_diagnostics.items()
        })
        mode_usage = mode_prob.mean(dim=0) if mode_prob.numel() else \
            mode_prob.new_zeros((0,))
        for index, value in enumerate(mode_usage):
            diagnostics[f'mode_usage_{index}'] = float(value.cpu())

        if relation_prob is None or relation_prob.numel() == 0:
            diagnostics['relation_entropy'] = 0.0
            relation_modes = int(getattr(
                self.args, 'num_relation_modes', 0))
            relation_usage = candidate_prob.new_zeros((relation_modes,))
        else:
            detached_relation = relation_prob.detach()
            diagnostics['relation_entropy'] = float(
                self._categorical_entropy(detached_relation).cpu())
            relation_usage = detached_relation.mean(dim=0)
        for index, value in enumerate(relation_usage):
            diagnostics[f'relation_usage_{index}'] = float(value.cpu())

        energy_output = structured.get('energy_output')
        if energy_output is not None and 'effective_energy' in energy_output:
            effective_energy = energy_output['effective_energy'].detach()
            diagnostics['average_pairwise_energy'] = (
                float(effective_energy.mean().cpu())
                if effective_energy.numel() else 0.0)
        else:
            diagnostics['average_pairwise_energy'] = 0.0

        sampled_candidate_index = structured.get('joint_candidate_index')
        if sampled_candidate_index is not None:
            branch_index = sampled_candidate_index[:, :self.args.num_samples]
            if branch_index.shape[1] > 0:
                sorted_index = branch_index.sort(dim=1).values
                unique_count = 1 + (
                    sorted_index[:, 1:] != sorted_index[:, :-1]).sum(dim=1)
                diagnostics['branch_unique_candidates'] = float(
                    unique_count.float().mean().cpu())
                diagnostics['branch_unique_fraction'] = float(
                    (unique_count.float() / branch_index.shape[1]).mean().cpu())

        refined_goal = structured.get('joint_goal_points_world')
        unrefined_goal = structured.get('joint_goal_points_world_unrefined')
        if refined_goal is not None and unrefined_goal is not None:
            refinement_delta = torch.linalg.vector_norm(
                refined_goal.detach() - unrefined_goal.detach(), dim=-1)
            diagnostics['continuous_refinement_mean_delta'] = float(
                refinement_delta.mean().cpu())
            diagnostics['continuous_refinement_max_delta'] = float(
                refinement_delta.max().cpu())

        complexity = estimate_joint_goal_complexity(
            num_agents=max(agent_count, 1),
            num_candidates=self.args.num_goal_candidates,
            num_social_modes=int(mode_prob.shape[-1]),
            num_relation_modes=(
                self.args.num_relation_modes
                if self.args.goal_model_type in {'energy', 'joint'} else 1),
            pair_rank=(
                self.args.goal_pair_rank
                if self.args.goal_model_type in {'energy', 'joint'} else 1),
            num_edges=edge_count,
            num_refinement_steps=self.args.num_refinement_steps,
        )
        diagnostics['naive_log10_states'] = complexity['naive_log10_states']
        diagnostics['structured_computation_estimate'] = float(
            complexity['structured_computation_estimate'])
        if not all(np.isfinite(value) for value in diagnostics.values()):
            raise FloatingPointError(
                f'Non-finite joint-goal diagnostics: {diagnostics}')
        self.last_joint_diagnostics = diagnostics

    def _jdv2_goal_outputs(self, inputs, goal_prob_map, *, sample=False,
                           include_teacher=False):
        """Build the frozen JDV2 information chain without dense agent pairs.

        Training pseudocode::

            h = SocialMotionEncoder(history, sparse_edges)
            p_z = scene_prior(h)
            q_z, q_r = future_teachers(history, future)  # training only
            unary = log frozen candidate prior + zero-init residual
            p_r = dynamic_relation(base_relation, z, candidate geometry)
            goals = Parallel Conditional Refinement(p_z, unary, p_r, energy)
        """
        cache = inputs.get('jdv2_cache')
        if cache is None and not (
                self.args.phase == 'build-jdv2-cache' or
                self.args.fast_debug):
            raise RuntimeError(
                'Active JDV2 requires its explicit frozen cache; run '
                '--phase build-jdv2-cache')
        if cache is None:
            goal_candidates_map, candidate_prob = generate_goal_candidates(
                goal_prob_map, num_candidates=21, device=self.device,
                use_ttst=self.args.use_ttst)
            candidate_log_prior = candidate_prob.float().clamp_min(1e-8).log()
            goal_candidates_world = self._map_goals_to_world(
                goal_candidates_map, inputs['scene'])
        else:
            goal_candidates_map = cache['goal_candidates_map']
            goal_candidates_world = cache['goal_candidates_world']
            candidate_log_prior = cache['candidate_log_prior'].float()
            candidate_prob = F.softmax(candidate_log_prior, dim=-1)
        candidate_mask = torch.ones_like(candidate_prob, dtype=torch.bool)
        obs_world = inputs['obs_traj_world']
        scene_index = inputs['scene_index']
        if cache is None:
            edge_index, edge_feat, edge_weight = self.interaction_graph(
                obs_world, scene_index=scene_index)
        else:
            edge_index = cache['edge_index'].long()
            edge_feat = cache['edge_feat']
            edge_weight = cache['edge_weight']
        # Sparse message aggregation uses index_add_.  Keep both its
        # accumulator and weighted messages in FP32 under AMP; otherwise
        # autocast can create a BF16/FP16 agent accumulator while the FP32
        # graph weights promote messages back to FP32.
        with torch.autocast(
                device_type=obs_world.device.type, enabled=False):
            agent_feat = self.social_encoder(
                obs_world.float(), edge_index, edge_feat.float(),
                edge_weight.float(), scene_index=scene_index)
        # Relation probabilities are a mandatory FP32 island under AMP.
        with torch.autocast(
                device_type=agent_feat.device.type, enabled=False):
            base_relation_prob, base_relation_logits = self.relation_inference(
                agent_feat.float(), edge_index, edge_feat.float(), hard=False,
                return_logits=True)

        if self.strict_no_z:
            scene_prior = None
        elif self.args.use_scene_latent:
            scene_prior = self.jdv2_scene_prior(agent_feat, scene_index)
        else:
            scene_ids, compact = canonicalize_scene_index(
                scene_index, agent_feat.shape[0], agent_feat.device)
            history_scene = agent_feat.new_zeros((scene_ids.numel(), 128))
            scene_prior = {
                'scene_ids': scene_ids,
                'compact_scene_index': compact,
                'history_scene': history_scene,
                'logits': history_scene.new_zeros((scene_ids.numel(), 1)),
                'log_prob': history_scene.new_zeros((scene_ids.numel(), 1)),
                'prob': history_scene.new_ones((scene_ids.numel(), 1)),
                'attention': agent_feat.new_zeros((agent_feat.shape[0],)),
            }
        unary = self.jdv2_unary(
            agent_feat, goal_candidates_world, obs_world[:, -1],
            candidate_log_prior, candidate_mask=candidate_mask)

        scene_posterior = None
        relation_teacher = None
        relation_descriptor = None
        if include_teacher:
            future_position = inputs['world_coord'][
                self.args.obs_length:].permute(1, 0, 2).contiguous()
            position_with_anchor = torch.cat((
                obs_world[:, -1:].float(), future_position.float()), dim=1)
            future_velocity = torch.diff(
                position_with_anchor, dim=1) / float(self.args.trajectory_dt)
            if self.strict_no_z:
                scene_posterior = None
            elif self.args.use_scene_latent:
                scene_posterior = self.jdv2_future_teacher.scene_posterior(
                    future_position, future_velocity, obs_world[:, -1],
                    scene_prior['logits'], scene_index)
            else:
                scene_posterior = {
                    'logits': scene_prior['logits'],
                    'log_prob': scene_prior['log_prob'],
                    'prob': scene_prior['prob'],
                }
            teacher_cache = inputs.get('jdv2_teacher_cache')
            relation_descriptor = (
                teacher_cache['future_pair_descriptor']
                if teacher_cache is not None else
                future_pair_descriptor(
                    future_position, obs_world[:, -1], edge_index))
            relation_teacher = self.jdv2_future_teacher.relation_posterior(
                relation_descriptor)

        sampled = None
        if sample:
            sampled = self.jdv2_sampler(
                unary['score'], goal_candidates_world,
                None if self.strict_no_z else scene_prior['prob'],
                scene_index, edge_index, edge_feat, agent_feat,
                obs_world[:, -1], base_relation_logits,
                self.jdv2_dynamic_relation, self.jdv2_joint_energy,
                candidate_mask=candidate_mask,
                sampling_mode=self.args.joint_sampling_mode,
                use_scene_latent=self.args.use_scene_latent,
                use_dynamic_relation=self.args.use_dynamic_relation,
                use_joint_energy=self.args.use_joint_energy)

        output = {
            'goal_candidates_map': goal_candidates_map,
            'goal_candidates_world': goal_candidates_world,
            'candidate_prob': candidate_prob,
            'candidate_log_prior': candidate_log_prior,
            'candidate_mask': candidate_mask,
            'agent_feat': agent_feat,
            'edge_index': edge_index,
            'edge_feat': edge_feat,
            'edge_weight': edge_weight,
            'base_relation_prob': base_relation_prob,
            'base_relation_logits': base_relation_logits,
            'relation_teacher': relation_teacher,
            'relation_descriptor': relation_descriptor,
            'unary_score': unary['score'],
            'sampled': sampled,
        }
        if not self.strict_no_z:
            output.update({
                'scene_prior': scene_prior,
                'scene_posterior': scene_posterior,
            })
        floating = {
            name: value for name, value in output.items()
            if torch.is_tensor(value) and value.is_floating_point()}
        if any(not torch.isfinite(value).all()
               for value in floating.values()):
            bad = [name for name, value in floating.items()
                   if not torch.isfinite(value).all()]
            raise FloatingPointError(f'Non-finite JDV2 tensors: {bad}')
        self.last_joint_diagnostics = {
            'num_agents': float(agent_feat.shape[0]),
            'num_edges': float(edge_index.shape[1]),
            'avg_degree': (2.0 * edge_index.shape[1] /
                           max(agent_feat.shape[0], 1)),
            'relation_entropy': float((
                -(base_relation_prob.float().clamp_min(1e-8) *
                  base_relation_prob.float().clamp_min(1e-8).log()).sum(-1).mean()
                if base_relation_prob.numel() else
                base_relation_prob.new_zeros(())).detach().cpu()),
        }
        if not self.strict_no_z:
            self.last_joint_diagnostics['scene_latent_entropy'] = float((
                -(scene_prior['prob'].float() *
                  scene_prior['log_prob'].float()).sum(-1).mean()
            ).detach().cpu())
            prior_mean = scene_prior['prob'].detach().float().mean(dim=0)
            for index, value in enumerate(prior_mean):
                self.last_joint_diagnostics[f'p_z_mean_{index}'] = float(
                    value.cpu())
                self.last_joint_diagnostics[
                    f'scene_mode_usage_{index}'] = float(value.cpu())
            if scene_posterior is not None:
                posterior_mean = scene_posterior['prob'].detach().float().mean(
                    dim=0)
                self.last_joint_diagnostics[
                    'scene_posterior_entropy'] = float((
                        -(scene_posterior['prob'].float() *
                          scene_posterior['log_prob'].float()).sum(-1).mean()
                    ).detach().cpu())
                self.last_joint_diagnostics['q_z_prior_l1'] = float(
                    (scene_posterior['prob'].detach().float() -
                     scene_prior['prob'].detach().float()).abs().mean().cpu())
                for index, value in enumerate(posterior_mean):
                    self.last_joint_diagnostics[f'q_z_mean_{index}'] = float(
                        value.cpu())
        base_relation_mean = (
            base_relation_prob.detach().float().mean(dim=0)
            if base_relation_prob.numel() else
            base_relation_prob.new_zeros((self.args.jdv2_relation_modes,))
        )
        for index, value in enumerate(base_relation_mean):
            self.last_joint_diagnostics[
                f'base_relation_usage_{index}'] = float(value.cpu())
        if relation_teacher is not None:
            teacher_prob = relation_teacher['prob'].detach().float()
            teacher_mean = (
                teacher_prob.mean(dim=0) if teacher_prob.numel() else
                teacher_prob.new_zeros((self.args.jdv2_relation_modes,)))
            self.last_joint_diagnostics['teacher_relation_entropy'] = float((
                -(teacher_prob.clamp_min(1e-8) *
                  teacher_prob.clamp_min(1e-8).log()).sum(-1).mean()
                if teacher_prob.numel() else teacher_prob.new_zeros(())
            ).cpu())
            for index, value in enumerate(teacher_mean):
                self.last_joint_diagnostics[
                    f'teacher_relation_usage_{index}'] = float(value.cpu())
        if sampled is not None:
            if not self.strict_no_z:
                sampled_mode = sampled['scene_mode'].detach().long()
                sampled_mode_usage = F.one_hot(
                    sampled_mode, num_classes=self.args.jdv2_scene_modes
                ).float().mean(dim=(0, 1))
                for index, value in enumerate(sampled_mode_usage):
                    self.last_joint_diagnostics[
                        f'sampled_scene_mode_usage_{index}'] = float(
                            value.cpu())
            sampled_relation = sampled['relation_prob'].detach().float()
            sampled_relation_mean = (
                sampled_relation.mean(dim=(0, 1))
                if sampled_relation.numel() else
                sampled_relation.new_zeros((self.args.jdv2_relation_modes,)))
            for index, value in enumerate(sampled_relation_mean):
                self.last_joint_diagnostics[
                    f'sampled_relation_usage_{index}'] = float(value.cpu())
            with torch.no_grad():
                initial = sampled['initial_candidate_index'].long()
                selected = sampled['candidate_index'].long()
                order = torch.argsort(
                    candidate_log_prior.float(), dim=-1,
                    descending=True, stable=True)
                rank = torch.empty_like(order)
                positions = torch.arange(
                    1, order.shape[1] + 1, device=order.device
                )[None].expand_as(order)
                rank.scatter_(1, order, positions)
                initial_rank = rank.gather(1, initial)
                initial_unique = initial.new_tensor([
                    torch.unique(row).numel() for row in initial],
                    dtype=torch.float32)
                gt_goal = inputs['world_coord'][-1].float()
                candidate_error = torch.linalg.vector_norm(
                    goal_candidates_world.float() - gt_goal[:, None], dim=-1)
                initial_error = candidate_error.gather(1, initial).min(-1).values
                selected_error = candidate_error.gather(
                    1, selected).min(-1).values
                suffix = 'e_gt0' if edge_index.shape[1] else 'e_eq0'
                sampler_diagnostics = {
                    'sampler_initial_unique_candidates': initial_unique.mean(),
                    'sampler_rank1_coverage':
                        (initial_rank <= 1).any(-1).float().mean(),
                    'sampler_top3_coverage':
                        (initial_rank <= 3).any(-1).float().mean(),
                    'sampler_top5_coverage':
                        (initial_rank <= 5).any(-1).float().mean(),
                    'sampler_initial_goal_oracle': initial_error.mean(),
                    'sampler_selected_goal_oracle': selected_error.mean(),
                }
                for name, value in sampler_diagnostics.items():
                    scalar = float(value.cpu())
                    self.last_joint_diagnostics[name] = scalar
                    self.last_joint_diagnostics[f'{name}_{suffix}'] = scalar
        return output

    def _jdv2_contexts(self, inputs, goal_points_map):
        """Run the unchanged GDTS goal-relative history encoder."""
        x = inputs['x_augmented'].detach()
        num_agents, num_worlds = goal_points_map.shape[:2]
        contexts = []
        for sample_index in range(num_worlds):
            goal = goal_points_map[:, sample_index].detach()
            current_agents = torch.zeros(
                (num_agents, self.args.obs_length, 8), device=self.device,
                dtype=x.dtype)
            current_agents[:, :, 2:] = x[
                :self.args.obs_length, :, :6].permute(1, 0, 2)
            current_agents[:, :, :2] = x[
                :self.args.obs_length, :, 6:8].permute(1, 0, 2) - \
                goal[:, None]
            contexts.append(self.encoder.encode_hist(
                node_hist=current_agents,
                dropout_keep_prob=1).unsqueeze(1))
        return torch.stack(contexts)

    def _jdv2_encode(self, inputs, goal_logit_map, goal_prob_map, if_test,
                     for_loss):
        structured = self._jdv2_goal_outputs(
            inputs, goal_prob_map, sample=if_test,
            include_teacher=for_loss)
        if if_test:
            sampled = structured['sampled']
            branch_world = sampled['goals']
            branch_map = self._map_goals_to_map(branch_world, inputs['scene'])
            # Keep the heatmap argmax as the internal trunk endpoint.  It is
            # never exposed as a 21st evaluation sample.
            trunk_index = structured['candidate_prob'].argmax(
                dim=-1, keepdim=True)
            trunk_world = structured['goal_candidates_world'].gather(
                1, trunk_index[:, :, None].expand(-1, -1, 2))
            trunk_map = self._map_goals_to_map(trunk_world, inputs['scene'])
            all_goal_map = torch.cat((branch_map, trunk_map), dim=1)
            all_goal_world = torch.cat((branch_world, trunk_world), dim=1)
            structured.update({
                'joint_candidate_index': torch.cat((
                    sampled['candidate_index'], trunk_index), dim=1),
                'joint_goal_points_world': all_goal_world,
                'joint_goal_points_map': all_goal_map,
                'relation_prob': sampled['relation_prob'],
                'dependency_state': {
                    'edge_index': structured['edge_index'],
                    'edge_weight': structured['edge_weight'],
                    'relation_embedding': sampled['relation_embedding'],
                    'last_position_world': inputs['obs_traj_world'][:, -1],
                    'last_position_map': inputs['x_augmented'][
                        self.args.obs_length - 1, :, 6:8],
                    'scene': inputs['scene'],
                },
            })
            if not self.strict_no_z:
                structured['sampled_scene_mode'] = sampled['scene_mode']
            goal_points_map = all_goal_map
        else:
            goal_points_map = inputs['x_augmented'][
                -1, :, 6:8].unsqueeze(1)
        contexts = self._jdv2_contexts(inputs, goal_points_map)
        structured['goal_logit_map'] = goal_logit_map.unsqueeze(0).expand(
            goal_points_map.shape[1], -1, -1, -1, -1)
        structured['goal_point'] = goal_points_map.permute(1, 0, 2)
        return contexts, structured

    def _structured_goal_outputs(self, inputs, goal_prob_map,
                                 if_test=False, for_loss=False):
        """Build and optionally sample the tractable scene-level goal model.

        Shapes follow ``N`` agents, ``K`` candidates, ``R`` global modes,
        ``M`` relation modes and ``E`` canonical sparse edges. No ``K**N`` or
        dense ``[N,N,K,K]`` object is constructed.
        """
        if self.active_goal_model_type == 'independent':
            raise RuntimeError('Structured outputs requested in baseline mode.')

        # goal_candidates_map: [N,K,2], candidate_prob: [N,K]
        goal_candidates_map, candidate_heatmap_prob = generate_goal_candidates(
            goal_prob_map,
            num_candidates=self.args.num_goal_candidates,
            device=self.device,
            use_ttst=self.args.use_ttst,
        )
        if self.args.goal_candidate_prior == 'uniform':
            # TTST already represents dense heatmap regions with more nearby
            # cluster centers. Reweighting those representatives by pointwise
            # heatmap height counts that density twice and collapses sampling.
            candidate_prob = torch.full_like(
                candidate_heatmap_prob,
                1.0 / candidate_heatmap_prob.shape[-1])
        else:
            candidate_prob = candidate_heatmap_prob
        candidate_mask = torch.ones_like(candidate_prob, dtype=torch.bool)
        candidate_log_prior = candidate_prob.clamp_min(1e-8).log()
        goal_candidates_world = self._map_goals_to_world(
            goal_candidates_map, inputs['scene'])

        obs_world = inputs['obs_traj_world']                 # [N,T_obs,2]
        scene_index = inputs['scene_index']                  # [N]
        edge_index, edge_feat, edge_weight = self.interaction_graph(
            obs_world, scene_index=scene_index)              # [2,E],[E,D],[E]
        graph_gate_prob = self.interaction_graph.last_gate_prob
        agent_feat = self.social_encoder(
            obs_world, edge_index, edge_feat, edge_weight,
            scene_index=scene_index)                         # [N,D]

        model_type = self.args.goal_model_type
        if model_type in {'social', 'lowrank', 'joint'}:
            distribution = self.joint_goal_model(
                agent_feat=agent_feat,
                goal_candidates=goal_candidates_world,
                scene_index=scene_index,
                last_pos=obs_world[:, -1],
                candidate_mask=candidate_mask,
                candidate_log_prior=candidate_log_prior,
            )
        else:
            # Sparse-Energy deliberately removes the global latent mode and
            # retains the original heatmap marginal as its unary potential.
            scene_ids, compact_scene_index = canonicalize_scene_index(
                scene_index, goal_candidates_world.shape[0],
                goal_candidates_world.device)
            mode_log_prob = goal_candidates_world.new_zeros(
                (scene_ids.numel(), 1))
            conditional_log_prob = candidate_log_prior[:, None, :]
            distribution = {
                'scene_ids': scene_ids,
                'compact_scene_index': compact_scene_index,
                'candidate_mask': candidate_mask,
                'mode_logits': mode_log_prob,
                'mode_log_prob': mode_log_prob,
                'mode_prob': mode_log_prob.exp(),
                'conditional_goal_logits': conditional_log_prob,
                'conditional_goal_log_prob': conditional_log_prob,
                'conditional_goal_prob': conditional_log_prob.exp(),
                'candidate_log_prior': candidate_log_prior,
            }

        relation_prob = None
        relation_logits = None
        energy_output = None
        if model_type in {'energy', 'joint'}:
            relation_prob, relation_logits = self.relation_inference(
                agent_feat, edge_index, edge_feat,
                hard=self.args.hard_relation, return_logits=True)
            energy_output = self.goal_energy(
                agent_feat=agent_feat,
                goal_candidates=goal_candidates_world,
                last_pos=obs_world[:, -1],
                edge_index=edge_index,
                relation_prob=relation_prob,
                edge_feat=edge_feat,
                edge_weight=edge_weight,
                scene_index=scene_index,
                return_full_matrix=for_loss,
            )

        refinement_anchor_goal = None
        refinement_output_goal = None
        if for_loss and self.args.continuous_refinement:
            # One deterministic complete proposal per social mode.  Training
            # selects the best current anchor per scene in the objective, so
            # the residual head improves localization without collapsing all
            # modes toward one endpoint.
            mode_candidate_index = distribution[
                'conditional_goal_log_prob'].detach().argmax(dim=-1)
            refinement_anchor_goal = goal_candidates_world.gather(
                1, mode_candidate_index.unsqueeze(-1).expand(-1, -1, 2))
            refinement_output_goal = self.continuous_goal_refiner(
                joint_goal=refinement_anchor_goal,
                last_pos=obs_world[:, -1],
                agent_feat=agent_feat,
                edge_index=edge_index,
                edge_feat=edge_feat,
                edge_weight=edge_weight,
                relation_prob=relation_prob,
            )

        structured = dict(distribution)
        structured.update({
            'goal_candidates_map': goal_candidates_map,
            'goal_candidates_world': goal_candidates_world,
            'candidate_prob': candidate_prob,
            'candidate_heatmap_prob': candidate_heatmap_prob,
            'candidate_mask': candidate_mask,
            'agent_feat': agent_feat,
            'edge_index': edge_index,
            'edge_feat': edge_feat,
            'edge_weight': edge_weight,
            'graph_gate_prob': graph_gate_prob,
            'relation_prob': relation_prob,
            'relation_logits': relation_logits,
            'energy_output': energy_output,
            'refinement_anchor_goal': refinement_anchor_goal,
            'refinement_output_goal': refinement_output_goal,
        })

        tensors_to_check = {
            key: value for key, value in structured.items()
            if torch.is_tensor(value) and value.is_floating_point()
        }
        if energy_output is not None:
            tensors_to_check.update({
                f'energy_output.{key}': value
                for key, value in energy_output.items()
                if torch.is_tensor(value) and value.is_floating_point()
            })
        if any(not torch.isfinite(value).all()
               for value in tensors_to_check.values()):
            bad = [key for key, value in tensors_to_check.items()
                   if not torch.isfinite(value).all()]
            raise FloatingPointError(
                f'Non-finite structured goal tensors: {bad}')

        if if_test:
            sampling_energy = (energy_output
                               if self.args.joint_goal_enabled else None)
            branch_candidate_mask = candidate_mask
            if (self.args.joint_sampling_strategy == 'coverage' and
                    self.args.use_ttst and
                    goal_candidates_world.shape[1] - 1 >=
                    self.args.num_samples):
                # TTST appends its heatmap argmax after the cluster centers.
                # Keep that deterministic point for the diffusion-tree trunk,
                # while the S branches retain the original S-center coverage.
                branch_candidate_mask = candidate_mask.clone()
                branch_candidate_mask[:, -1] = False
            branch_details = self.joint_sampler.sample(
                goal_candidates_world,
                distribution['mode_log_prob'],
                distribution['conditional_goal_log_prob'],
                scene_index=scene_index,
                energy_output=sampling_energy,
                candidate_mask=branch_candidate_mask,
                num_samples=self.args.num_samples,
                sampling_mode=self.args.joint_sampling_mode,
                return_details=True,
            )
            if self.args.use_ttst and goal_candidates_world.shape[1] > 1:
                trunk_candidate_index = torch.full(
                    (goal_candidates_world.shape[0], 1),
                    goal_candidates_world.shape[1] - 1,
                    dtype=torch.long, device=goal_candidates_world.device)
            else:
                trunk_candidate_index = candidate_heatmap_prob.argmax(
                    dim=-1, keepdim=True)
            trunk_gather_index = trunk_candidate_index.unsqueeze(-1).expand(
                -1, -1, 2)
            trunk_goal_points = goal_candidates_world.gather(
                1, trunk_gather_index)
            trunk_social_mode = distribution['mode_log_prob'].argmax(
                dim=-1, keepdim=True)
            all_candidate_index = torch.cat(
                (branch_details['candidate_index'],
                 trunk_candidate_index), dim=1)                # [N,S+1]
            gather_index = all_candidate_index.unsqueeze(-1).expand(-1, -1, 2)
            unrefined_goal_points_map = goal_candidates_map.gather(
                1, gather_index)                              # [N,S+1,2]
            unrefined_goal_points_world = torch.cat(
                (branch_details['joint_goal_points'],
                 trunk_goal_points), dim=1)
            if self.args.continuous_refinement:
                refined_goal_points_world = self.continuous_goal_refiner(
                    joint_goal=unrefined_goal_points_world,
                    last_pos=obs_world[:, -1],
                    agent_feat=agent_feat,
                    edge_index=edge_index,
                    edge_feat=edge_feat,
                    edge_weight=edge_weight,
                    relation_prob=relation_prob,
                )
                structured['joint_goal_points_map'] = self._map_goals_to_map(
                    refined_goal_points_world, inputs['scene'])
                structured['joint_goal_points_world'] = \
                    refined_goal_points_world
                structured['joint_goal_points_world_unrefined'] = \
                    unrefined_goal_points_world
            else:
                structured['joint_goal_points_map'] = \
                    unrefined_goal_points_map
                structured['joint_goal_points_world'] = \
                    unrefined_goal_points_world
            structured['joint_candidate_index'] = all_candidate_index
            structured['sampled_social_mode'] = torch.cat(
                (branch_details['social_mode_index'],
                 trunk_social_mode), dim=1)

        self._record_joint_diagnostics(structured)
        return structured

    def prepare_inputs(self, batch_data, batch_id):
        """
        Prepare inputs to be fed to a generic model.
        """
        # we need to remove first dimension which is added by torch.DataLoader
        # float is needed to convert to 32bit float
        integer_inputs = {
            "scene_index", "scene_ptr", "frame_ids", "batch_format_version"
        }
        def prepare_value(key, value):
            if isinstance(value, dict):
                return {nested_key: prepare_value(nested_key, nested_value)
                        for nested_key, nested_value in value.items()}
            if torch.is_tensor(value):
                value = value.squeeze(0).to(self.device)
                return value.long() if key in integer_inputs else value.float()
            return value

        selected_inputs = {}
        for key, value in batch_data.items():
            selected_inputs[key] = prepare_value(key, value)
        # extract seq_list
        seq_list = selected_inputs["seq_list"]

        # state augmentation to [T, B, 8]: rel_px, rel_py, vx, vy, ax, ay, abs_px, abs_py
        num_agents = selected_inputs["abs_pixel_coord"].shape[1]
        abs_x = selected_inputs["abs_pixel_coord"]
        x_augmented = torch.zeros(self.args.seq_length, num_agents, 8, device=self.device)
        x_augmented[:,:,6:] = abs_x # original absolute px, py

        x = abs_x[:self.args.obs_length,:,:2] - abs_x[self.args.obs_length-1,:,:2] 
        vx = derivative_of(x, dt=1) 
        ax = derivative_of(vx, dt=1) 
        x_augmented[:self.args.obs_length,:,:2] = x
        x_augmented[:self.args.obs_length,:,2:4] = vx
        x_augmented[:self.args.obs_length,:,4:6] = ax

        y = abs_x[self.args.obs_length:,:,:2] - abs_x[self.args.obs_length-1,:,:2] 
        vy = derivative_of(y, dt=1) 
        ay = derivative_of(vy, dt=1) 
        x_augmented[self.args.obs_length:,:,:2] = y
        x_augmented[self.args.obs_length:,:,2:4] = vy
        x_augmented[self.args.obs_length:,:,4:6] = ay

        selected_inputs["x_augmented"] = x_augmented # TBC

        scene_name = batch_id["scene_name"][0]
        scene = self.dataset.scenes[scene_name]
        selected_inputs["scene"] = scene

        if "scene_index" not in selected_inputs:
            selected_inputs["scene_index"] = torch.zeros(
                num_agents, dtype=torch.long, device=self.device)
        if self.active_goal_model_type != 'independent':
            version = selected_inputs.get("batch_format_version")
            if version is None or int(version.item()) != 2:
                raise RuntimeError(
                    'Social goal models require synchronized scene-window '
                    'batches (format v2). Re-run preprocessing with '
                    '--force_reprocess True.')
            frame_ids = selected_inputs.get("frame_ids")
            if frame_ids is None or frame_ids.shape != (
                    self.args.seq_length, num_agents):
                raise RuntimeError('Missing synchronized frame_ids [T,N].')
            if not frame_ids.eq(frame_ids[:, :1]).all():
                raise RuntimeError(
                    'A social batch contains non-synchronized agent columns.')

        # Graph, TTC, collision and structured losses use one coordinate
        # system across ETH/UCY, SDD and inD: world metres.
        full_pixel_coord = selected_inputs["abs_pixel_coord"] * \
            self.args.down_factor
        world_coord = scene.make_world_coord_torch(full_pixel_coord)
        selected_inputs["world_coord"] = world_coord
        selected_inputs["obs_traj_world"] = world_coord[
            :self.args.obs_length].permute(1, 0, 2).contiguous()

        return selected_inputs, seq_list.detach()

    def init_losses(self):
        if self.jdv2_active:
            stage = self.args.training_stage
            if stage == 'joint_goal':
                if self.strict_no_z:
                    return {
                        'jdv2_pl_post': 0,
                        'jdv2_pl_prior': 0,
                        'jdv2_relation_kl': 0}
                return {
                    'jdv2_mixture_pl': 0,
                    'jdv2_posterior_distill': 0,
                    'jdv2_relation_kl': 0}
            if stage == 'joint_trajectory':
                return {'jdv2_diffusion_loss': 0,
                        'jdv2_relative_loss': 0}
            losses = {
                'jdv2_mixture_pl': 0,
                'jdv2_posterior_distill': 0,
                'jdv2_relation_kl': 0}
            if self.args.use_dependency_corrector:
                losses.update({'jdv2_diffusion_loss': 0,
                               'jdv2_relative_loss': 0})
            return losses
        if self.args.training_stage == 'multiway_coupling':
            return {
                'loss_alignment': 0,
                'loss_pair_score': 0,
                'loss_pair_assignment': 0,
                'loss_no_harm': 0,
                'loss_perm_entropy': 0,
                'loss_relation_prior': 0,
                'loss_gate_reg': 0,
            }
        if self.args.training_stage == 'alignment':
            return {
                name: 0 for name, weight in (
                    ('trajectory_alignment_loss',
                     self.args.lambda_trajectory_alignment),
                    ('trajectory_pair_loss',
                     self.args.lambda_trajectory_pair),
                    ('alignment_no_harm_loss',
                     self.args.lambda_alignment_no_harm),
                    ('alignment_entropy_loss',
                     self.args.lambda_alignment_entropy),
                ) if weight > 0
            }
        losses = {
            "diffusion_loss": 0,
            "goal_BCE_loss": 0,
        }
        model_type = self.active_goal_model_type
        if model_type in {'social', 'lowrank', 'joint'}:
            losses['mode_loss'] = 0
        if model_type in {'energy', 'joint'}:
            losses.update({
                'pseudo_likelihood_loss': 0,
                'relation_entropy_loss': 0,
                'relation_balance_loss': 0,
            })
        if model_type in {'lowrank', 'joint'}:
            losses['mode_balance_loss'] = 0
        if (model_type in {'lowrank', 'joint'} and
                self.args.lambda_joint_rank > 0):
            losses['joint_rank_loss'] = 0
        if self.args.adaptive_graph and self.args.lambda_graph_density > 0:
            losses['graph_density_loss'] = 0
        if (self.args.continuous_refinement and
                self.args.lambda_continuous_refinement > 0):
            losses['continuous_refinement_loss'] = 0
        if (self.args.continuous_refinement and
                self.args.lambda_refinement_delta > 0):
            losses['refinement_delta_loss'] = 0
        if self.args.trajectory_alignment:
            for name, weight in (
                    ('trajectory_alignment_loss',
                     self.args.lambda_trajectory_alignment),
                    ('trajectory_pair_loss',
                     self.args.lambda_trajectory_pair),
                    ('alignment_no_harm_loss',
                     self.args.lambda_alignment_no_harm),
                    ('alignment_entropy_loss',
                     self.args.lambda_alignment_entropy)):
                if weight > 0:
                    losses[name] = 0
        return losses

    def set_losses_coeffs(self):

        if self.jdv2_active:
            beta = (0.1 if self.args.training_stage == 'joint_finetune'
                    else jdv2_warmup_beta(self.args.jdv2_stage_progress))
            goal_scale = (self.args.lambda_JG
                          if self.args.training_stage == 'joint_finetune'
                          else 1.0)
            coefficients = {}
            if self.args.training_stage in {'joint_goal', 'joint_finetune'}:
                if self.strict_no_z:
                    coefficients.update({
                        'jdv2_pl_post': 0.5 * goal_scale,
                        'jdv2_pl_prior': 0.5 * goal_scale,
                        'jdv2_relation_kl': beta * goal_scale,
                    })
                else:
                    coefficients.update({
                        'jdv2_mixture_pl': goal_scale,
                        'jdv2_posterior_distill': beta * goal_scale,
                        'jdv2_relation_kl': beta * goal_scale,
                    })
            if self.args.training_stage in {
                    'joint_trajectory', 'joint_finetune'} and \
                    self.args.use_dependency_corrector:
                coefficients.update({
                    'jdv2_diffusion_loss': self.args.lambda_diff,
                    'jdv2_relative_loss': self.args.lambda_relative,
                })
            return coefficients

        if self.args.training_stage == 'multiway_coupling':
            return {
                'loss_alignment': self.args.lambda_alignment,
                'loss_pair_score': self.args.lambda_pair_score,
                'loss_pair_assignment': (
                    self.args.lambda_pair_assignment
                    if self.args.use_pair_assignment_loss else 0.0),
                'loss_no_harm': self.args.lambda_no_harm,
                'loss_perm_entropy': self.args.lambda_perm_entropy,
                'loss_relation_prior': self.args.lambda_relation_prior,
                'loss_gate_reg': self.args.lambda_gate_reg,
            }

        if self.args.training_stage == 'alignment':
            return {
                name: weight for name, weight in (
                    ('trajectory_alignment_loss',
                     self.args.lambda_trajectory_alignment),
                    ('trajectory_pair_loss',
                     self.args.lambda_trajectory_pair),
                    ('alignment_no_harm_loss',
                     self.args.lambda_alignment_no_harm),
                    ('alignment_entropy_loss',
                     self.args.lambda_alignment_entropy),
                ) if weight > 0
            }

        losses_coeffs = {
            "diffusion_loss": (1 if self.active_goal_model_type == 'independent'
                               else self.args.lambda_diff),
            "goal_BCE_loss": (20 if self.active_goal_model_type == 'independent'
                              else self.args.lambda_goal),
        }
        model_type = self.active_goal_model_type
        if model_type in {'social', 'lowrank', 'joint'}:
            losses_coeffs['mode_loss'] = self.args.lambda_mode
        if model_type in {'energy', 'joint'}:
            losses_coeffs.update({
                'pseudo_likelihood_loss': self.args.lambda_PL,
                'relation_entropy_loss': self.args.lambda_relation_entropy,
                'relation_balance_loss': self.args.lambda_relation_balance,
            })
        if model_type in {'lowrank', 'joint'}:
            losses_coeffs['mode_balance_loss'] = self.args.lambda_mode_balance
        if (model_type in {'lowrank', 'joint'} and
                self.args.lambda_joint_rank > 0):
            losses_coeffs['joint_rank_loss'] = self.args.lambda_joint_rank
        if self.args.adaptive_graph and self.args.lambda_graph_density > 0:
            losses_coeffs['graph_density_loss'] = \
                self.args.lambda_graph_density
        if (self.args.continuous_refinement and
                self.args.lambda_continuous_refinement > 0):
            losses_coeffs['continuous_refinement_loss'] = \
                self.args.lambda_continuous_refinement
        if (self.args.continuous_refinement and
                self.args.lambda_refinement_delta > 0):
            losses_coeffs['refinement_delta_loss'] = \
                self.args.lambda_refinement_delta
        if self.args.trajectory_alignment:
            for name, weight in (
                    ('trajectory_alignment_loss',
                     self.args.lambda_trajectory_alignment),
                    ('trajectory_pair_loss',
                     self.args.lambda_trajectory_pair),
                    ('alignment_no_harm_loss',
                     self.args.lambda_alignment_no_harm),
                    ('alignment_entropy_loss',
                     self.args.lambda_alignment_entropy)):
                if weight > 0:
                    losses_coeffs[name] = weight
        return losses_coeffs
    
    
    def init_train_metrics(self):
        train_metrics = {
            "ADE": [],
            "FDE": [],
        }
        return train_metrics

    def init_test_metrics(self):
        if self.args.trajectory_coupling == 'multiway_v4':
            return {
                'ADE': [],
                'FDE': [],
                'ADE_world': [],
                'FDE_world': [],
                'minADE@K': [],
                'minFDE@K': [],
                'Raw_minADE': [],
                'Aligned_minADE': [],
                'Raw_minFDE': [],
                'Aligned_minFDE': [],
                'Raw_JADE': [],
                'JADE': [],
                'Raw_JFDE': [],
                'JFDE': [],
            }
        test_metrics = {
            "ADE": [],
            "FDE": [],
            "ADE_world": [],
            "FDE_world": [],
            "minADE@K": [],
            "minFDE@K": [],
        }
        if self.active_goal_model_type != 'independent':
            test_metrics.update({
                "JADE": [],
                "JFDE": [],
                "Goal_minFDE": [],
                "Joint_Goal_Endpoint_Error": [],
                "Joint_Goal_Compatibility": [],
            })
            if self.jdv2_active:
                test_metrics['Relative_Motion_Error'] = []
            if self.args.trajectory_alignment:
                test_metrics.update({
                    "Raw_JADE": [],
                    "Raw_JFDE": [],
                })
            if self.args.collision_threshold_meter is not None:
                test_metrics["Collision_Rate"] = []
            if self.args.goal_recall_threshold_meter is not None:
                test_metrics["Goal_Recall@K"] = []
        return test_metrics

    def init_best_metrics(self):
        best_metrics = {
            "ADE": 1e9,
            "FDE": 1e9,
            "ADE_world": 1e9,
            "FDE_world": 1e9,
            "minADE@K": 1e9,
            "minFDE@K": 1e9,
        }
        if self.active_goal_model_type != 'independent':
            best_metrics.update({"JADE": 1e9, "JFDE": 1e9})
        return best_metrics

    def best_valid_metric(self):
        if self.args.best_metric != 'auto':
            return self.args.best_metric
        # Legacy GDTS retains its historical pixel ADE selection. Structured
        # experiments default to a coherent, world-coordinate scene metric.
        if not self.jdv2_active:
            return ('ADE' if self.active_goal_model_type == 'independent'
                    else 'JADE')
        if self.args.training_stage == 'joint_goal':
            return 'JFDE'
        return 'JADE'

    def compute_loss_mask(self, seq_list, obs_length: int = 8):
        """
        Get a mask to denote whether to account predictions during loss
        computation. It is supposed to calculate losses for a person at
        time t only if his data exists from time 0 to time t.

        Parameters
        ----------
        seq_list : PyTorch tensor
            input is seq_list[1:]. Size = (seq_len,N_pedestrians). Boolean mask
            that is =1 if pedestrian i is present at time-step t.
        obs_length : int
            number of observation time-steps

        Returns
        -------
        loss_mask : PyTorch tensor
            Shape: (seq_len,N_pedestrians)
            loss_mask[t,i] = 1 if pedestrian i if present from beginning till time t
        """
        loss_mask = seq_list.cumprod(dim=0)
        # we should not compute losses for step 0, as ground_truth and
        # predictions are always equal there
        loss_mask[0:obs_length] = 1
        return loss_mask
    
    def compute_model_metrics(self,
                              metric_name,
                              predictions,
                              metric_mask,
                              all_aux_outputs,
                              inputs,
                              obs_length=8):
        """
        Compute model metrics for a generic model.
        Return a list of floats (the given metric values computed on the batch)
        """

        # scale back to original dimension
        predictions = predictions.detach() * self.args.down_factor
        ground_truth = inputs["x_augmented"][:,:,6:8].detach()
        ground_truth = ground_truth.detach() * self.args.down_factor

        if metric_name in {
                'Raw_JADE', 'Raw_JFDE', 'Raw_minADE', 'Raw_minFDE'}:
            # Evaluation receives aligned samples.  Invert every agent's hard
            # permutation to report a paired, same-random-draw control metric.
            permutation = all_aux_outputs.get(
                'multiway_coupling_permutation')
            if permutation is None:
                permutation = all_aux_outputs.get(
                    'trajectory_alignment_permutation')
            if permutation is None:
                raise RuntimeError(
                    'Raw joint metrics require trajectory alignment metadata')
            inverse = torch.empty_like(permutation)
            local_index = torch.arange(
                permutation.shape[1], device=permutation.device)[None].expand_as(
                    permutation)
            inverse.scatter_(1, permutation, local_index)
            agent_first = predictions.permute(2, 0, 1, 3)
            raw_agent_first = agent_first.gather(
                1, inverse[:, :, None, None].expand(
                    -1, -1, predictions.shape[1], 2))
            predictions = raw_agent_first.permute(1, 2, 0, 3).contiguous()

        # convert to world coordinates
        scene = inputs["scene"]
        pred_world = []
        for i in range(predictions.shape[0]):
            pred_world.append(scene.make_world_coord_torch(predictions[i]))
        pred_world = torch.stack(pred_world)

        GT_world = scene.make_world_coord_torch(ground_truth)


        scene_index = inputs.get("scene_index")
        if metric_name == 'ADE':
            return ADE_best_of(predictions, ground_truth, metric_mask, obs_length)
        elif metric_name == 'FDE':
            return FDE_best_of(predictions, ground_truth, metric_mask, obs_length)
        elif metric_name == 'ADE_world':
            return ADE_best_of(pred_world, GT_world, metric_mask, obs_length)
        elif metric_name == 'FDE_world':
            return FDE_best_of(pred_world, GT_world, metric_mask, obs_length)
        elif metric_name == 'minADE@K':
            return minADE_at_K(pred_world, GT_world, metric_mask, obs_length)
        elif metric_name == 'minFDE@K':
            return minFDE_at_K(pred_world, GT_world, metric_mask, obs_length)
        elif metric_name in {'Raw_minADE', 'Aligned_minADE'}:
            return minADE_at_K(
                pred_world, GT_world, metric_mask, obs_length)
        elif metric_name in {'Raw_minFDE', 'Aligned_minFDE'}:
            return minFDE_at_K(
                pred_world, GT_world, metric_mask, obs_length)
        elif metric_name == 'JADE':
            return JADE(pred_world, GT_world, metric_mask, scene_index,
                        obs_length)
        elif metric_name == 'JFDE':
            return JFDE(pred_world, GT_world, metric_mask, scene_index,
                        obs_length)
        elif metric_name == 'Raw_JADE':
            return JADE(pred_world, GT_world, metric_mask, scene_index,
                        obs_length)
        elif metric_name == 'Raw_JFDE':
            return JFDE(pred_world, GT_world, metric_mask, scene_index,
                        obs_length)
        elif metric_name == 'Collision_Rate':
            return collision_rate(
                pred_world,
                collision_threshold_meter=self.args.collision_threshold_meter,
                metric_mask=metric_mask,
                scene_index=scene_index,
                obs_length=obs_length,
                method='segment',
                interpolation_steps=self.args.collision_interpolation_steps,
            )
        elif metric_name == 'Relative_Motion_Error':
            edge_index = all_aux_outputs['edge_index'].long()
            if edge_index.shape[1] == 0:
                return [0.0]
            src, dst = edge_index
            predicted_relative = (
                pred_world[:, obs_length:, dst] -
                pred_world[:, obs_length:, src])
            target_relative = GT_world[obs_length:, dst] - \
                GT_world[obs_length:, src]
            error = torch.linalg.vector_norm(
                predicted_relative - target_relative[None], dim=-1)
            _, compact = canonicalize_scene_index(
                scene_index, GT_world.shape[1], GT_world.device)
            values = []
            for scene_id in range(int(compact.max().item()) + 1):
                scene_edges = compact[src].eq(scene_id)
                if not bool(scene_edges.any()):
                    values.append(0.0)
                    continue
                per_sample = error[:, :, scene_edges].mean(dim=(1, 2))
                values.append(float(per_sample.min().detach().cpu()))
            return values
        elif metric_name in {'Goal_minFDE', 'Goal_Recall@K'}:
            # Candidate recall measures the K heatmap candidates, independently
            # of how many joint diffusion branches S are requested.
            goal_world = all_aux_outputs[
                'goal_candidates_world'].permute(1, 0, 2)       # [K,N,2]
            goal_gt_world = GT_world[-1]
            if metric_name == 'Goal_minFDE':
                return goal_minFDE(goal_world, goal_gt_world, metric_mask)
            return goal_recall_at_K(
                goal_world, goal_gt_world,
                threshold_meter=self.args.goal_recall_threshold_meter,
                metric_mask=metric_mask)
        elif metric_name in {
                'Joint_Goal_Endpoint_Error', 'Joint_Goal_Compatibility'}:
            # goal_point contains S branch goals followed by one tree-trunk
            # MAP/argmax goal. Metrics intentionally exclude the trunk entry.
            goal_pixel = all_aux_outputs["goal_point"][
                :self.args.num_samples] * self.args.down_factor
            goal_world = scene.make_world_coord_torch(goal_pixel)
            goal_gt_world = GT_world[-1]
            if metric_name == 'Joint_Goal_Endpoint_Error':
                return joint_goal_endpoint_error(
                    goal_world, goal_gt_world, metric_mask, scene_index)
            return joint_goal_compatibility(
                goal_world, goal_gt_world, metric_mask, scene_index)
        else:
            raise ValueError("This metric has not been implemented yet!")

    def _legacy_encode(self, inputs, if_test=False):
        x = inputs["x_augmented"].detach().clone() # T, B, 8
        num_agents = x.shape[1]

        # START SAMPLES LOOP
        all_context =[]
        all_aux_outputs = []
    
        ##################
        # PREDICT GOAL
        ##################
        tensor_image = inputs["tensor_image"].unsqueeze(0).repeat(num_agents, 1, 1, 1) 
        obs_traj_maps = inputs["input_traj_maps"][:, :self.args.obs_length]
        input_goal_module = torch.cat((tensor_image, obs_traj_maps), dim=1)
        goal_logit_map_start = self.goal_module(input_goal_module) # (num_agents, C+T, H, W) -> (num_agents, C_out, H, W), C_out = 12
        goal_prob_map = torch.sigmoid(goal_logit_map_start[:, -1:]) 
        if if_test:
            goal_point_start = TTST_test_time_sampling_trick(
                goal_prob_map, num_goals=self.args.num_samples, device=self.device)
            goal_point_start = goal_point_start.squeeze(2).permute(1, 0, 2) # final result: (num_agents, num_samples, 2)
        else:
            goal_point_start = x[-1,:,6:8].repeat(self.args.num_samples+1, 1, 1) 
            goal_point_start = goal_point_start.permute(1,0,2) # (num_agents, num_samples, 2)

        x_ori = x[self.args.obs_length-1,:,6:8]
        ################################
        # History Trajectory Encoding
        ################################
        if if_test:
            num_iters = self.args.num_samples+1
        else:
            num_iters = 1
        for sample_idx in range(num_iters):
            goal_point = goal_point_start[:, sample_idx].to(self.device) # (B,num_sample,2)->(B,2)
            goal_point = goal_point.detach()

            # agents trajectory up to now
            current_agents = torch.zeros([num_agents, self.args.obs_length, 8]).to(self.device)
            current_agents[:,:,2:] = x[:self.args.obs_length,:,:6].permute(1,0,2)
            current_agents[:,:,:2] = x[:self.args.obs_length,:,6:8].permute(1,0,2) - goal_point.unsqueeze(1)

            temporal_input_embedded = self.encoder.encode_hist(node_hist=current_agents, dropout_keep_prob=1) # [B, To, 2] -> [B, embedding_size], batch_first=True
            temporal_input_embedded = temporal_input_embedded.unsqueeze(1) # (B,1,embedding_size)


            aux_outputs = {
            "goal_logit_map": goal_logit_map_start,
            "goal_point": goal_point, # B, 2
            }

            all_context.append(temporal_input_embedded)
            all_aux_outputs.append(aux_outputs)

        all_context  = torch.stack(all_context)
        all_aux_outputs = {k: torch.stack([d[k] for d in all_aux_outputs])
                        for k in all_aux_outputs[0].keys()}
        return all_context, all_aux_outputs # (20,Tp+Tf,B,2) 

    def encode(self, inputs, if_test=False, for_loss=False):
        """Encode one teacher-forced or jointly sampled goal per branch.

        The independent path calls the original implementation unchanged.
        Social variants only replace endpoint selection; the legacy
        goal-relative history LSTM remains the diffusion context encoder.
        """
        if self.active_goal_model_type == 'independent':
            return self._legacy_encode(inputs, if_test=if_test)

        x = inputs['x_augmented'].detach().clone()              # [T,N,8]
        num_agents = x.shape[1]
        tensor_image = inputs['tensor_image'].unsqueeze(0).repeat(
            num_agents, 1, 1, 1)
        obs_traj_maps = inputs['input_traj_maps'][
            :, :self.args.obs_length]
        goal_module_input = torch.cat((tensor_image, obs_traj_maps), dim=1)
        goal_logit_map = self.goal_module(goal_module_input)    # [N,Tf,H,W]
        goal_prob_map = torch.sigmoid(goal_logit_map[:, -1:])   # [N,1,H,W]
        if self.jdv2_active:
            return self._jdv2_encode(
                inputs, goal_logit_map, goal_prob_map, if_test, for_loss)
        structured = self._structured_goal_outputs(
            inputs, goal_prob_map, if_test=if_test, for_loss=for_loss)

        if if_test:
            # [N,S+1,2], with each column a complete scene configuration.
            goal_point_start = structured['joint_goal_points_map']
            num_iters = self.args.num_samples + 1
        else:
            # Keep the original teacher-forced diffusion loss exactly: joint
            # objectives train candidate selection in parallel, while the
            # denoiser is conditioned on the true endpoint during training.
            goal_point_start = x[-1, :, 6:8].unsqueeze(1)       # [N,1,2]
            num_iters = 1

        all_context = []
        goal_points = []
        for sample_idx in range(num_iters):
            goal_point = goal_point_start[:, sample_idx].to(
                self.device).detach()                           # [N,2]
            current_agents = torch.zeros(
                (num_agents, self.args.obs_length, 8),
                device=self.device)
            current_agents[:, :, 2:] = x[
                :self.args.obs_length, :, :6].permute(1, 0, 2)
            current_agents[:, :, :2] = x[
                :self.args.obs_length, :, 6:8].permute(1, 0, 2) - \
                goal_point.unsqueeze(1)
            context = self.encoder.encode_hist(
                node_hist=current_agents,
                dropout_keep_prob=1).unsqueeze(1)               # [N,1,D]
            all_context.append(context)
            goal_points.append(goal_point)

        aux_outputs = dict(structured)
        # Retain legacy public layouts expected by Goal_BCE_loss and metrics.
        aux_outputs['goal_logit_map'] = goal_logit_map.unsqueeze(0).expand(
            num_iters, -1, -1, -1, -1)
        aux_outputs['goal_point'] = torch.stack(goal_points)    # [I,N,2]
        return torch.stack(all_context), aux_outputs            # [I,N,1,D]

    def _future_predictions_world(self, predictions, inputs):
        """Convert ``[K,T,N,2]`` map coordinates to ``[N,K,Tf,2]`` metres."""
        future_pixel = predictions[:, self.args.obs_length:] * \
            float(self.args.down_factor)
        scene = inputs['scene']
        future_world = torch.stack([
            scene.make_world_coord_torch(sample)
            for sample in future_pixel
        ])
        return future_world.permute(2, 0, 1, 3).contiguous()

    def _decode_contexts(self, inputs, all_context):
        """Run the unchanged diffusion tree and integrate velocity outputs."""
        x = inputs['x_augmented'].detach().clone()
        num_agents = x.shape[1]
        velocity = self.ts_sample(all_context=all_context)
        predictions = torch.zeros(
            [self.args.num_samples, self.args.seq_length, num_agents, 2],
            device=self.device, dtype=x.dtype)
        predictions[:, :self.args.obs_length] = x[
            :self.args.obs_length, :, 6:8].repeat(
                self.args.num_samples, 1, 1, 1)
        for step in range(self.args.obs_length, self.args.seq_length):
            predictions[:, step] = (
                predictions[:, step - 1] +
                velocity[:, :, step - self.args.obs_length])
        return predictions

    def _v4_upstream_predictions(self, inputs):
        """Generate the actual fixed K-bank selected by the V4 upstream flag."""
        context = (torch.no_grad()
                   if self.args.freeze_upstream_generator else nullcontext())
        with context:
            if self.args.upstream_generator == 'stage0_independent':
                all_context, auxiliary = self._legacy_encode(
                    inputs, if_test=True)
            elif self.args.upstream_generator == 'stage1_joint':
                all_context, auxiliary = self.encode(
                    inputs, if_test=True)
            else:
                raise ValueError(
                    'Unsupported upstream_generator='
                    f'{self.args.upstream_generator!r}')
            predictions = self._decode_contexts(inputs, all_context)
        if self.args.freeze_upstream_generator:
            predictions = predictions.detach()
            auxiliary = {
                key: (value.detach() if torch.is_tensor(value) else value)
                for key, value in auxiliary.items()
            }
        return predictions, auxiliary

    def _v4_graph_inputs(self, inputs, upstream_auxiliary):
        """Return observation graph and optional Stage1 relation prior."""
        if (self.args.upstream_generator == 'stage1_joint' and
                'edge_index' in upstream_auxiliary):
            return (
                upstream_auxiliary['edge_index'],
                upstream_auxiliary['edge_feat'],
                upstream_auxiliary['edge_weight'],
                upstream_auxiliary.get('relation_prob'))
        edge_index, edge_feat, edge_weight = self.interaction_graph(
            inputs['obs_traj_world'], scene_index=inputs['scene_index'])
        # Stage0 is intentionally a pure marginal generator.  It has no
        # learned social prior, so q0 is the explicit uniform distribution
        # supplied by CandidateConditionedRelation rather than a frozen random
        # Stage1 head.
        return edge_index, edge_feat, edge_weight, None

    def _run_multiway_coupling(self, predictions, auxiliary, inputs, hard):
        """Apply V4 to a real trajectory bank and expose auditable metadata."""
        trajectories_world = self._future_predictions_world(
            predictions, inputs)
        edge_index, edge_feat, edge_weight, relation_prior = \
            self._v4_graph_inputs(inputs, auxiliary)
        coupling = self.multiway_coupler(
            trajectories=trajectories_world,
            last_pos=inputs['obs_traj_world'][:, -1],
            edge_index=edge_index,
            edge_feat=edge_feat,
            edge_weight=edge_weight,
            relation_prior=relation_prior,
            scene_index=inputs['scene_index'],
            hard=hard,
        )
        if hard:
            agent_first = predictions.permute(2, 0, 1, 3)
            permutation = coupling['permutation']
            predictions = agent_first.gather(
                1, permutation[:, :, None, None].expand(
                    -1, -1, predictions.shape[1], 2)
            ).permute(1, 2, 0, 3).contiguous()

        auxiliary.update({
            'multiway_coupling_permutation': coupling.get('permutation'),
            'multiway_soft_permutation': coupling['soft_permutation'],
            'multiway_synchronized_permutation': coupling[
                'synchronized_permutation'],
            'multiway_component_ids': coupling['component_ids'],
            'multiway_component_sizes': coupling['component_sizes'],
            'multiway_keep_confidence': coupling['keep_confidence'],
            'multiway_pair_gate': coupling['pair_gate'],
            'multiway_relation_posterior': coupling[
                'relation_posterior'],
            'multiway_relation_prior': coupling['relation_prior'],
            'multiway_edge_index': edge_index,
            'multiway_edge_weight': edge_weight,
            'multiway_raw_future_world': trajectories_world,
            'multiway_greedy_permutation': coupling.get(
                'greedy_permutation'),
            'multiway_changed_fraction': coupling.get(
                'changed_fraction'),
            'multiway_projection_metadata': coupling.get(
                'projection_metadata'),
        })
        relation_probability = coupling['relation_posterior'].clamp_min(1e-8)
        relation_entropy = (
            -(relation_probability * relation_probability.log()).sum(-1).mean()
            if relation_probability.numel() else
            trajectories_world.sum() * 0.0)
        num_agents = trajectories_world.shape[0]
        num_edges = edge_index.shape[1]
        self.last_joint_diagnostics.update({
            'avg_num_agents': float(num_agents),
            'avg_num_edges': float(num_edges),
            'avg_degree': float(2 * num_edges / max(num_agents, 1)),
            'max_component_size': float(
                coupling['component_sizes'].max().detach().cpu()),
            'keep_confidence': float(
                coupling['keep_confidence'].mean().detach().cpu()),
            'mean_pair_gate': (
                float(coupling['pair_gate'].mean().detach().cpu())
                if coupling['pair_gate'].numel() else 0.0),
            'relation_entropy': float(relation_entropy.detach().cpu()),
            'permutation_entropy': float(
                coupling['permutation_entropy'].detach().cpu()),
            'sinkhorn_row_error': float(
                coupling['sinkhorn_row_error'].detach().cpu()),
            'sinkhorn_col_error': float(
                coupling['sinkhorn_col_error'].detach().cpu()),
            'cycle_consistency_error': float(
                coupling['cycle_consistency_error'].detach().cpu()),
        })
        if coupling.get('changed_fraction') is not None:
            self.last_joint_diagnostics['changed_fraction'] = float(
                coupling['changed_fraction'].detach().cpu())
        return predictions, auxiliary, coupling

    def prepare_cached_trajectory_pack(self, pack):
        """Move one cached pack to the device and build isolated scene graphs.

        Graphs are deliberately constructed one scene at a time.  Local edge
        indices are offset only after graph construction, which makes a
        cross-scene edge impossible even when two scenes contain identical
        coordinates.
        """
        integer_keys = {
            'scene_index', 'scene_ptr', 'cached_seed_indices', 'cached_seeds'
        }
        inputs = {}
        for key, value in pack.items():
            if torch.is_tensor(value):
                value = value.to(self.device, non_blocking=True)
                value = value.long() if key in integer_keys else (
                    value if value.dtype == torch.bool else value.float())
            inputs[key] = value
        required = {
            'obs_world', 'gt_future_world', 'raw_future_world',
            'future_mask', 'agent_mask', 'scene_index', 'scene_ptr'
        }
        missing = required - set(inputs)
        if missing:
            raise ValueError(f'Cached trajectory pack missing {sorted(missing)}')
        raw = inputs['raw_future_world']
        if raw.ndim != 4 or raw.shape[1] != self.args.num_samples:
            raise ValueError(
                'Cached raw_future_world must have [N,K,T_pred,2] with '
                f'K={self.args.num_samples}; cached seeds must not be merged.')
        scene_ptr = inputs['scene_ptr']
        if (scene_ptr.ndim != 1 or scene_ptr.numel() < 2 or
                int(scene_ptr[0]) != 0 or
                int(scene_ptr[-1]) != raw.shape[0] or
                not bool((scene_ptr[1:] >= scene_ptr[:-1]).all())):
            raise ValueError('scene_ptr is not a valid packed prefix sum')

        edge_indices = []
        edge_features = []
        edge_weights = []
        per_scene_edges = []
        for scene_id in range(scene_ptr.numel() - 1):
            start = int(scene_ptr[scene_id])
            end = int(scene_ptr[scene_id + 1])
            local_obs = inputs['obs_world'][start:end]
            local_scene_index = torch.zeros(
                end - start, dtype=torch.long, device=self.device)
            local_index, local_feat, local_weight = self.interaction_graph(
                local_obs, scene_index=local_scene_index)
            edge_indices.append(local_index + start)
            edge_features.append(local_feat)
            edge_weights.append(local_weight)
            per_scene_edges.append(int(local_index.shape[1]))
        if edge_indices:
            inputs['edge_index'] = torch.cat(edge_indices, dim=1)
            inputs['edge_feat'] = torch.cat(edge_features, dim=0)
            inputs['edge_weight'] = torch.cat(edge_weights, dim=0)
        else:
            inputs['edge_index'] = torch.empty(
                (2, 0), dtype=torch.long, device=self.device)
            inputs['edge_feat'] = inputs['obs_world'].new_empty(
                (0, EDGE_FEATURE_DIM))
            inputs['edge_weight'] = inputs['obs_world'].new_empty((0,))
        edge_index = inputs['edge_index']
        scene_index = inputs['scene_index']
        if edge_index.shape[1] and not torch.equal(
                scene_index[edge_index[0]], scene_index[edge_index[1]]):
            raise AssertionError('Packed graph contains a cross-scene edge')
        inputs['actual_edges_per_scene'] = per_scene_edges
        inputs['estimated_pair_elements'] = int(
            edge_index.shape[1] * self.args.num_samples ** 2)
        return inputs

    def cached_multiway_coupling(self, inputs, *, hard=False):
        """Run only trainable V4 modules on a cached packed trajectory bank."""
        coupling = self.multiway_coupler(
            trajectories=inputs['raw_future_world'],
            last_pos=inputs['obs_world'][:, -1],
            edge_index=inputs['edge_index'],
            edge_feat=inputs['edge_feat'],
            edge_weight=inputs['edge_weight'],
            relation_prior=None,
            scene_index=inputs['scene_index'],
            hard=hard,
        )
        self.last_joint_diagnostics.update({
            'avg_num_agents': float(inputs['raw_future_world'].shape[0]),
            'avg_num_edges': float(inputs['edge_index'].shape[1]),
            'avg_num_scenes': float(inputs['scene_ptr'].numel() - 1),
            'estimated_pair_elements': float(
                inputs['estimated_pair_elements']),
            'avg_degree': float(
                2 * inputs['edge_index'].shape[1] /
                max(inputs['raw_future_world'].shape[0], 1)),
            'max_component_size': float(
                coupling['component_sizes'].max().detach().cpu()),
            'keep_confidence': float(
                coupling['keep_confidence'].mean().detach().cpu()),
            'mean_pair_gate': (
                float(coupling['pair_gate'].mean().detach().cpu())
                if coupling['pair_gate'].numel() else 0.0),
            'permutation_entropy': float(
                coupling['permutation_entropy'].detach().cpu()),
            'sinkhorn_row_error': float(
                coupling['sinkhorn_row_error'].detach().cpu()),
            'sinkhorn_col_error': float(
                coupling['sinkhorn_col_error'].detach().cpu()),
        })
        return coupling

    def _cached_coupling_loss_details(self, inputs, coupling):
        loss_function = (
            compute_scene_balanced_multiway_coupling_loss
            if self.args.scene_balanced_loss else
            compute_multiway_coupling_loss)
        return loss_function(
            permutations=coupling['soft_permutation'],
            pair_scores=coupling['pair_score'],
            raw_future=coupling['raw_trajectories'],
            ground_truth=inputs['gt_future_world'],
            edge_index=inputs['edge_index'],
            future_mask=inputs['future_mask'],
            agent_mask=inputs['agent_mask'],
            scene_index=inputs['scene_index'],
            relation_posterior=coupling['relation_posterior'],
            relation_prior=coupling['relation_prior'],
            pair_gate=coupling['pair_gate'],
            edge_weight=inputs['edge_weight'],
            lambda_fde=self.args.coupling_fde_weight,
            lambda_rel_geom=self.args.coupling_rel_geom_weight,
            lambda_rel_end=self.args.coupling_rel_endpoint_weight,
            tau_joint=self.args.coupling_joint_temperature,
            tau_pair_target=self.args.coupling_pair_target_temperature,
            tau_pair_pred=self.args.coupling_pair_pred_temperature,
            no_harm_margin=self.args.coupling_no_harm_margin,
            weight_alignment=1.0,
            weight_pair_score=1.0,
            weight_pair_assignment=1.0,
            weight_no_harm=1.0,
            weight_perm_entropy=1.0,
            weight_relation_prior=1.0,
            weight_gate_reg=1.0,
            gate_regularization=(
                'l1' if self.args.lambda_gate_reg > 0 else 'none'),
            return_details=True,
        )

    def get_cached_multiway_loss(self, inputs):
        coupling = self.cached_multiway_coupling(inputs, hard=False)
        details = self._cached_coupling_loss_details(inputs, coupling)
        names = (
            'loss_alignment', 'loss_pair_score', 'loss_pair_assignment',
            'loss_no_harm', 'loss_perm_entropy', 'loss_relation_prior',
            'loss_gate_reg')
        losses = {name: details[name] for name in names}
        self.last_joint_diagnostics.update({
            name: float(value.detach().cpu())
            for name, value in losses.items()
        })
        self.last_joint_diagnostics['loss_total'] = float(sum(
            self.set_losses_coeffs()[name] * losses[name]
            for name in names).detach().cpu())
        if any(not torch.isfinite(value).all() for value in losses.values()):
            raise FloatingPointError(f'Non-finite packed V4 loss: {losses}')
        return losses, coupling

    @torch.no_grad()
    def cached_pack_metrics(self, inputs, coupling):
        """Evaluate every packed scene independently with unchanged metrics."""
        raw = inputs['raw_future_world']
        aligned = gather_trajectory_bank(raw, coupling['permutation'])
        result = {name: [] for name in self.init_test_metrics()}
        scene_ptr = inputs['scene_ptr']
        for scene_id, scene_name in enumerate(inputs['scene_names']):
            start = int(scene_ptr[scene_id])
            end = int(scene_ptr[scene_id + 1])
            obs = inputs['obs_world'][start:end]
            gt_future = inputs['gt_future_world'][start:end]
            mask = inputs['agent_mask'][start:end]
            raw_scene = raw[start:end]
            aligned_scene = aligned[start:end]
            num_samples = raw_scene.shape[1]
            obs_samples = obs.permute(1, 0, 2).unsqueeze(0).expand(
                num_samples, -1, -1, -1)
            raw_world = torch.cat(
                (obs_samples, raw_scene.permute(1, 2, 0, 3)), dim=1)
            aligned_world = torch.cat(
                (obs_samples, aligned_scene.permute(1, 2, 0, 3)), dim=1)
            gt_world = torch.cat(
                (obs, gt_future), dim=1).permute(1, 0, 2)
            local_scene_index = torch.zeros(
                end - start, dtype=torch.long, device=self.device)
            scene = self.dataset.scenes[scene_name]
            aligned_pixel = torch.stack([
                scene.make_pixel_coord_torch(sample)
                for sample in aligned_world
            ])
            gt_pixel = scene.make_pixel_coord_torch(gt_world)

            result['ADE'].extend(ADE_best_of(
                aligned_pixel, gt_pixel, mask, self.args.obs_length))
            result['FDE'].extend(FDE_best_of(
                aligned_pixel, gt_pixel, mask, self.args.obs_length))
            result['ADE_world'].extend(ADE_best_of(
                aligned_world, gt_world, mask, self.args.obs_length))
            result['FDE_world'].extend(FDE_best_of(
                aligned_world, gt_world, mask, self.args.obs_length))
            result['minADE@K'].extend(minADE_at_K(
                aligned_world, gt_world, mask, self.args.obs_length))
            result['minFDE@K'].extend(minFDE_at_K(
                aligned_world, gt_world, mask, self.args.obs_length))
            result['Raw_minADE'].extend(minADE_at_K(
                raw_world, gt_world, mask, self.args.obs_length))
            result['Aligned_minADE'].extend(minADE_at_K(
                aligned_world, gt_world, mask, self.args.obs_length))
            result['Raw_minFDE'].extend(minFDE_at_K(
                raw_world, gt_world, mask, self.args.obs_length))
            result['Aligned_minFDE'].extend(minFDE_at_K(
                aligned_world, gt_world, mask, self.args.obs_length))
            result['Raw_JADE'].extend(JADE(
                raw_world, gt_world, mask, local_scene_index,
                self.args.obs_length))
            result['JADE'].extend(JADE(
                aligned_world, gt_world, mask, local_scene_index,
                self.args.obs_length))
            result['Raw_JFDE'].extend(JFDE(
                raw_world, gt_world, mask, local_scene_index,
                self.args.obs_length))
            result['JFDE'].extend(JFDE(
                aligned_world, gt_world, mask, local_scene_index,
                self.args.obs_length))
        return result, aligned

    @torch.no_grad()
    def cached_pack_diagnostics(self, inputs, coupling, aligned):
        raw = inputs['raw_future_world']
        preservation = assert_marginal_preservation(
            raw, aligned, inputs['gt_future_world'],
            agent_mask=inputs['agent_mask'], tolerance=1e-6)
        hard = hard_alignment_diagnostics(
            raw_future=raw,
            ground_truth=inputs['gt_future_world'],
            permutation=coupling['permutation'],
            agent_mask=inputs['agent_mask'],
            scene_index=inputs['scene_index'],
            edge_index=inputs['edge_index'],
            lambda_fde=self.args.coupling_fde_weight,
            tolerance=1e-6,
        )
        greedy = hard_alignment_diagnostics(
            raw_future=raw,
            ground_truth=inputs['gt_future_world'],
            permutation=coupling['greedy_permutation'],
            agent_mask=inputs['agent_mask'],
            scene_index=inputs['scene_index'],
            edge_index=inputs['edge_index'],
            lambda_fde=self.args.coupling_fde_weight,
            tolerance=1e-6,
        )
        relation = coupling['relation_posterior'].clamp_min(1e-8)
        soft = compute_scene_balanced_multiway_coupling_loss(
            permutations=coupling['soft_permutation'], pair_scores=None,
            raw_future=raw, ground_truth=inputs['gt_future_world'],
            edge_index=inputs['edge_index'], agent_mask=inputs['agent_mask'],
            scene_index=inputs['scene_index'],
            lambda_fde=self.args.coupling_fde_weight,
            tau_joint=self.args.coupling_joint_temperature,
            weight_alignment=1.0, weight_pair_score=0.0,
            weight_pair_assignment=0.0, weight_no_harm=0.0,
            weight_perm_entropy=0.0, weight_relation_prior=0.0,
            weight_gate_reg=0.0)
        scalar = {
            'alignment_oracle': hard['alignment_oracle'],
            'alignment_headroom': hard['alignment_headroom'],
            'alignment_recovery_ratio': hard['alignment_recovery_ratio'],
            'delta_JADE': hard['delta_JADE'],
            'delta_JFDE': hard['delta_JFDE'],
            'changed_fraction': coupling['changed_fraction'],
            'keep_confidence': coupling['keep_confidence'].mean(),
            'mean_pair_gate': (coupling['pair_gate'].mean()
                               if coupling['pair_gate'].numel()
                               else raw.sum() * 0.0),
            'relation_entropy': (-(relation * relation.log()).sum(-1).mean()
                                 if relation.numel()
                                 else raw.sum() * 0.0),
            'permutation_entropy': coupling['permutation_entropy'],
            'sinkhorn_row_error': coupling['sinkhorn_row_error'],
            'sinkhorn_col_error': coupling['sinkhorn_col_error'],
            'soft_hard_gap': (
                soft['loss_alignment'] - hard['aligned_joint_cost']).abs(),
            'marginal_ADE_max_abs_error': preservation[
                'marginal_ADE_max_abs_error'],
            'marginal_FDE_max_abs_error': preservation[
                'marginal_FDE_max_abs_error'],
            'avg_num_agents': float(raw.shape[0] / max(
                inputs['scene_ptr'].numel() - 1, 1)),
            'avg_num_edges': float(inputs['edge_index'].shape[1] / max(
                inputs['scene_ptr'].numel() - 1, 1)),
            'avg_degree': float(
                2 * inputs['edge_index'].shape[1] / max(raw.shape[0], 1)),
            'max_component_size': float(
                coupling['component_sizes'].max().cpu()),
            'greedy_JADE': greedy['Aligned_JADE'],
            'hungarian_minus_greedy_JADE': (
                hard['Aligned_JADE'] - greedy['Aligned_JADE']),
        }
        return {
            'scalar': scalar,
            'N_size_buckets': hard['N_size_buckets'],
            'component_size_buckets': hard.get(
                'component_size_buckets', {}),
        }

    @torch.no_grad()
    def multiway_batch_diagnostics(self, auxiliary, inputs, metric_mask):
        """Compute hard/soft V4 diagnostics for one synchronized window."""
        raw = auxiliary['multiway_raw_future_world']
        permutation = auxiliary['multiway_coupling_permutation']
        ground_truth = inputs['world_coord'][self.args.obs_length:].permute(
            1, 0, 2).contiguous()
        aligned = gather_trajectory_bank(raw, permutation)
        preservation = assert_marginal_preservation(
            raw, aligned, ground_truth, agent_mask=metric_mask,
            tolerance=1e-6)
        hard = hard_alignment_diagnostics(
            raw_future=raw,
            ground_truth=ground_truth,
            permutation=permutation,
            agent_mask=metric_mask,
            scene_index=inputs['scene_index'],
            edge_index=auxiliary['multiway_edge_index'],
            lambda_fde=self.args.coupling_fde_weight,
            tolerance=1e-6,
        )
        greedy_permutation = auxiliary.get('multiway_greedy_permutation')
        greedy = None
        if greedy_permutation is not None:
            greedy = hard_alignment_diagnostics(
                raw_future=raw,
                ground_truth=ground_truth,
                permutation=greedy_permutation,
                agent_mask=metric_mask,
                scene_index=inputs['scene_index'],
                lambda_fde=self.args.coupling_fde_weight,
                tolerance=1e-6,
            )
        soft = compute_multiway_coupling_loss(
            permutations=auxiliary['multiway_soft_permutation'],
            pair_scores=None,
            raw_future=raw,
            ground_truth=ground_truth,
            edge_index=auxiliary['multiway_edge_index'],
            agent_mask=metric_mask,
            scene_index=inputs['scene_index'],
            lambda_fde=self.args.coupling_fde_weight,
            tau_joint=self.args.coupling_joint_temperature,
            weight_alignment=1.0,
            weight_pair_score=0.0,
            weight_pair_assignment=0.0,
            weight_no_harm=0.0,
            weight_perm_entropy=0.0,
            weight_relation_prior=0.0,
            weight_gate_reg=0.0,
        )
        scalar = {
            'alignment_oracle': hard['alignment_oracle'],
            'alignment_headroom': hard['alignment_headroom'],
            'alignment_recovery_ratio': hard[
                'alignment_recovery_ratio'],
            'delta_JADE': hard['delta_JADE'],
            'delta_JFDE': hard['delta_JFDE'],
            'soft_hard_gap': (
                soft['loss_alignment'] - hard['aligned_joint_cost']).abs(),
            'marginal_ADE_max_abs_error': preservation[
                'marginal_ADE_max_abs_error'],
            'marginal_FDE_max_abs_error': preservation[
                'marginal_FDE_max_abs_error'],
            'changed_fraction': auxiliary['multiway_changed_fraction'],
            'keep_confidence': auxiliary[
                'multiway_keep_confidence'].mean(),
            'mean_pair_gate': (
                auxiliary['multiway_pair_gate'].mean()
                if auxiliary['multiway_pair_gate'].numel()
                else raw.sum() * 0.0),
            'relation_entropy': (
                -(auxiliary['multiway_relation_posterior'].clamp_min(1e-8) *
                  auxiliary['multiway_relation_posterior'].clamp_min(
                      1e-8).log()).sum(-1).mean()
                if auxiliary['multiway_relation_posterior'].numel()
                else raw.sum() * 0.0),
            'permutation_entropy': self.last_joint_diagnostics[
                'permutation_entropy'],
            'sinkhorn_row_error': self.last_joint_diagnostics[
                'sinkhorn_row_error'],
            'sinkhorn_col_error': self.last_joint_diagnostics[
                'sinkhorn_col_error'],
            'avg_num_agents': float(raw.shape[0]),
            'avg_num_edges': float(
                auxiliary['multiway_edge_index'].shape[1]),
            'avg_degree': float(
                2 * auxiliary['multiway_edge_index'].shape[1] /
                max(raw.shape[0], 1)),
            'max_component_size': float(
                auxiliary['multiway_component_sizes'].max().cpu()),
        }
        if greedy is not None:
            scalar.update({
                'greedy_JADE': greedy['Aligned_JADE'],
                'hungarian_minus_greedy_JADE': (
                    hard['Aligned_JADE'] - greedy['Aligned_JADE']),
            })
        return {
            'scalar': scalar,
            'N_size_buckets': hard['N_size_buckets'],
            'component_size_buckets': hard.get(
                'component_size_buckets', {}),
        }

    def _apply_trajectory_alignment(
            self, predictions, auxiliary, inputs, hard=True):
        """Reorder full trajectories using relation-aware path compatibility."""
        trajectories_world = self._future_predictions_world(
            predictions, inputs)
        alignment = self.trajectory_aligner.align(
            trajectories=trajectories_world,
            last_pos=inputs['obs_traj_world'][:, -1],
            agent_feat=auxiliary['agent_feat'],
            edge_index=auxiliary['edge_index'],
            edge_feat=auxiliary['edge_feat'],
            edge_weight=auxiliary['edge_weight'],
            relation_prob=auxiliary.get('relation_prob'),
            scene_index=inputs['scene_index'],
            hard=hard,
        )
        if hard:
            agent_first = predictions.permute(2, 0, 1, 3)
            permutation = alignment['permutation']
            aligned = agent_first.gather(
                1, permutation[:, :, None, None].expand(
                    -1, -1, predictions.shape[1], 2))
            predictions = aligned.permute(1, 2, 0, 3).contiguous()
        auxiliary['trajectory_alignment_permutation'] = alignment[
            'permutation']
        auxiliary['trajectory_alignment_confidence'] = alignment[
            'keep_confidence']
        auxiliary['trajectory_alignment_edge_gate'] = alignment['edge_gate']
        auxiliary['trajectory_alignment_changed_fraction'] = alignment[
            'changed_fraction']
        self.last_joint_diagnostics.update({
            'alignment_keep_confidence': float(
                alignment['keep_confidence'].mean().detach().cpu()),
            'alignment_changed_fraction': float(
                alignment['changed_fraction'].detach().cpu()),
            'alignment_mean_edge_gate': (
                float(alignment['edge_gate'].mean().detach().cpu())
                if alignment['edge_gate'].numel() else 0.0),
        })
        return predictions, auxiliary

    def forward(self, inputs, if_test=False, apply_trajectory_alignment=True):
        if self.args.trajectory_coupling == 'multiway_v4':
            predictions, auxiliary = self._v4_upstream_predictions(inputs)
            if if_test:
                predictions, auxiliary, _ = self._run_multiway_coupling(
                    predictions, auxiliary, inputs, hard=True)
            return predictions, auxiliary

        x = inputs["x_augmented"].detach().clone()
        num_agents= x.shape[1]

        all_context, all_aux_outputs = self.encode(inputs, if_test=if_test) 

        vy = self.ts_sample(
            all_context=all_context,
            dependency_state=(
                all_aux_outputs.get('dependency_state')
                if self.jdv2_active else None))
        y = torch.zeros([self.args.num_samples, self.args.seq_length, num_agents, 2]).to(self.device)
        y[:, :self.args.obs_length] = x[:self.args.obs_length,:,6:8].repeat(self.args.num_samples, 1, 1, 1)
        # y[:, self.args.obs_length:] = vy.permute(0,2,1,3) # (20, Tp+Tf, B, 2) -> (20, B, Tp+Tf, 2)
        for i in range(self.args.obs_length, self.args.seq_length):
            y[:, i] = y[:, i-1] + vy[:, :, i-self.args.obs_length]

        if (if_test and self.args.trajectory_alignment and
                apply_trajectory_alignment):
            y, all_aux_outputs = self._apply_trajectory_alignment(
                y, all_aux_outputs, inputs, hard=True)
        return y, all_aux_outputs  # (20,Tp+Tf,B,2) 

    def _trajectory_alignment_training_losses(self, inputs, seq_list):
        """Train alignment on the same complete K trajectories used at test."""
        # The generator is a frozen proposal mechanism in alignment stage.  A
        # detached actual diffusion sample avoids back-propagating through the
        # long sampler while eliminating the former goal-proxy mismatch.
        with torch.no_grad():
            predictions, auxiliary = self.forward(
                inputs, if_test=True, apply_trajectory_alignment=False)
            trajectories_world = self._future_predictions_world(
                predictions, inputs).detach()

        ground_truth = inputs['world_coord'][self.args.obs_length:].permute(
            1, 0, 2).contiguous()
        future_mask = seq_list[self.args.obs_length:].permute(
            1, 0).to(device=self.device, dtype=torch.bool)
        details = self.trajectory_aligner.loss(
            trajectories=trajectories_world,
            ground_truth=ground_truth,
            future_mask=future_mask,
            last_pos=inputs['obs_traj_world'][:, -1],
            agent_feat=auxiliary['agent_feat'].detach(),
            edge_index=auxiliary['edge_index'],
            edge_feat=auxiliary['edge_feat'].detach(),
            edge_weight=auxiliary['edge_weight'].detach(),
            relation_prob=(
                auxiliary['relation_prob'].detach()
                if auxiliary.get('relation_prob') is not None else None),
            scene_index=inputs['scene_index'],
            target_temperature=(
                self.args.trajectory_alignment_target_temperature),
            fde_weight=self.args.trajectory_alignment_fde_weight,
        )
        self.last_joint_diagnostics.update({
            'L_align_raw': float(details['raw_joint_surrogate'].cpu()),
            'L_align_soft': float(details['aligned_joint_surrogate'].cpu()),
            'alignment_keep_confidence': float(
                details['mean_keep_confidence'].cpu()),
            'alignment_changed_fraction': float(
                details['alignment_changed_fraction'].cpu()),
            'alignment_mean_edge_gate': float(
                details['alignment_mean_edge_gate'].cpu()),
        })
        configured = {
            'trajectory_alignment_loss': self.args.lambda_trajectory_alignment,
            'trajectory_pair_loss': self.args.lambda_trajectory_pair,
            'alignment_no_harm_loss': self.args.lambda_alignment_no_harm,
            'alignment_entropy_loss': self.args.lambda_alignment_entropy,
        }
        return {
            key: details[key] for key, weight in configured.items()
            if weight > 0
        }

    def _multiway_coupling_training_losses(self, inputs, seq_list):
        """Train V4 on complete trajectories from the configured generator."""
        predictions, auxiliary = self._v4_upstream_predictions(inputs)
        _, auxiliary, coupling = self._run_multiway_coupling(
            predictions, auxiliary, inputs, hard=False)
        ground_truth = inputs['world_coord'][self.args.obs_length:].permute(
            1, 0, 2).contiguous()
        future_mask = seq_list[self.args.obs_length:].permute(
            1, 0).to(device=self.device, dtype=torch.bool)
        agent_mask = seq_list.cumprod(dim=0)[-1].to(
            device=self.device, dtype=torch.bool)
        details = compute_multiway_coupling_loss(
            permutations=coupling['soft_permutation'],
            pair_scores=coupling['pair_score'],
            raw_future=coupling['raw_trajectories'],
            ground_truth=ground_truth,
            edge_index=auxiliary['multiway_edge_index'],
            future_mask=future_mask,
            agent_mask=agent_mask,
            scene_index=inputs['scene_index'],
            relation_posterior=coupling['relation_posterior'],
            relation_prior=coupling['relation_prior'],
            pair_gate=coupling['pair_gate'],
            edge_weight=auxiliary['multiway_edge_weight'],
            lambda_fde=self.args.coupling_fde_weight,
            lambda_rel_geom=self.args.coupling_rel_geom_weight,
            lambda_rel_end=self.args.coupling_rel_endpoint_weight,
            tau_joint=self.args.coupling_joint_temperature,
            tau_pair_target=self.args.coupling_pair_target_temperature,
            tau_pair_pred=self.args.coupling_pair_pred_temperature,
            no_harm_margin=self.args.coupling_no_harm_margin,
            # Weighting remains in trainer.set_losses_coeffs so raw loss
            # components and the final weighted total are both logged.
            weight_alignment=1.0,
            weight_pair_score=1.0,
            weight_pair_assignment=1.0,
            weight_no_harm=1.0,
            weight_perm_entropy=1.0,
            weight_relation_prior=1.0,
            weight_gate_reg=1.0,
            gate_regularization=(
                'l1' if self.args.lambda_gate_reg > 0 else 'none'),
            return_details=True,
        )
        names = (
            'loss_alignment', 'loss_pair_score', 'loss_pair_assignment',
            'loss_no_harm', 'loss_perm_entropy', 'loss_relation_prior',
            'loss_gate_reg')
        losses = {name: details[name] for name in names}
        self.last_joint_diagnostics.update({
            name: float(details[name].detach().cpu()) for name in names
        })
        self.last_joint_diagnostics['loss_total'] = float(sum(
            self.set_losses_coeffs()[name] * losses[name]
            for name in names).detach().cpu())
        if any(not torch.isfinite(value).all() for value in losses.values()):
            raise FloatingPointError(f'Non-finite V4 loss: {losses}')
        return losses

    def _jdv2_pl_from_local(self, unary, target, mode_probability,
                            scene_index, local_energy):
        """Finish a scene-balanced PL reduction after edge-chunk accumulation."""
        _, compact = canonicalize_scene_index(
            scene_index, unary.shape[0], unary.device)
        with torch.autocast(device_type=unary.device.type, enabled=False):
            conditional_log_prob = F.log_softmax(
                unary.float()[:, None, :] - local_energy.float(), dim=-1)
            per_agent_mode = -torch.einsum(
                'nk,nzk->nz', target.float(), conditional_log_prob.float())
            probability = mode_probability.float()
            probability = probability / probability.sum(
                dim=-1, keepdim=True).clamp_min(1e-12)
            per_agent = (per_agent_mode * probability[compact]).sum(dim=-1)
        return scene_balanced_mean(per_agent, compact)

    def _jdv2_no_z_goal_losses(self, inputs):
        """Compute the strict no-z dual composite objective."""
        _, structured = self.encode(inputs, if_test=False, for_loss=True)
        target = build_soft_goal_target(
            structured['goal_candidates_world'], inputs['world_coord'][-1],
            sigma_goal=self.args.goal_soft_sigma,
            candidate_mask=structured['candidate_mask'])
        unary = structured['unary_score']
        scene_index = inputs['scene_index']
        num_agents, num_candidates = unary.shape
        local_prior = unary.new_zeros(
            (num_agents, num_candidates), dtype=torch.float32)
        local_post = torch.zeros_like(local_prior)
        edge_index = structured['edge_index']
        relation_kl_parts = []
        relation_usage_sum = unary.new_zeros(
            (self.args.jdv2_relation_modes,), dtype=torch.float32)
        relation_entropy_sum = 0.0
        relation_usage_count = 0
        energy_sum = 0.0
        energy_square_sum = 0.0
        energy_count = 0
        energy_min = float('inf')
        energy_max = float('-inf')
        chunk_size = self.args.jdv2_edge_chunk_size
        for start in range(0, edge_index.shape[1], chunk_size):
            stop = min(start + chunk_size, edge_index.shape[1])
            chunk_edge = edge_index[:, start:stop]
            chunk_feat = structured['edge_feat'][start:stop]
            chunk_base = structured['base_relation_logits'][start:stop]
            full_relation = self.jdv2_dynamic_relation.full_pair_relation(
                chunk_base, structured['goal_candidates_world'],
                inputs['obs_traj_world'][:, -1], chunk_edge,
                mode_enabled=False)
            if full_relation['log_prob'].ndim != 4:
                raise RuntimeError(
                    'Strict no-z relation must have shape [E,K,K,M]')
            if not self.args.use_dynamic_relation:
                base_log = F.log_softmax(chunk_base.float(), dim=-1)
                prior_relation_log = base_log[:, None, None, :].expand(
                    stop - start, num_candidates, num_candidates,
                    self.args.jdv2_relation_modes)
            else:
                prior_relation_log = full_relation['log_prob']

            if self.args.use_joint_energy:
                factors = self.jdv2_joint_energy.factors(
                    structured['agent_feat'],
                    structured['goal_candidates_world'],
                    inputs['obs_traj_world'][:, -1], chunk_edge, chunk_feat,
                    mode_enabled=False)
                if factors['left_factor'].ndim != 4:
                    raise RuntimeError(
                        'Strict no-z energy factors must be [E,M,K,R]')
                relation_energy = self.jdv2_joint_energy.relation_energy(
                    factors['left_factor'], factors['right_factor'])
                prior_effective = self.jdv2_joint_energy.effective_energy(
                    relation_energy, prior_relation_log)
                if self.args.use_dynamic_relation:
                    teacher_log = structured['relation_teacher']['log_prob'][
                        start:stop]
                    post_relation_log = teacher_log[:, None, None, :].expand_as(
                        relation_energy)
                else:
                    post_relation_log = prior_relation_log
                post_effective = self.jdv2_joint_energy.effective_energy(
                    relation_energy, post_relation_log)
            else:
                prior_effective = unary.new_zeros(
                    (stop - start, num_candidates, num_candidates),
                    dtype=torch.float32)
                post_effective = torch.zeros_like(prior_effective)

            probability = full_relation['prob'].detach().float()
            relation_usage_sum += probability.sum(dim=(0, 1, 2))
            relation_usage_count += probability.numel() // \
                max(self.args.jdv2_relation_modes, 1)
            if probability.numel():
                relation_entropy_sum += float((
                    -(probability.clamp_min(1e-8) *
                      probability.clamp_min(1e-8).log()).sum(-1)
                ).sum().cpu())
            detached_energy = prior_effective.detach().float()
            if detached_energy.numel():
                energy_sum += float(detached_energy.sum().cpu())
                energy_square_sum += float(
                    detached_energy.square().sum().cpu())
                energy_count += detached_energy.numel()
                energy_min = min(energy_min, float(detached_energy.min().cpu()))
                energy_max = max(energy_max, float(detached_energy.max().cpu()))

            src, dst = chunk_edge.long()
            with torch.autocast(
                    device_type=unary.device.type, enabled=False):
                local_prior.index_add_(0, src, torch.einsum(
                    'ekl,el->ek', prior_effective.float(), target[dst].float()))
                local_prior.index_add_(0, dst, torch.einsum(
                    'ekl,ek->el', prior_effective.float(), target[src].float()))
                local_post.index_add_(0, src, torch.einsum(
                    'ekl,el->ek', post_effective.float(), target[dst].float()))
                local_post.index_add_(0, dst, torch.einsum(
                    'ekl,ek->el', post_effective.float(), target[src].float()))
            if self.args.use_dynamic_relation:
                relation_kl_parts.append(jdv2_no_z_relation_kl_per_edge(
                    structured['relation_teacher']['log_prob'][start:stop],
                    full_relation['log_prob'], target, chunk_edge))

        pl_post = jdv2_no_z_pseudo_likelihood_from_local(
            unary, target, scene_index, local_post)
        pl_prior = jdv2_no_z_pseudo_likelihood_from_local(
            unary, target, scene_index, local_prior)
        if relation_kl_parts:
            relation_kl = torch.cat(relation_kl_parts).mean()
        else:
            relation_kl = unary.float().sum() * 0.0
        losses = {
            'jdv2_pl_post': pl_post,
            'jdv2_pl_prior': pl_prior,
            'jdv2_relation_kl': relation_kl,
        }
        coefficients = self.set_losses_coeffs()
        self.last_joint_diagnostics.update({
            'L_PL_post': float(pl_post.detach().cpu()),
            'L_PL_prior': float(pl_prior.detach().cpu()),
            'L_r': float(relation_kl.detach().cpu()),
            'mean_KL_r': float(relation_kl.detach().cpu()),
            'L_no_z': float(sum(
                coefficients[name] * value for name, value in losses.items()
            ).detach().cpu()),
        })
        if relation_usage_count:
            relation_usage = relation_usage_sum / float(relation_usage_count)
            self.last_joint_diagnostics['dynamic_relation_entropy'] = \
                relation_entropy_sum / float(relation_usage_count)
        else:
            relation_usage = relation_usage_sum
            self.last_joint_diagnostics['dynamic_relation_entropy'] = 0.0
        for index, value in enumerate(relation_usage):
            self.last_joint_diagnostics[
                f'predicted_relation_usage_{index}'] = float(value.cpu())
        if energy_count:
            energy_mean = energy_sum / energy_count
            energy_variance = max(
                energy_square_sum / energy_count - energy_mean ** 2, 0.0)
            self.last_joint_diagnostics.update({
                'energy_mean': energy_mean,
                'energy_std': energy_variance ** 0.5,
                'energy_min': energy_min,
                'energy_max': energy_max,
            })
        else:
            self.last_joint_diagnostics.update({
                'energy_mean': 0.0, 'energy_std': 0.0,
                'energy_min': 0.0, 'energy_max': 0.0,
            })
        return losses

    def _jdv2_goal_losses(self, inputs):
        """Compute the V2 marginalized composite objective in edge chunks."""
        if self.strict_no_z:
            return self._jdv2_no_z_goal_losses(inputs)
        _, structured = self.encode(inputs, if_test=False, for_loss=True)
        target = build_soft_goal_target(
            structured['goal_candidates_world'], inputs['world_coord'][-1],
            sigma_goal=self.args.goal_soft_sigma,
            candidate_mask=structured['candidate_mask'])
        unary = structured['unary_score']
        pz = structured['scene_prior']['prob']
        pz_log = structured['scene_prior']['log_prob']
        qz_log = structured['scene_posterior']['log_prob']
        scene_index = inputs['scene_index']
        num_agents, num_candidates = unary.shape
        num_modes = pz.shape[1]
        local_prior = unary.new_zeros(
            (num_agents, num_modes, num_candidates), dtype=torch.float32)
        local_post = torch.zeros_like(local_prior)
        edge_index = structured['edge_index']
        relation_kl_by_mode = []
        relation_kl_edge_scene = []
        relation_usage_sum = unary.new_zeros(
            (self.args.jdv2_relation_modes,), dtype=torch.float32)
        relation_usage_count = 0
        energy_sum = 0.0
        energy_square_sum = 0.0
        energy_count = 0
        energy_min = float('inf')
        energy_max = float('-inf')
        chunk_size = self.args.jdv2_edge_chunk_size
        for start in range(0, edge_index.shape[1], chunk_size):
            stop = min(start + chunk_size, edge_index.shape[1])
            chunk_edge = edge_index[:, start:stop]
            chunk_feat = structured['edge_feat'][start:stop]
            chunk_base = structured['base_relation_logits'][start:stop]
            full_relation = self.jdv2_dynamic_relation.full_pair_relation(
                chunk_base, structured['goal_candidates_world'],
                inputs['obs_traj_world'][:, -1], chunk_edge,
                mode_enabled=self.args.use_scene_latent)
            if not self.args.use_dynamic_relation:
                shape = (stop - start, num_modes, num_candidates,
                         num_candidates, 4)
                base_log = F.log_softmax(chunk_base.float(), dim=-1)
                prior_relation_log = base_log[:, None, None, None].expand(shape)
            else:
                prior_relation_log = full_relation['log_prob']

            if self.args.use_joint_energy:
                factors = self.jdv2_joint_energy.factors(
                    structured['agent_feat'],
                    structured['goal_candidates_world'],
                    inputs['obs_traj_world'][:, -1], chunk_edge, chunk_feat,
                    mode_enabled=self.args.use_scene_latent)
                relation_energy = self.jdv2_joint_energy.relation_energy(
                    factors['left_factor'], factors['right_factor'])
                prior_effective = self.jdv2_joint_energy.effective_energy(
                    relation_energy, prior_relation_log)
                if self.args.use_dynamic_relation:
                    teacher_log = structured['relation_teacher']['log_prob'][
                        start:stop]
                    post_relation_log = teacher_log[
                        :, None, None, None, :].expand_as(relation_energy)
                else:
                    post_relation_log = prior_relation_log
                post_effective = self.jdv2_joint_energy.effective_energy(
                    relation_energy, post_relation_log)
            else:
                prior_effective = unary.new_zeros(
                    (stop - start, num_modes, num_candidates,
                     num_candidates), dtype=torch.float32)
                post_effective = torch.zeros_like(prior_effective)

            # Detached sufficient statistics keep formal diagnostics bounded
            # to O(R) host state and do not alter the Stage-A objective.
            relation_probability = full_relation['prob'].detach().float()
            relation_usage_sum += relation_probability.sum(
                dim=(0, 1, 2, 3))
            relation_usage_count += relation_probability.numel() // \
                max(self.args.jdv2_relation_modes, 1)
            detached_energy = prior_effective.detach().float()
            if detached_energy.numel():
                energy_sum += float(detached_energy.sum().cpu())
                energy_square_sum += float(
                    detached_energy.square().sum().cpu())
                energy_count += detached_energy.numel()
                energy_min = min(
                    energy_min, float(detached_energy.min().cpu()))
                energy_max = max(
                    energy_max, float(detached_energy.max().cpu()))

            src, dst = chunk_edge.long()
            with torch.autocast(
                    device_type=unary.device.type, enabled=False):
                local_prior.index_add_(0, src, torch.einsum(
                    'ezkl,el->ezk', prior_effective.float(),
                    target[dst].float()))
                local_prior.index_add_(0, dst, torch.einsum(
                    'ezkl,ek->ezl', prior_effective.float(),
                    target[src].float()))
                local_post.index_add_(0, src, torch.einsum(
                    'ezkl,el->ezk', post_effective.float(),
                    target[dst].float()))
                local_post.index_add_(0, dst, torch.einsum(
                    'ezkl,ek->ezl', post_effective.float(),
                    target[src].float()))

            if self.args.use_dynamic_relation:
                chunk_kl = jdv2_relation_kl_per_edge_mode(
                    structured['relation_teacher']['log_prob'][start:stop],
                    full_relation['log_prob'], target, chunk_edge)
                relation_kl_by_mode.append(chunk_kl)
                _, compact = canonicalize_scene_index(
                    scene_index, num_agents, unary.device)
                relation_kl_edge_scene.append(compact[chunk_edge[0].long()])

        log_score_prior = jdv2_scene_mode_log_score(
            unary, target, scene_index, local_prior)
        log_score_post = jdv2_scene_mode_log_score(
            unary, target, scene_index, local_post)
        mixture = jdv2_mixture_composite_loss(
            pz_log, log_score_post, log_score_prior)
        gamma = mixture['responsibility']
        gamma_teacher = gamma.detach()
        posterior_distill_per_scene = jdv2_posterior_distillation(
            gamma_teacher, qz_log, reduction='none')
        posterior_distill = posterior_distill_per_scene.mean()

        if relation_kl_by_mode:
            edge_mode_kl = torch.cat(relation_kl_by_mode, dim=0)
            edge_scene = torch.cat(relation_kl_edge_scene, dim=0)
            with torch.autocast(
                    device_type=unary.device.type, enabled=False):
                relation_kl = (
                    edge_mode_kl.float() *
                    gamma_teacher.float()[edge_scene]
                ).sum(dim=-1).mean()
        else:
            relation_kl = unary.float().sum() * 0.0

        losses = {
            'jdv2_mixture_pl': mixture['loss'],
            'jdv2_posterior_distill': posterior_distill,
            'jdv2_relation_kl': relation_kl,
        }
        pl_post_diag = -(gamma_teacher * log_score_post.detach()).sum(
            dim=-1).mean()
        pl_prior_diag = -(gamma_teacher * log_score_prior.detach()).sum(
            dim=-1).mean()
        qz = qz_log.detach().float().exp()
        pz_detached = pz.detach().float()
        gamma_detached = gamma_teacher.float()
        q_p_kl_per_scene = (
            qz * (qz.clamp_min(1e-12).log() -
                  pz_detached.clamp_min(1e-12).log())).sum(dim=-1)
        gamma_q_kl_per_scene = posterior_distill_per_scene.detach().float()
        self.last_joint_diagnostics.update({
            'L_mix': float(mixture['loss'].detach().cpu()),
            'L_q': float(posterior_distill.detach().cpu()),
            'L_PL_post': float(pl_post_diag.cpu()),
            'L_PL_prior': float(pl_prior_diag.cpu()),
            'L_r': float(relation_kl.detach().cpu()),
            'KL_gamma_q': float(gamma_q_kl_per_scene.mean().cpu()),
            'diagnostic_KL_q_p': float(q_p_kl_per_scene.mean().cpu()),
            'q_z_prior_l1': float(
                (qz - pz_detached).abs().mean().cpu()),
        })
        coefficients = self.set_losses_coeffs()
        self.last_joint_diagnostics['L_JG'] = float(sum(
            coefficients[name] * value for name, value in losses.items()
        ).detach().cpu())
        self.last_joint_diagnostics['mean_KL_z'] = float(
            q_p_kl_per_scene.mean().cpu())
        self.last_joint_diagnostics['mean_KL_r'] = float(
            relation_kl.detach().cpu())

        # Read-only latent diagnostics, reported separately for scenes with
        # and without sparse interaction edges.
        _, compact = canonicalize_scene_index(
            scene_index, num_agents, unary.device)
        scene_count = pz.shape[0]
        scene_edge_count = torch.zeros(
            scene_count, dtype=torch.long, device=unary.device)
        if edge_index.shape[1]:
            scene_edge_count.index_add_(
                0, compact[edge_index[0].long()],
                torch.ones(edge_index.shape[1], dtype=torch.long,
                           device=unary.device))
        p_entropy = -(pz_detached * pz_detached.clamp_min(1e-12).log()).sum(-1)
        q_entropy = -(qz * qz.clamp_min(1e-12).log()).sum(-1)
        gamma_entropy = -(gamma_detached *
                          gamma_detached.clamp_min(1e-12).log()).sum(-1)
        score_variance = mixture['combined_log_score'].detach().float().var(
            dim=-1, unbiased=False)
        mean_abs_q_p = (qz - pz_detached).abs().mean(dim=-1)

        # Reuse the future encoder for a no-gradient within-scene compatible
        # shuffle. This changes only diagnostics, never the optimized loss.
        with torch.no_grad():
            future = inputs['world_coord'][self.args.obs_length:].permute(
                1, 0, 2).contiguous().float()
            shuffled = future.clone()
            for scene_id in torch.unique(scene_index, sorted=True):
                indices = torch.nonzero(
                    scene_index.eq(scene_id), as_tuple=False).flatten()
                if indices.numel() > 1:
                    shuffled[indices] = future[indices.roll(1)]
            anchor = inputs['obs_traj_world'][:, -1].float()
            shuffled_velocity = torch.diff(
                torch.cat((anchor[:, None], shuffled), dim=1), dim=1
            ) / float(self.args.trajectory_dt)
            if self.args.use_scene_latent:
                shuffled_q = self.jdv2_future_teacher.scene_posterior(
                    shuffled, shuffled_velocity, anchor,
                    structured['scene_prior']['logits'], scene_index)['prob']
                future_shuffle_l1 = (
                    shuffled_q.float() - qz).abs().sum(dim=-1)
            else:
                future_shuffle_l1 = qz.new_zeros((scene_count,))

        for label, mask in (
                ('e_gt0', scene_edge_count > 0),
                ('e_eq0', scene_edge_count == 0)):
            count = int(mask.sum().item())
            self.last_joint_diagnostics[f'scene_count_{label}'] = float(count)
            if count == 0:
                continue
            diagnostics = {
                f'L_mix_{label}': mixture['per_scene_nll'].detach()[mask].mean(),
                f'L_q_{label}': posterior_distill_per_scene.detach()[mask].mean(),
                f'p_entropy_{label}': p_entropy[mask].mean(),
                f'q_entropy_{label}': q_entropy[mask].mean(),
                f'gamma_entropy_{label}': gamma_entropy[mask].mean(),
                f'between_z_log_score_variance_{label}':
                    score_variance[mask].mean(),
                f'KL_gamma_q_{label}': gamma_q_kl_per_scene[mask].mean(),
                f'diagnostic_KL_q_p_{label}': q_p_kl_per_scene[mask].mean(),
                f'mean_abs_q_p_{label}': mean_abs_q_p[mask].mean(),
                f'future_shuffle_l1_{label}': future_shuffle_l1[mask].mean(),
            }
            for name, value in diagnostics.items():
                self.last_joint_diagnostics[name] = float(value.cpu())
            for name, probability in (
                    ('p', pz_detached), ('q', qz),
                    ('gamma', gamma_detached)):
                usage = probability[mask].mean(dim=0)
                hard = F.one_hot(
                    probability[mask].argmax(dim=-1),
                    num_classes=num_modes).float().mean(dim=0)
                for index in range(num_modes):
                    self.last_joint_diagnostics[
                        f'{name}_usage_{index}_{label}'] = float(
                            usage[index].cpu())
                    self.last_joint_diagnostics[
                        f'{name}_hard_usage_{index}_{label}'] = float(
                            hard[index].cpu())
        for index, value in enumerate(gamma_detached.mean(dim=0)):
            self.last_joint_diagnostics[f'gamma_z_mean_{index}'] = float(
                value.cpu())
        if relation_usage_count:
            relation_usage = relation_usage_sum / float(relation_usage_count)
        else:
            relation_usage = relation_usage_sum
        for index, value in enumerate(relation_usage):
            self.last_joint_diagnostics[
                f'predicted_relation_usage_{index}'] = float(value.cpu())
        if energy_count:
            energy_mean = energy_sum / energy_count
            energy_variance = max(
                energy_square_sum / energy_count - energy_mean ** 2, 0.0)
            self.last_joint_diagnostics.update({
                'energy_mean': energy_mean,
                'energy_std': energy_variance ** 0.5,
                'energy_min': energy_min,
                'energy_max': energy_max,
            })
        else:
            self.last_joint_diagnostics.update({
                'energy_mean': 0.0, 'energy_std': 0.0,
                'energy_min': 0.0, 'energy_max': 0.0,
            })
        return losses

    def _jdv2_noisy_velocity_world(self, noisy_velocity_map, inputs):
        """Convert only the corrector geometry adapter to world m/s."""
        last_map = inputs['x_augmented'][
            self.args.obs_length - 1, :, 6:8]
        position_map = last_map[:, None] + torch.cumsum(
            noisy_velocity_map, dim=1)
        position_world = inputs['scene'].make_world_coord_torch(
            position_map * float(self.args.down_factor))
        anchored = torch.cat((
            inputs['obs_traj_world'][:, -1:].float(),
            position_world.float()), dim=1)
        velocity_world = torch.diff(anchored, dim=1) / float(
            self.args.trajectory_dt)
        return velocity_world, position_world

    def set_jdv2_training_sampling_context(
            self, seed: int, epoch: int, batch_index: int,
            batches_per_epoch: int) -> int:
        """Set a stable, resume-safe Stage-A sampler context for Stage B."""
        if epoch < 1 or batch_index < 0 or batches_per_epoch < 1:
            raise ValueError("invalid Stage-B training batch coordinates")
        serial = (int(epoch) - 1) * int(batches_per_epoch) + int(batch_index)
        self.jdv2_sampler.set_sampling_context(int(seed), serial)
        return serial

    def _jdv2_corrector_zero(self):
        """Return an exact differentiable zero connected to every corrector."""
        zero = next(self.jdv2_corrector.parameters()).new_zeros(())
        for parameter in self.jdv2_corrector.parameters():
            zero = zero + parameter.sum() * 0.0
        return zero

    def _jdv2_dependency_losses(self, inputs, t=None):
        """Train V1 on one frozen Stage-A oracle joint world per scene."""
        if self.args.training_stage != 'joint_trajectory':
            raise RuntimeError(
                'Stage-B V1 is defined only for joint_trajectory')
        with torch.no_grad():
            x = inputs['x_augmented']
            num_agents = x.shape[1]
            image = inputs['tensor_image'].unsqueeze(0).repeat(
                num_agents, 1, 1, 1)
            maps = inputs['input_traj_maps'][:, :self.args.obs_length]
            goal_logits = self.goal_module(torch.cat((image, maps), dim=1))
            goal_prob = torch.sigmoid(goal_logits[:, -1:])
            structured = self._jdv2_goal_outputs(
                inputs, goal_prob, sample=True, include_teacher=False)
            sampled = structured['sampled']
            selection = jdv2_select_scene_oracle_branch(
                sampled['goals'], inputs['world_coord'][-1],
                inputs['scene_index'], structured['edge_index'],
                sampled['relation_embedding'])
            selected_goal = selection['selected_goal']
            relation_embedding = selection['selected_relation_embedding']
            selected_goal_map = self._map_goals_to_map(
                selected_goal, inputs['scene'])
            context = self._jdv2_contexts(inputs, selected_goal_map)[0]

        x = inputs['x_augmented']
        velocity_target = x[self.args.obs_length:, :, 2:4].permute(
            1, 0, 2).contiguous()
        batch_size = velocity_target.shape[0]
        active_timesteps = jdv2_active_corrector_timesteps(
            self.var_sched.num_steps, self.args.ddim_step,
            self.args.branch_stage_step)
        timestep = jdv2_scene_timesteps(
            inputs['scene_index'], active_timesteps, supplied_t=t)
        edge_index = structured['edge_index']
        if edge_index.shape[1]:
            src, dst = edge_index.long()
            if not torch.equal(timestep[src], timestep[dst]):
                raise RuntimeError('edge endpoints have inconsistent timestep')
        alpha_bar = self.var_sched.alpha_bars[timestep]
        beta = self.var_sched.betas[timestep]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)
        noise = torch.randn_like(velocity_target)
        noisy_velocity = c0 * velocity_target + c1 * noise
        with torch.no_grad():
            epsilon_base = self.diffnet(
                noisy_velocity, beta=beta, context=context)
        noisy_world, _ = self._jdv2_noisy_velocity_world(
            noisy_velocity, inputs)
        delta = self.jdv2_corrector(
            noisy_world, inputs['obs_traj_world'][:, -1],
            edge_index, relation_embedding, timestep,
            edge_weight=structured['edge_weight'])
        epsilon_joint = epsilon_base + delta
        degree = torch.zeros(
            batch_size, dtype=torch.long, device=velocity_target.device)
        if edge_index.shape[1]:
            ones = torch.ones(
                edge_index.shape[1], dtype=torch.long,
                device=velocity_target.device)
            degree.index_add_(0, edge_index[0].long(), ones)
            degree.index_add_(0, edge_index[1].long(), ones)
        active_agent = degree > 0
        if active_agent.any():
            diffusion_loss = F.mse_loss(
                epsilon_joint[active_agent].float(),
                noise[active_agent].float(), reduction='mean')
        else:
            diffusion_loss = self._jdv2_corrector_zero()
        predicted_clean = (
            noisy_velocity.float() - c1.float() * epsilon_joint.float()) / \
            c0.float().clamp_min(1e-8)
        _, predicted_world = self._jdv2_noisy_velocity_world(
            predicted_clean, inputs)
        target_world = inputs['world_coord'][
            self.args.obs_length:].permute(1, 0, 2).contiguous()
        relative_loss = (sparse_relative_motion_loss(
            predicted_world, target_world, edge_index)
            if edge_index.shape[1] else self._jdv2_corrector_zero())
        oracle_error = selection['scene_error'].gather(
            1, selection['scene_branch'][:, None]).mean()
        timestep_histogram = torch.bincount(
            timestep, minlength=self.var_sched.num_steps + 1)
        self.last_joint_diagnostics.update({
            'trainable_parameter_count': float(sum(
                parameter.numel()
                for parameter in self.jdv2_corrector.parameters()
                if parameter.requires_grad)),
            'oracle_branch_goal_error': float(oracle_error.cpu()),
            'degree_positive_agent_count': float(active_agent.sum().item()),
            'e0_skipped_count': float(edge_index.shape[1] == 0),
            'delta_epsilon_rms': float(
                delta.detach().float().square().mean().sqrt().cpu()),
            'epsilon_base_rms': float(
                epsilon_base.detach().float().square().mean().sqrt().cpu()),
            'delta_base_rms_ratio': float((
                delta.detach().float().square().mean().sqrt() /
                epsilon_base.detach().float().square().mean().sqrt().clamp_min(
                    1e-12)).cpu()),
            'L_diff': float(diffusion_loss.detach().cpu()),
            'L_relative': float(relative_loss.detach().cpu()),
        })
        for active_timestep in active_timesteps:
            self.last_joint_diagnostics[
                f'active_timestep_count_{active_timestep}'] = float(
                    timestep_histogram[active_timestep].item())
        return {'jdv2_diffusion_loss': diffusion_loss,
                'jdv2_relative_loss': relative_loss}


    def _base_loss_components(self, inputs, seq_list, t=None, if_test=False):
        """Compute the two unchanged GDTS objectives and return auxiliaries."""
        all_context, all_aux_outputs = self.encode(
            inputs, if_test=if_test,
            for_loss=self.active_goal_model_type != 'independent')

        # compute goal BCE loss
        loss_mask = self.compute_loss_mask(seq_list, self.args.obs_length).to(self.device)
        out_maps_GT_goal = inputs["input_traj_maps"][:, self.args.obs_length:]
        goal_logit_map = all_aux_outputs["goal_logit_map"]
        goal_BCE_loss = Goal_BCE_loss(goal_logit_map, out_maps_GT_goal, loss_mask)

        # compute diffusion loss
        x = inputs["x_augmented"].detach().clone()
        vx_gt = x[self.args.obs_length:,:,2:4] # predict the velocity, (Tf, B, 2) 
        vx_gt = vx_gt.permute(1,0,2) # (B, Tf, 2)
        batch_size, _, point_dim = vx_gt.size() # (B, Tf, 2)
        if t is None:
            t = self.var_sched.uniform_sample_t(batch_size)

        timestep = torch.as_tensor(t, dtype=torch.long, device=vx_gt.device)
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        if timestep.shape != (batch_size,):
            raise ValueError('Diffusion timestep must be scalar or [N].')
        alpha_bar = self.var_sched.alpha_bars[timestep]
        beta = self.var_sched.betas[timestep]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # [N,1,1]
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # [N,1,1]

        e_rand = torch.randn_like(vx_gt)  # [N,T_pred,2]
        context = all_context[0]
        e_theta = self.diffnet(c0 * vx_gt + c1 * e_rand, beta=beta, context=context) 

        diffusion_loss = F.mse_loss(e_theta.reshape(-1, point_dim), e_rand.reshape(-1, point_dim), reduction='mean')
        
        losses = {
            "diffusion_loss": diffusion_loss,
            "goal_BCE_loss": goal_BCE_loss,
        }

        return losses, all_aux_outputs

    def get_loss(self, inputs, seq_list, t=None, if_test=False):
        """Return baseline plus ablation-specific structured goal losses."""
        if self.jdv2_active:
            losses = {}
            if self.args.training_stage in {'joint_goal', 'joint_finetune'}:
                losses.update(self._jdv2_goal_losses(inputs))
            if self.args.training_stage in {
                    'joint_trajectory', 'joint_finetune'} and \
                    self.args.use_dependency_corrector:
                losses.update(self._jdv2_dependency_losses(inputs, t=t))
            if any(not torch.isfinite(value).all()
                   for value in losses.values()):
                bad = {
                    name: (tuple(value.shape), str(value.dtype))
                    for name, value in losses.items()
                    if not torch.isfinite(value).all()}
                raise FloatingPointError(
                    f'Non-finite JDV2 loss tensors: {bad}')
            return losses
        if self.args.training_stage == 'multiway_coupling':
            return self._multiway_coupling_training_losses(inputs, seq_list)
        if self.args.training_stage == 'alignment':
            losses = self._trajectory_alignment_training_losses(
                inputs, seq_list)
            if any(not torch.isfinite(value).all()
                   for value in losses.values()):
                raise FloatingPointError(
                    f'Non-finite trajectory-alignment loss: {losses}')
            return losses
        losses, auxiliary = self._base_loss_components(
            inputs, seq_list, t=t, if_test=if_test)
        model_type = self.active_goal_model_type
        if model_type == 'independent':
            return losses

        # All endpoint terms use world metres, matching graph features and
        # goal_soft_sigma. q_i(k) is a normalized soft GT assignment [N,K].
        soft_target = build_soft_goal_target(
            auxiliary['goal_candidates_world'],
            inputs['world_coord'][-1],
            sigma_goal=self.args.goal_soft_sigma,
            candidate_mask=auxiliary['candidate_mask'])
        mode_log_prob = auxiliary['mode_log_prob']
        conditional_log_prob = auxiliary['conditional_goal_log_prob']
        scene_index = inputs['scene_index']

        if model_type in {'social', 'lowrank', 'joint'}:
            losses['mode_loss'] = low_rank_social_mode_loss(
                mode_log_prob, conditional_log_prob, soft_target,
                scene_index=scene_index,
                normalize_by_num_agents=(
                    self.args.normalize_mode_loss_by_agents))

        if model_type in {'energy', 'joint'}:
            energy_output = auxiliary['energy_output']
            if energy_output is None or 'effective_energy' not in energy_output:
                raise RuntimeError(
                    'Pairwise pseudo-likelihood requires sparse effective energy.')
            unary_log_prob = marginal_goal_log_prob(
                mode_log_prob, conditional_log_prob,
                scene_index=scene_index)
            losses['pseudo_likelihood_loss'] = pseudo_likelihood_loss(
                unary_log_prob=unary_log_prob,
                soft_goal_target=soft_target,
                edge_index=auxiliary['edge_index'],
                effective_pair_energy=energy_output['effective_energy'],
                energy_weight=self.args.energy_weight,
                neighbor_mode=self.args.pairwise_neighbor_target,
                scene_index=scene_index,
                edge_weight=auxiliary['edge_weight'],
                normalize_by_degree=(
                    self.args.pair_energy_normalization == 'mean'))

            relation_prob = auxiliary['relation_prob']
            # Entropy penalization encourages identifiable per-edge modes;
            # batch-level KL prevents every edge from collapsing to one mode.
            losses['relation_entropy_loss'] = self._categorical_entropy(
                relation_prob)
            losses['relation_balance_loss'] = self._usage_balance_loss(
                relation_prob, family='relation')

        if model_type in {'lowrank', 'joint'}:
            losses['mode_balance_loss'] = self._usage_balance_loss(
                auxiliary['mode_prob'], family='mode')

        if (model_type in {'lowrank', 'joint'} and
                self.args.lambda_joint_rank > 0):
            effective_pair_energy = None
            if model_type == 'joint':
                effective_pair_energy = auxiliary[
                    'energy_output']['effective_energy']
            losses['joint_rank_loss'] = scene_joint_ranking_loss(
                mode_log_prob=mode_log_prob,
                conditional_goal_log_prob=conditional_log_prob,
                goal_candidates=auxiliary['goal_candidates_world'],
                ground_truth_goal=inputs['world_coord'][-1],
                scene_index=scene_index,
                edge_index=(auxiliary['edge_index']
                            if model_type == 'joint' else None),
                effective_pair_energy=effective_pair_energy,
                candidate_mask=auxiliary['candidate_mask'],
                energy_weight=self.args.energy_weight,
                target_temperature=(
                    self.args.joint_rank_target_temperature),
                score_temperature=self.args.joint_rank_score_temperature,
                normalize_pair_energy=(
                    self.args.pair_energy_normalization == 'mean'),
            )

        if self.args.adaptive_graph and self.args.lambda_graph_density > 0:
            gate_prob = auxiliary['graph_gate_prob']
            if gate_prob is None or gate_prob.numel() == 0:
                losses['graph_density_loss'] = \
                    auxiliary['agent_feat'].sum() * 0.0
            else:
                target_density = self.args.adaptive_graph_target_density
                losses['graph_density_loss'] = (
                    gate_prob.mean() - target_density).square()

        if self.args.continuous_refinement and (
                self.args.lambda_continuous_refinement > 0 or
                self.args.lambda_refinement_delta > 0):
            refinement_terms = best_joint_continuous_refinement_loss(
                anchor_goal=auxiliary['refinement_anchor_goal'],
                refined_goal=auxiliary['refinement_output_goal'],
                ground_truth_goal=inputs['world_coord'][-1],
                scene_index=scene_index,
            )
            if self.args.lambda_continuous_refinement > 0:
                losses['continuous_refinement_loss'] = refinement_terms[
                    'continuous_refinement_loss']
            if self.args.lambda_refinement_delta > 0:
                losses['refinement_delta_loss'] = refinement_terms[
                    'refinement_delta_loss']

        if self.args.trajectory_alignment:
            losses.update(self._trajectory_alignment_training_losses(
                inputs, seq_list))

        if hasattr(self, 'last_joint_diagnostics'):
            named = {
                'L_goal': losses['goal_BCE_loss'],
                'L_diff': losses['diffusion_loss'],
                'L_mode': losses.get('mode_loss'),
                'L_PL': losses.get('pseudo_likelihood_loss'),
                'L_rank': losses.get('joint_rank_loss'),
                'L_graph': losses.get('graph_density_loss'),
                'L_refine': losses.get('continuous_refinement_loss'),
                'L_refine_delta': losses.get('refinement_delta_loss'),
                'L_align': losses.get('trajectory_alignment_loss'),
                'L_align_pair': losses.get('trajectory_pair_loss'),
                'L_align_guard': losses.get('alignment_no_harm_loss'),
                'L_align_entropy': losses.get('alignment_entropy_loss'),
            }
            self.last_joint_diagnostics.update({
                key: float(value.detach().cpu())
                for key, value in named.items() if value is not None
            })
        if any(not torch.isfinite(value).all() for value in losses.values()):
            raise FloatingPointError(f'Non-finite loss component: {losses}')
        return losses

    def sample(self, all_context):
        context = all_context[-1]
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, self.args.pred_length, 2]).to(context.device) 
        traj = {self.var_sched.num_steps: x_T} 
        all_outputs = []
        for t in range(self.var_sched.num_steps, self.determined_step, -1): # from x_T to 0
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            # print(t)
            # print(alpha_bar)
            sigma = self.var_sched.get_sigmas(t)

            c0 = 1.0 / torch.sqrt(alpha) # scalar
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar) # scalar
            x_t = traj[t]# result from previous denoising
            beta = self.var_sched.betas[[t]*batch_size]
            # print(beta)
            e_theta = self.diffnet(x_t, beta=beta, context=context) # predict noise, output: B * 12 * 2
            x_next = c0 * (x_t - c1 * e_theta) #+ sigma * z# compute the denoised x from x_t, whose dimension is [batch_size, num_points, point_dim]
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            del traj[t]
        middle_result = traj[self.determined_step]
        for sample_idx in range(self.args.num_samples):
            context = all_context[sample_idx]
            traj[self.determined_step] = middle_result
            for t in range(self.determined_step, 0, -1): # from x_T to 0
                z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                sigma = self.var_sched.get_sigmas(t)

                c0 = 1.0 / torch.sqrt(alpha) # scalar
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar) # scalar
                x_t = traj[t]# result from previous denoising
                beta = self.var_sched.betas[[t]*batch_size]
                e_theta = self.diffnet(x_t, beta=beta, context=context) # predict noise, output: B * 12 * 2
                x_next = c0 * (x_t - c1 * e_theta) + sigma * z # compute the denoised x from x_t, whose dimension is [batch_size, num_points, point_dim]
                traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
                traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
                del traj[t]
            all_outputs.append(traj[0])
            # del traj
        all_outputs = torch.stack(all_outputs)
        # all_intermediate_traj_outputs = traj # (diffusion_step, B, 12, 2), only the last sample
        return all_outputs # (B, 12, 2)

    def ddim_sample(self, all_context):
        ddim_step = int(20)
        context = all_context[-1]
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, self.args.pred_length, 2]).to(context.device) # generate guassian noise as initial, when t=T timestep
        ts = np.linspace(self.var_sched.num_steps, 0, (ddim_step + 1)) # (21,)
        all_outputs = []
        simple_var = False
        eta = 1
        if simple_var:
            eta = 1
        for mode_idx in range(self.args.num_samples):
            context = all_context[mode_idx]
            x_t = x_T
            for i in range(1,ddim_step + 1):
                cur_t = int(ts[i - 1]) #- 1 
                prev_t = int(ts[i]) #- 1 
                ab_cur = self.var_sched.alpha_bars[cur_t]
                ab_prev = self.var_sched.alpha_bars[prev_t] if prev_t >= 0 else 1
                beta = self.var_sched.betas[[cur_t]*batch_size]
                eps = self.diffnet(x_t, beta=beta, context=context)
                var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
                noise = torch.randn_like(x_t)

                first_term = (ab_prev / ab_cur)**0.5 * x_t
                second_term = ((1 - ab_prev - var)**0.5 -
                                (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
                if simple_var:
                    third_term = (1 - ab_cur / ab_prev)**0.5 * noise
                else:
                    third_term = var**0.5 * noise
                x_t = first_term + second_term + third_term
            all_outputs.append(x_t)
            # del traj
        all_outputs = torch.stack(all_outputs)
        return all_outputs # (batch_size, num_points, point_dim) = (B, 12, 2)

    def ddpm_sample(self, all_context):
        context = all_context[-1]
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, self.args.pred_length, 2]).to(context.device) 
        simple_var = False
        all_outputs = []
        for mode_idx in range(self.args.num_samples):
            context = all_context[mode_idx]
            traj = {self.var_sched.num_steps: x_T} 
            x_t = x_T
            for t in range(self.var_sched.num_steps , 0, -1):
                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                # print([t]*batch_size)
                beta = self.var_sched.betas[[t]*batch_size]
                # beta = self.var_sched.betas[t]
                # print(beta)
                # beta = torch.full((batch_size,), beta, device=context.device, dtype=torch.long)
                if t == 0:
                    noise = 0
                else:
                    if simple_var:
                        var = self.var_sched.betas[t]
                    else:
                        var = (1 - self.var_sched.alpha_bars[t - 1]) / (
                            1 - self.var_sched.alpha_bars[t]) * self.var_sched.betas[t]
                    noise = torch.randn_like(x_t)
                    noise *= torch.sqrt(var)
                x_t = traj[t]
                eps = self.diffnet(x_t, beta=beta, context=context)

                x_next = (x_t -
                        (1 - alpha) / torch.sqrt(1 - alpha_bar) *
                        eps) / torch.sqrt(alpha) + noise
                traj[t-1] = x_next.detach()
                traj[t] = traj[t].cpu()
                del traj[t]
            all_outputs.append(traj[0])
        all_outputs = torch.stack(all_outputs)
        return all_outputs

    def ddim_ts_sample(self, all_context):
        context = all_context[-1]
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, self.args.pred_length, 2]).to(context.device) 
        # print(x_T.shape)
        all_outputs = []
        ddim_step = int(20)
        determined_step = int(8)
        ts = np.linspace(self.var_sched.num_steps, 0, (ddim_step + 1)) # (21,)
        simple_var = False
        eta = 0
        if simple_var:
            eta = 1
        x_t = x_T
        for i in range(1,determined_step+1):
            cur_t = int(ts[i - 1]) #- 1 
            prev_t = int(ts[i]) #- 1 
            ab_cur = self.var_sched.alpha_bars[cur_t]
            ab_prev = self.var_sched.alpha_bars[prev_t] if prev_t >= 0 else 1
            beta = self.var_sched.betas[[cur_t]*batch_size]
            eps = self.diffnet(x_t, beta=beta, context=context)
            var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
            noise = torch.randn_like(x_t)

            first_term = (ab_prev / ab_cur)**0.5 * x_t
            second_term = ((1 - ab_prev - var)**0.5 -
                            (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
            if simple_var:
                third_term = (1 - ab_cur / ab_prev)**0.5 * noise
            else:
                third_term = var**0.5 * noise
            x_t = first_term + second_term + third_term
        middle_result = x_t
        for sample_idx in range(self.args.num_samples):
            context = all_context[sample_idx]
            x_t = middle_result
            for i in range(determined_step+1,ddim_step+1):
                cur_t = int(ts[i - 1]) #- 1 
                prev_t = int(ts[i]) #- 1 
                ab_cur = self.var_sched.alpha_bars[cur_t]
                ab_prev = self.var_sched.alpha_bars[prev_t] if prev_t >= 0 else 1
                beta = self.var_sched.betas[[cur_t]*batch_size]
                eps = self.diffnet(x_t, beta=beta, context=context)
                var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
                noise = torch.randn_like(x_t)

                first_term = (ab_prev / ab_cur)**0.5 * x_t
                second_term = ((1 - ab_prev - var)**0.5 -
                                (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
                if simple_var:
                    third_term = (1 - ab_cur / ab_prev)**0.5 * noise
                else:
                    third_term = var**0.5 * noise
                x_t = first_term + second_term + third_term
            all_outputs.append(x_t)
            # del traj
        all_outputs = torch.stack(all_outputs)
        return all_outputs # (B, 12, 2)
    
    def ts_sample(self, all_context, dependency_state=None):
        all_outputs = []

        # trunk stage 
        context = all_context[-1]
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, self.args.pred_length, 2]).to(context.device) 
        x_t = x_T
        for t in range(self.var_sched.num_steps, self.args.trunk_stage_step, -1):
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]

            c0 = 1.0 / torch.sqrt(alpha) # scalar
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar) # scalar
            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.diffnet(x_t, beta=beta, context=context) 
            x_next = c0 * (x_t - c1 * e_theta) 
            x_t = x_next     
        
        # branch stage
        middle_result = x_t
        simple_var = True if self.args.dataset in ['eth5', 'ind'] else False
        eta = 0
        if simple_var:
            eta = 1
        ts = np.linspace(self.var_sched.num_steps, 0, (self.args.ddim_step + 1)) # (21,)
        for sample_idx in range(self.args.num_samples):
            context = all_context[sample_idx]
            x_t = middle_result
            
            for i in range(int(self.args.branch_stage_step + 1), int(self.args.ddim_step + 1)):
                cur_t = int(ts[i - 1]) #- 1 
                prev_t = int(ts[i]) #- 1 
                ab_cur = self.var_sched.alpha_bars[cur_t]
                ab_prev = self.var_sched.alpha_bars[prev_t] if prev_t >= 0 else 1
                beta = self.var_sched.betas[[cur_t] * batch_size]
                eps = self.diffnet(x_t, beta=beta, context=context)
                if (dependency_state is not None and
                        self.args.use_dependency_corrector):
                    last_map = dependency_state['last_position_map']
                    position_map = last_map[:, None] + torch.cumsum(x_t, dim=1)
                    position_world = dependency_state[
                        'scene'].make_world_coord_torch(
                            position_map * float(self.args.down_factor))
                    anchor = dependency_state['last_position_world'][:, None]
                    noisy_world_velocity = torch.diff(
                        torch.cat((anchor.float(), position_world.float()),
                                  dim=1), dim=1) / float(
                                      self.args.trajectory_dt)
                    relation_embedding = dependency_state[
                        'relation_embedding'][:, sample_idx]
                    delta = self.jdv2_corrector(
                        noisy_world_velocity,
                        dependency_state['last_position_world'],
                        dependency_state['edge_index'], relation_embedding,
                        torch.tensor(cur_t, device=x_t.device),
                        edge_weight=dependency_state['edge_weight'])
                    eps = eps + delta
                var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
                noise = torch.randn_like(x_t)

                first_term = (ab_prev / ab_cur)**0.5 * x_t
                second_term = ((1 - ab_prev - var)**0.5 -
                                (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
                if simple_var:
                    third_term = (1 - ab_cur / ab_prev)**0.5 * noise
                else:
                    third_term = var**0.5 * noise
                x_t = first_term + second_term + third_term

            all_outputs.append(x_t) 
        all_outputs = torch.stack(all_outputs)
        # all_intermediate_traj_outputs = traj # (diffusion_step, B, 12, 2), only the last sample
        return all_outputs # (B, 12, 2)
