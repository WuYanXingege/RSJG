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
)


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
        self.active_goal_model_type = self.args.goal_model_type
        if (self.args.goal_model_type != 'independent' and
                self.args.training_stage == 'baseline'):
            self.active_goal_model_type = 'independent'

        if self.args.goal_model_type != 'independent':
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

    def _configure_training_stage(self, epoch=1):
        """Apply the staged-training freeze policy without changing modules.

        ``joint`` freezes the original U-Net, history encoder and diffusion
        denoiser. ``finetune`` trains every component. ``baseline`` leaves the
        GDTS components trainable and freezes the newly added modules; the
        exact historical baseline is still obtained with
        ``goal_model_type=independent``.
        """
        stage = self.args.training_stage
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
        selected_inputs = {}
        for key, value in batch_data.items():
            if torch.is_tensor(value):
                value = value.squeeze(0).to(self.device)
                value = value.long() if key in integer_inputs else value.float()
            selected_inputs[key] = value
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
        if self.args.goal_model_type != 'independent':
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
        if self.args.goal_model_type != 'independent':
            test_metrics.update({
                "JADE": [],
                "JFDE": [],
                "Goal_minFDE": [],
                "Joint_Goal_Endpoint_Error": [],
                "Joint_Goal_Compatibility": [],
            })
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
        if self.args.goal_model_type != 'independent':
            best_metrics.update({"JADE": 1e9, "JFDE": 1e9})
        return best_metrics

    def best_valid_metric(self):
        if self.args.best_metric != 'auto':
            return self.args.best_metric
        # Legacy GDTS retains its historical pixel ADE selection. Structured
        # experiments default to a coherent, world-coordinate scene metric.
        return ('ADE' if self.args.goal_model_type == 'independent'
                else 'JADE')

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

        vy = self.ts_sample(all_context=all_context) # [20, B, 1, 512] -> [20, B, Tf, 2]
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
    
    def ts_sample(self, all_context):
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
