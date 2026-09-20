import os
import time
import datetime
import json
from contextlib import nullcontext
import torch
import numpy as np
from tqdm import tqdm
try:
    import wandb
except ImportError:  # WandB is optional when --use_wandb False.
    wandb = None
from src.data_loader import get_dataloader
from src.metrics import compute_metric_mask
from src.utils import (
    add_dict_prefix, formatted_time, isolated_random_seed,
    print_model_summary,
)
from src.models.model import GDTS   
from src.trajectory_bank_cache import (
    get_trajectory_bank_dataloader,
    pack_statistics,
)
from src.joint_dependency_v2_cache import load_cache_record


V4_DIAGNOSTIC_NAMES = (
    'alignment_oracle', 'alignment_headroom',
    'alignment_recovery_ratio', 'delta_JADE', 'delta_JFDE',
    'changed_fraction', 'keep_confidence', 'mean_pair_gate',
    'relation_entropy', 'permutation_entropy', 'sinkhorn_row_error',
    'sinkhorn_col_error', 'soft_hard_gap',
    'marginal_ADE_max_abs_error', 'marginal_FDE_max_abs_error',
    'avg_num_agents', 'avg_num_edges', 'avg_degree',
    'max_component_size', 'greedy_JADE',
    'hungarian_minus_greedy_JADE',
)

class trainer(object):
    def __init__(self, args):
        self.args = args
        # initialize data loaders
        self.data_loaders = dict()
        loader_factory = (get_trajectory_bank_dataloader
                          if args.use_trajectory_bank_cache else
                          get_dataloader)
        self.data_loaders['train'] = loader_factory(args, split='train') \
            if args.use_trajectory_bank_cache else \
            loader_factory(args, set_name='train')
        self.data_loaders['valid'] = loader_factory(args, split='valid') \
            if args.use_trajectory_bank_cache else \
            loader_factory(args, set_name='valid')
        self.data_loaders['test'] = loader_factory(args, split='test') \
            if args.use_trajectory_bank_cache else \
            loader_factory(args, set_name='test')
        if (getattr(args, 'goal_model_type', None) == 'joint_dependency_v2'
                and getattr(args, 'jdv2_active', False)):
            manifest = getattr(
                self.data_loaders['train'].dataset, 'jdv2_manifest', None)
            if manifest is None:
                raise RuntimeError('Active JDV2 loader has no validated cache')
            args.jdv2_cache_manifest_hash = manifest['manifest_hash']
            args.jdv2_source_checkpoint_hash = manifest[
                'goal_checkpoint_hash']
        if args.use_trajectory_bank_cache:
            for split, loader in self.data_loaders.items():
                stats = pack_statistics(loader)
                pair_mb = (stats['avg_edge_upper_bound_per_pack'] *
                           args.num_samples ** 2 * 4 / 1024 ** 2)
                print(
                    f"Packed {split}: windows={stats['num_windows']}, "
                    f"packs={stats['num_packs']}, "
                    f"avg scenes/agents/edge_bound="
                    f"{stats['avg_scenes_per_pack']:.2f}/"
                    f"{stats['avg_agents_per_pack']:.2f}/"
                    f"{stats['avg_edge_upper_bound_per_pack']:.2f}, "
                    f"pair-score lower-bound memory={pair_mb:.2f} MiB, "
                    f"oversized singletons={stats['oversized_singletons']}")
        self._write_evaluation_protocol()
        # initialize device
        self.device = self._set_device()
        self._configure_amp()
        self._pending_training_state = None
        # initialize network
        self.net = GDTS(self.args, self.device).to(self.device)
        initialization_checkpoint = self.args.pretrain_path
        if self.net.jdv2_active and initialization_checkpoint is None:
            initialization_checkpoint = self.args.jdv2_source_checkpoint
        if initialization_checkpoint:
            self._load_state_file(
                initialization_checkpoint, baseline_initialization=True)

        # Prepare log curve file and initialize best validation metrics
        self.log_curve_file = os.path.join(self.args.model_dir, 'log_curve.txt')

        # Best metrics
        self.best_metrics = self.net.init_best_metrics()
        self.best_metrics_epochs = {k: -1 for k in self.best_metrics.keys()}
        self._best_selection = (float('inf'), float('inf'))

    def _configure_amp(self):
        """Configure the frozen FP16/BF16 policy without changing legacy."""
        self.amp_enabled = bool(getattr(self.args, 'amp_enabled', False))
        dtype_name = getattr(self.args, 'amp_dtype', 'fp32')
        self.amp_dtype = {
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
        }[dtype_name]
        if self.amp_enabled and dtype_name == 'fp16' and \
                self.device.type != 'cuda':
            raise RuntimeError('FP16 AMP requires CUDA')
        if (self.amp_enabled and dtype_name == 'bf16' and
                self.device.type == 'cuda' and
                not torch.cuda.is_bf16_supported()):
            raise RuntimeError('Requested BF16 is unsupported by this GPU')
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp_enabled and dtype_name == 'fp16')

    def _autocast_context(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=True)

    def _jdv2_architecture_config(self):
        strict_no_z = self.args.jdv2_latent_objective == 'strict_no_z'
        config = {
            'scene_modes': 0 if strict_no_z else self.args.jdv2_scene_modes,
            'relation_modes': self.args.jdv2_relation_modes,
            'goal_candidates': self.args.num_goal_candidates,
            'joint_samples': self.args.num_samples,
            'prediction_steps': self.args.pred_length,
            'energy_rank': self.args.jdv2_energy_rank,
            'agent_dim': self.args.social_feature_dim,
            'latent_objective': self.args.jdv2_latent_objective,
        }
        if strict_no_z:
            config['architecture_variant'] = 'strict_no_z'
        return config

    def _jdv2_ablation_config(self):
        return {name: bool(getattr(self.args, name)) for name in (
            'use_scene_latent', 'use_dynamic_relation', 'use_joint_energy',
            'use_dependency_corrector')}

    def _selection_tuple(self, valid_metrics):
        primary_name = self.net.best_valid_metric()
        primary = float(valid_metrics['valid_' + primary_name])
        tie = float('inf')
        if (self.net.jdv2_active and
                self.args.training_stage == 'joint_trajectory'):
            tie = float(valid_metrics['valid_JFDE'])
        return primary, tie

    @staticmethod
    def _selection_is_better(current, best, tolerance=1e-12):
        if current[0] < best[0] - tolerance:
            return True
        return (abs(current[0] - best[0]) <= tolerance and
                current[1] < best[1] - tolerance)

    @staticmethod
    def _accumulate_joint_diagnostics(sums, counts, diagnostics):
        """Accumulate finite scalar diagnostics without retaining tensors."""
        for name, value in diagnostics.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                value = float(value.detach().cpu())
            if not isinstance(value, (int, float, np.number)):
                continue
            value = float(value)
            if not np.isfinite(value):
                raise FloatingPointError(
                    f'Non-finite joint diagnostic {name}={value}')
            weight = 1.0
            if name.endswith('_e_gt0') and name != 'scene_count_e_gt0':
                weight = float(diagnostics.get('scene_count_e_gt0', 0.0))
            elif name.endswith('_e_eq0') and name != 'scene_count_e_eq0':
                weight = float(diagnostics.get('scene_count_e_eq0', 0.0))
            if weight <= 0:
                continue
            sums[name] = sums.get(name, 0.0) + value * weight
            counts[name] = counts.get(name, 0.0) + weight

    @staticmethod
    def _finalize_joint_diagnostics(sums, counts):
        return {
            name: sums[name] / counts[name]
            for name in sorted(sums) if counts.get(name, 0) > 0
        }

    @staticmethod
    def _jdv2_early_collapse_signature(diagnostics):
        """Return the locked E>0 collapse signature and supporting values."""
        required = [
            'p_entropy_e_gt0', 'q_entropy_e_gt0',
            'gamma_entropy_e_gt0', 'future_shuffle_l1_e_gt0',
        ]
        for family in ('p', 'q', 'gamma'):
            required.extend(
                f'{family}_usage_{mode}_e_gt0' for mode in range(4))
        required.extend(
            f'gamma_hard_usage_{mode}_e_gt0' for mode in range(4))
        if any(name not in diagnostics for name in required):
            return False, {'reason': 'insufficient_e_gt0_diagnostics'}
        usage = {
            family: [diagnostics[f'{family}_usage_{mode}_e_gt0']
                     for mode in range(4)]
            for family in ('p', 'q', 'gamma')}
        dominant = {family: int(np.argmax(values))
                    for family, values in usage.items()}
        same_mode = len(set(dominant.values())) == 1
        mode = dominant['gamma']
        signature = (
            same_mode and
            all(max(values) > 0.99 for values in usage.values()) and
            diagnostics['p_entropy_e_gt0'] < 1e-3 and
            diagnostics['q_entropy_e_gt0'] < 1e-3 and
            diagnostics['gamma_entropy_e_gt0'] < 1e-3 and
            diagnostics['future_shuffle_l1_e_gt0'] < 1e-6 and
            diagnostics[f'gamma_hard_usage_{mode}_e_gt0'] >= 0.95)
        return signature, {
            'dominant_modes': dominant,
            'maximum_soft_usage': {
                family: float(max(values))
                for family, values in usage.items()},
            'p_entropy': float(diagnostics['p_entropy_e_gt0']),
            'q_entropy': float(diagnostics['q_entropy_e_gt0']),
            'gamma_entropy': float(diagnostics['gamma_entropy_e_gt0']),
            'future_shuffle_l1': float(
                diagnostics['future_shuffle_l1_e_gt0']),
            'gamma_dominant_hard_usage': float(
                diagnostics[f'gamma_hard_usage_{mode}_e_gt0']),
        }

    def _write_jdv2_epoch_diagnostics(
            self, epoch, train_losses, valid_metrics, runtime_seconds):
        """Persist one compact, auditable Stage-A/Stage-B epoch record."""
        if not self.net.jdv2_active:
            return
        diagnostics_dir = os.path.join(self.args.model_dir, 'diagnostics')
        os.makedirs(diagnostics_dir, exist_ok=True)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        else:
            peak_allocated = 0
            peak_reserved = 0
        record = {
            'epoch': int(epoch),
            'training_stage': self.args.training_stage,
            'stage_progress': float(self.args.jdv2_stage_progress),
            'loss_coefficients': {
                name: float(value)
                for name, value in self.net.set_losses_coeffs().items()},
            'train_losses': {
                name: float(value) for name, value in train_losses.items()},
            'validation_metrics': {
                name: float(value) for name, value in valid_metrics.items()},
            'train_joint_diagnostics': getattr(
                self, 'last_train_joint_diagnostics', {}),
            'validation_joint_diagnostics': getattr(
                self, 'last_valid_joint_diagnostics', {}),
            'system': {
                'runtime_seconds': float(runtime_seconds),
                'peak_cuda_allocated_bytes': peak_allocated,
                'peak_cuda_reserved_bytes': peak_reserved,
            },
        }
        epoch_path = os.path.join(
            diagnostics_dir, f'epoch_{int(epoch):03d}.json')
        with open(epoch_path, 'w') as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
        with open(os.path.join(
                self.args.model_dir, 'joint_diagnostics.jsonl'), 'a') as handle:
            handle.write(json.dumps(record, sort_keys=True) + '\n')
        if valid_metrics:
            with open(os.path.join(
                    self.args.model_dir, 'validation_log.jsonl'), 'a') as handle:
                handle.write(json.dumps({
                    'epoch': int(epoch),
                    'metrics': record['validation_metrics'],
                    'joint_diagnostics': record[
                        'validation_joint_diagnostics'],
                }, sort_keys=True) + '\n')

    def _write_evaluation_protocol(self):
        """Persist the exact model-selection/test separation used by a run."""
        protocol = {
            'model_selection_split': self.args.model_selection_split,
            'final_test_split': self.args.final_test_split,
            'upstream_generator': self.args.upstream_generator,
            'upstream_checkpoint': self.args.pretrain_path,
            'freeze_upstream_generator': (
                self.args.freeze_upstream_generator),
            'internal_validation_fraction': (
                self.args.internal_validation_fraction),
            'internal_validation_seed': self.args.internal_validation_seed,
            'internal_validation_strategy': (
                self.args.internal_validation_strategy),
            'internal_validation_source': getattr(
                self.data_loaders['valid'].dataset,
                'internal_validation_source', None),
            'train_window_count': len(self.data_loaders['train'].dataset),
            'valid_window_count': len(self.data_loaders['valid'].dataset),
            'test_window_count': len(self.data_loaders['test'].dataset),
            'use_trajectory_bank_cache': (
                self.args.use_trajectory_bank_cache),
            'use_multi_scene_packing': self.args.use_multi_scene_packing,
            'scene_balanced_loss': self.args.scene_balanced_loss,
            'num_cached_seeds_per_window': (
                self.args.num_cached_seeds_per_window),
            'validation_seed': self.args.validation_seed,
            'jdv2_latent_objective': self.args.jdv2_latent_objective,
        }
        if self.args.use_trajectory_bank_cache:
            protocol['trajectory_bank_manifest'] = \
                self.data_loaders['train'].dataset.manifest
            protocol['packing'] = {
                split: pack_statistics(loader)
                for split, loader in self.data_loaders.items()
            }
        for split in ('train', 'valid'):
            ids = getattr(self.data_loaders[split].dataset, 'source_ids', None)
            if ids is not None:
                protocol[f'{split}_cache_ids'] = ids
        os.makedirs(self.args.model_dir, exist_ok=True)
        protocol_path = os.path.join(
            self.args.model_dir, 'evaluation_protocol.json')
        with open(protocol_path, 'w') as handle:
            json.dump(protocol, handle, indent=2, sort_keys=True)
        print(
            'Evaluation protocol: model_selection_split='
            f'{self.args.model_selection_split}, '
            f'final_test_split={self.args.final_test_split}')

    def _save_checkpoint(self, epoch, best_epoch=False, last_epoch=False):
        """
        Save model and optimizer states
        """
        saved_models_path = os.path.join(self.args.model_dir, 'saved_models')
        if not os.path.exists(saved_models_path):
            os.makedirs(saved_models_path)
        # Save current checkpoint
        if best_epoch and last_epoch:
            raise ValueError('A checkpoint cannot be both best and last')
        if best_epoch:
            saved_model_name = os.path.join(
                saved_models_path, 'best_model.pt')
        elif last_epoch:
            saved_model_name = os.path.join(
                saved_models_path, 'last_model.pt')
        else:
            saved_model_name = os.path.join(saved_models_path, 'epoch_' +str(epoch).zfill(3) + '.pt')
        payload = {
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
        }
        if self.net.jdv2_active:
            payload.update({
                'optimizer_state_dict': (
                    self.optimizer.state_dict()
                    if hasattr(self, 'optimizer') else None),
                'scheduler_state_dict': (
                    self.scheduler.state_dict()
                    if getattr(self, 'scheduler', None) is not None else None),
                'grad_scaler_state_dict': (
                    self.scaler.state_dict() if self.scaler.is_enabled()
                    else None),
                'architecture_config': self._jdv2_architecture_config(),
                'ablation_config': self._jdv2_ablation_config(),
                'training_stage': self.args.training_stage,
                'latent_objective': self.args.jdv2_latent_objective,
                'stage_progress': self.args.jdv2_stage_progress,
                'stage_optimizer_steps_completed': int(getattr(
                    self, '_stage_optimizer_steps_completed', 0)),
                'cache_manifest_hash': self.args.jdv2_cache_manifest_hash,
                'source_checkpoint_hash': (
                    self.args.jdv2_source_checkpoint_hash),
                'best_metric': {
                    'name': self.net.best_valid_metric(),
                    'primary': self._best_selection[0],
                    'tie_break': self._best_selection[1],
                    'seed': self.args.seed,
                },
                'best_selection': {
                    'primary': self._best_selection[0],
                    'tie_break': self._best_selection[1],
                },
                'best_metrics': {
                    key: float(value)
                    for key, value in self.best_metrics.items()},
                'best_metrics_epochs': dict(self.best_metrics_epochs),
            })
        torch.save(payload, saved_model_name)

    def _restore_best_state(self, checkpoint):
        """Restore checkpoint-selection history, including old checkpoints."""
        selection = checkpoint.get('best_selection')
        if selection is None:
            selection = checkpoint.get('best_metric', {})
        if 'primary' in selection:
            self._best_selection = (
                float(selection['primary']),
                float(selection.get('tie_break', float('inf'))))
        saved_metrics = checkpoint.get('best_metrics')
        saved_epochs = checkpoint.get('best_metrics_epochs')
        if saved_metrics is not None:
            for key in self.best_metrics:
                if key in saved_metrics:
                    self.best_metrics[key] = float(saved_metrics[key])
        if saved_epochs is not None:
            for key in self.best_metrics_epochs:
                if key in saved_epochs:
                    self.best_metrics_epochs[key] = int(saved_epochs[key])
        else:
            self._restore_best_tables_from_curve(int(checkpoint.get('epoch', 0)))

    def _restore_best_tables_from_curve(self, through_epoch):
        """Reconstruct historical per-metric minima for legacy JDV2 files."""
        if through_epoch <= 0 or not os.path.isfile(self.log_curve_file):
            return
        with open(self.log_curve_file) as handle:
            header = handle.readline().strip().split(',')
            rows = [line.strip().split(',') for line in handle if line.strip()]
        index = {name: offset for offset, name in enumerate(header)}
        for row in rows:
            if len(row) != len(header):
                continue
            epoch = int(float(row[index['epoch']]))
            if epoch > through_epoch:
                continue
            for key in self.best_metrics:
                column = 'valid_' + key
                if column not in index:
                    continue
                value = float(row[index[column]])
                if value < self.best_metrics[key]:
                    self.best_metrics[key] = value
                    self.best_metrics_epochs[key] = epoch

    def _load_state_file(self, saved_model_name, baseline_initialization=False):
        """Load a checkpoint with explicit legacy-to-joint compatibility."""
        checkpoint = torch.load(saved_model_name, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Only a legacy baseline used to initialize a larger structured model
        # is intentionally partial. Resuming/testing any current checkpoint
        # must be strict so missing joint modules cannot remain random silently.
        if not self.net.jdv2_active:
            strict = True
        else:
            source_is_v2 = any(
                key.startswith('jdv2_') for key in state_dict)
            strict = source_is_v2 or not baseline_initialization
        if self.net.jdv2_active and strict and source_is_v2:
            if checkpoint.get('latent_objective') != \
                    self.args.jdv2_latent_objective:
                raise RuntimeError('V2 checkpoint latent objective mismatch')
            expected_architecture = self._jdv2_architecture_config()
            expected_ablation = self._jdv2_ablation_config()
            if checkpoint.get('architecture_config') != expected_architecture:
                raise RuntimeError('V2 checkpoint architecture mismatch')
            if checkpoint.get('ablation_config') != expected_ablation:
                raise RuntimeError('V2 checkpoint ablation mismatch')
        incompatible = self.net.load_state_dict(state_dict, strict=strict)
        if not strict:
            missing = list(incompatible.missing_keys)
            unexpected = list(incompatible.unexpected_keys)
            allowed_prefixes = (
                'interaction_graph.', 'social_encoder.',
                'relation_inference.', 'jdv2_')
            invalid_missing = [
                key for key in missing
                if not key.startswith(allowed_prefixes)]
            if invalid_missing or unexpected:
                raise RuntimeError(
                    'Legacy -> JDV2 checkpoint incompatibility: '
                    f'invalid_missing={invalid_missing}, '
                    f'unexpected={unexpected}')
            print('Legacy baseline initialization: allowed JDV2 missing '
                  f'keys={missing}')
        if self.net.jdv2_active and strict:
            if checkpoint.get('cache_manifest_hash') != \
                    self.args.jdv2_cache_manifest_hash:
                raise RuntimeError('V2 checkpoint cache manifest mismatch')
            if checkpoint.get('source_checkpoint_hash') != \
                    self.args.jdv2_source_checkpoint_hash:
                raise RuntimeError('V2 checkpoint source checkpoint mismatch')
            source_stage = checkpoint.get('training_stage')
            target_stage = self.args.training_stage
            allowed_transition = (
                source_stage == target_stage or
                (source_stage == 'joint_goal' and
                 target_stage == 'joint_trajectory') or
                (source_stage in {'joint_goal', 'joint_trajectory'} and
                 target_stage == 'joint_finetune'))
            if not allowed_transition:
                raise RuntimeError(
                    f'Invalid V2 stage transition {source_stage!r} -> '
                    f'{target_stage!r}')
            self._pending_training_state = checkpoint
        return checkpoint.get('epoch', 0)

    def _load_checkpoint(self, load_checkpoint):
        """
        Load a pre-trained model. Can then be used to test or resume training.
        """
        if load_checkpoint is not None:
            # Create load model path
            if load_checkpoint == 'best':
                saved_model_name = os.path.join(self.args.model_dir, 'saved_models', 'best_model.pt')
            else:  # Load specific checkpoint
                assert int(load_checkpoint) > 0, \
                    "Check args.load_model. Must be an integer > 0"
                saved_model_name = os.path.join(
                    self.args.model_dir, 'saved_models', 'epoch_' + str(load_checkpoint).zfill(3) + '.pt')
            print("\nSaved model path:", saved_model_name)
            # Load model
            if os.path.isfile(saved_model_name):
                print('Loading checkpoint ...')
                model_epoch = self._load_state_file(saved_model_name)
                print('Loaded checkpoint at epoch', model_epoch, '\n')
                return model_epoch
            else:
                raise ValueError("No such pre-trained model:", saved_model_name)
        else:
            raise ValueError('You need to specify an epoch (int) if you want '
                             'to load a model or "best" to load the best '
                             'model! Check args.load_checkpoint')

    def _load_or_restart(self):
        """
        Load a pre-trained model to resume training or restart from scratch.
        Can start from scratch or resume training, depending on input
        self.args.load_checkpoint parameter.
        """
        # load pre-trained model to resume training
        if self.args.load_checkpoint is not None:
            loaded_epoch = self._load_checkpoint(self.args.load_checkpoint)
            source_stage = (self._pending_training_state or {}).get(
                'training_stage')
            if source_stage is not None and source_stage != \
                    self.args.training_stage:
                # A cross-stage checkpoint initializes weights only. Epoch,
                # curriculum progress, optimizer, and scheduler are local to
                # the target stage.
                self.args.jdv2_stage_progress = 0.0
                start_epoch = 1
            else:
                start_epoch = int(loaded_epoch) + 1
        else:
            start_epoch = 1
            # log_file header only the first time
            curve_metrics = ['ADE', 'FDE']
            if self.net.jdv2_active:
                curve_metrics = list(self.net.init_test_metrics().keys())
            if self.args.trajectory_alignment:
                curve_metrics.extend([
                    'ADE_world', 'FDE_world', 'JADE', 'JFDE',
                    'Raw_JADE', 'Raw_JFDE'])
            if self.args.trajectory_coupling == 'multiway_v4':
                curve_metrics = (
                    list(self.net.init_test_metrics().keys()) +
                    list(V4_DIAGNOSTIC_NAMES))
            self.curve_metric_names = curve_metrics
            self.curve_loss_names = sorted(self.net.init_losses().keys()) + [
                'loss_total']
            with open(self.log_curve_file, 'w') as f:
                f.write("epoch,learning_rate," +
                        ','.join('valid_' + name for name in curve_metrics) +
                        ',' +
                        ",".join(self.curve_loss_names) +
                        "\n")
        if not hasattr(self, 'curve_metric_names'):
            self.curve_metric_names = ['ADE', 'FDE']
            if self.net.jdv2_active:
                self.curve_metric_names = list(
                    self.net.init_test_metrics().keys())
            if self.args.trajectory_alignment:
                self.curve_metric_names.extend([
                    'ADE_world', 'FDE_world', 'JADE', 'JFDE',
                    'Raw_JADE', 'Raw_JFDE'])
            if self.args.trajectory_coupling == 'multiway_v4':
                self.curve_metric_names = (
                    list(self.net.init_test_metrics().keys()) +
                    list(V4_DIAGNOSTIC_NAMES))
        if not hasattr(self, 'curve_loss_names'):
            self.curve_loss_names = sorted(self.net.init_losses().keys()) + [
                'loss_total']
        return start_epoch

    def test(self, load_checkpoint):
        """
        Load a trained model and test it on the test set.
        """
        print('*** Test phase started ***')
        # some models do not need to be trained nor loaded
        if self.net.is_trainable:
            best_epoch = self._load_checkpoint(load_checkpoint)
        else:
            best_epoch = load_checkpoint
        print('Testing ...')
        total_results = []
        if self.args.use_trajectory_bank_cache:
            run_count = int(self.data_loaders['test'].dataset.manifest[
                'num_cached_seeds'])
            print(
                f'Cached evaluation protocol: {run_count} independent '
                f'seeds, each evaluated with K={self.args.num_samples}; '
                'candidate banks are not merged.')
        else:
            run_count = self.args.num_test_runs
        for run_idx in range(run_count):
            print(f"\nTest run #{run_idx} ...")
            if self.args.use_trajectory_bank_cache:
                self.data_loaders['test'].dataset.set_seed_index(run_idx)
            test_metrics = self._evaluate_epoch(best_epoch, mode='test')
            run_results = dict(**test_metrics)
            # print losses and metrics for run i
            print(f'Test_set: {self.args.test_set},',
                  f'test_run_idx: {run_idx},',
                  f'epoch: {load_checkpoint},',
                  ', '.join([f"{k}={v:.5f}" for k, v in run_results.items()]))
            total_results.append(run_results)
        average_results = {k: np.mean([i[k] for i in total_results])
                           for k in run_results}
        std_results = {k: np.std([i[k] for i in total_results])
                       for k in run_results}
        # print average losses and metrics for
        print("\n" + "#"*25)
        print("#"*5 + " FINAL RESULTS " + "#"*5)
        print("#" * 25)
        print(f'Test_set: {self.args.test_set},',
              f'epoch: {load_checkpoint},',
              ', '.join([f"{k}={v:.5f}" for k, v in average_results.items()]))
        if self.args.use_trajectory_bank_cache:
            print('Seed std: ' + ', '.join(
                f'{key}={value:.5f}' for key, value in std_results.items()))
        if self.args.trajectory_coupling == 'multiway_v4':
            result_path = os.path.join(self.args.model_dir,
                                       'final_test_results.json')
            with open(result_path, 'w') as handle:
                json.dump({
                    'checkpoint': load_checkpoint,
                    'checkpoint_epoch': best_epoch,
                    'split': self.args.final_test_split,
                    'runs': [
                        {key: float(value) for key, value in run.items()}
                        for run in total_results],
                    'average': {
                        key: float(value)
                        for key, value in average_results.items()},
                    'std': {
                        key: float(value)
                        for key, value in std_results.items()},
                    'cached_seed_protocol': (
                        'independent K per seed; mean/std across seeds'),
                }, handle, indent=2, sort_keys=True)
        return average_results

    def _set_optimizer(self, parameters):
        """Construct the configured main-training optimizer."""
        optimizer_class = {
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'SGD': torch.optim.SGD,
        }[self.args.optimizer]
        return optimizer_class(parameters, lr=self.args.learning_rate)

    def _optimizer_parameter_groups(self):
        """Build named differential-LR groups without duplicating tensors."""
        families = self.net.optimizer_parameter_families()
        groups = []
        specifications = (
            (('jdv2_goal_dependency', self.args.structured_lr_scale),
             ('jdv2_corrector', self.args.structured_lr_scale))
            if self.net.jdv2_active else
            (('structured', self.args.structured_lr_scale),
             ('baseline', self.args.baseline_lr_scale)))
        for name, scale in specifications:
            parameters = families[name]
            if not parameters:
                continue
            groups.append({
                'params': parameters,
                'lr': self.args.learning_rate * scale,
                'name': name,
            })
        if not groups:
            raise ValueError('No optimizer parameters selected for training.')
        print('Optimizer parameter groups: ' + ', '.join(
            f"{group['name']}={len(group['params'])} tensors, "
            f"lr={group['lr']:.3g}" for group in groups))
        return groups

    def _set_scheduler(self, optimizer):
        """Construct the configured scheduler, or return ``None``."""
        if self.args.scheduler == 'None':
            return None
        if self.args.scheduler == 'ExponentialLR':
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.995)
        if self.args.scheduler == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(self.args.num_epochs, 1))
        if self.args.scheduler == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=20,
                min_lr=1e-6)
        raise ValueError(f'Unsupported scheduler={self.args.scheduler!r}')

    def train(self):
        """
        Train the model. Wrapper for train_loop.
        """
        # find where to start
        start_epoch = self._load_or_restart()

        # Establish the correct progressive-freeze state before the summary.
        self.net.configure_training_epoch(start_epoch)

        # print model info
        print_model_summary(self.net)

        self.optimizer = self._set_optimizer(
            self._optimizer_parameter_groups())
        self.scheduler = self._set_scheduler(self.optimizer)
        self._stage_optimizer_steps_completed = 0
        if self._pending_training_state is not None:
            checkpoint = self._pending_training_state
            same_stage = checkpoint.get('training_stage') == \
                self.args.training_stage
            if same_stage and checkpoint.get('optimizer_state_dict') is not None:
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])
            if (same_stage and self.scheduler is not None and
                    checkpoint.get('scheduler_state_dict') is not None):
                self.scheduler.load_state_dict(
                    checkpoint['scheduler_state_dict'])
            if (same_stage and self.scaler.is_enabled() and
                    checkpoint.get('grad_scaler_state_dict') is not None):
                self.scaler.load_state_dict(
                    checkpoint['grad_scaler_state_dict'])
            if same_stage:
                self.args.jdv2_stage_progress = float(
                    checkpoint.get('stage_progress',
                                   self.args.jdv2_stage_progress))
                saved_steps = checkpoint.get(
                    'stage_optimizer_steps_completed')
                if saved_steps is None:
                    total_steps = max(
                        self.args.num_epochs *
                        len(self.data_loaders['train']), 1)
                    saved_steps = round(
                        self.args.jdv2_stage_progress * total_steps)
                self._stage_optimizer_steps_completed = int(saved_steps)
                self._restore_best_state(checkpoint)
            else:
                self.args.jdv2_stage_progress = 0.0
                self._stage_optimizer_steps_completed = 0

        # start training
        self._train_loop(start_epoch=start_epoch, end_epoch=self.args.num_epochs)

    def train_test(self):
        """
        Perform training and then test on the best validation epoch.
        """
        self.train()
        print()
        self.test(load_checkpoint='best')

    def _train_loop(self, start_epoch, end_epoch):
        """
        Train the model. Loop over the epochs, train and update network
        parameters, save model, check results on validation set,
        print results and save log data.
        """
        # saved metrics before validation begins
        valid_metrics = {"valid_ADE": 0, "valid_FDE": 0}

        # initial learning rate
        if self.scheduler is not None:
            learning_rate = self.optimizer.param_groups[0]['lr']
        else:
            learning_rate = self.args.learning_rate

        phase_name = 'Train'
        best_metric_name = self.net.best_valid_metric()

        print(f'*** {phase_name} phase started ***')
        print(f"Starting epoch: {start_epoch}, final epoch: {end_epoch}")

        self.current_date = datetime.datetime.now().strftime("%m%d")
        self.current_time = datetime.datetime.now().strftime("%H%M")
        if self.args.use_wandb and wandb is None:
            raise ImportError('wandb is required only when --use_wandb True')
        if self.args.use_wandb and wandb.run is None:
            wandb.init(settings=wandb.Settings(start_method="thread"),
                       project="GDTS", config=self.args, entity="iadc_erwinsun",
                       group=f"{self.args.dataset}",
                       job_type=f"{self.args.test_set}",
                       tags=None, name=f'{self.args.test_set}_{self.current_date}_{self.current_time}')

        validations_without_improvement = 0
        collapse_signature_epochs = 0
        previous_baseline_trainable = None
        for epoch in range(start_epoch, end_epoch + 1):
            if self.args.use_trajectory_bank_cache:
                train_loader = self.data_loaders['train']
                train_loader.dataset.set_epoch(epoch)
                train_loader.dataset.set_seed_index(None)
                train_loader.batch_sampler.set_epoch(epoch)
                # Checkpoint selection always uses cached seed index 0.
                self.data_loaders['valid'].dataset.set_seed_index(0)
            self.net.configure_training_epoch(epoch)
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)
            baseline_trainable = any(
                parameter.requires_grad
                for module in self.net._baseline_modules()
                for parameter in module.parameters())
            if baseline_trainable != previous_baseline_trainable:
                state = 'unfrozen' if baseline_trainable else 'frozen'
                print(f'Baseline parameter family is {state} at epoch {epoch}.')
                previous_baseline_trainable = baseline_trainable
            start_time = time.time()  # time epoch
            train_losses = self._train_epoch(epoch)
            should_stop = False
            stop_reason = None
            epoch_valid_metrics = {}

            if epoch % self.args.save_every == 0:
                self._save_checkpoint(epoch)  # save model
                print(f"Saved checkpoint at epoch {epoch}")

            # validation
            if epoch >= self.args.start_validation and epoch % self.args.validate_every == 0:
                valid_metrics = self._evaluate_epoch(epoch, mode='valid')
                epoch_valid_metrics = valid_metrics
                current_selection = self._selection_tuple(valid_metrics)
                selection_improved = self._selection_is_better(
                    current_selection, self._best_selection)

                # comment some of this print, if it is too long
                print(f'----Epoch {epoch},',
                      f'time/epoch={formatted_time(time.time() - start_time)},',
                      f'learning_rate={learning_rate:.5f},',
                      ', '.join([f"{loss_name}={loss_value:.5f}" for
                                 loss_name, loss_value in
                                 train_losses.items()]))
                print(', '.join([f"{metric_name}={metric_value:.3f}" for
                                 metric_name, metric_value in
                                 valid_metrics.items()]))

                # update best model and best metrics
                for k, v in self.best_metrics.items():
                    current_metric_loss = valid_metrics["valid_" + k]

                    if current_metric_loss < v:
                        self.best_metrics[k] = current_metric_loss
                        self.best_metrics_epochs[k] = epoch
                if selection_improved:
                    self._best_selection = current_selection
                    self._save_checkpoint(epoch, best_epoch=True)
                    print(
                        f"Saved best model at epoch {epoch}: "
                        f"primary={current_selection[0]:.6g}, "
                        f"tie_break={current_selection[1]:.6g}")

                print(', '.join([f"best_{metric_name}={metric_value:.3f}" for
                                 metric_name, metric_value in
                                 self.best_metrics.items()]))
                print(', '.join([f"best_epoch_{metric_name}="
                                 f"{metric_epoch}" for
                                 metric_name, metric_epoch in
                                 self.best_metrics_epochs.items()]))

                if selection_improved:
                    validations_without_improvement = 0
                else:
                    validations_without_improvement += 1
                if (self.args.early_stopping_patience > 0 and
                        validations_without_improvement >=
                        self.args.early_stopping_patience):
                    should_stop = True
                    stop_reason = (
                        f'{best_metric_name} did not improve for '
                        f'{validations_without_improvement} validation checks')
                
                # save metrics to log_curve.txt
                with open(self.log_curve_file, 'a') as f:
                    f.write(','.join(str(m) for m in [
                        epoch, learning_rate] +
                        [valid_metrics['valid_' + name]
                         for name in self.curve_metric_names] +
                        [train_losses['train_' + loss_name]
                         for loss_name in self.curve_loss_names]) + '\n')
            else:
                print(f'----Epoch {epoch},',
                      f'time/epoch={formatted_time(time.time() - start_time)},',
                      f'learning_rate={learning_rate:.5f},',
                      ', '.join([f"{loss_name}={loss_value:.5f}" for
                                 loss_name, loss_value in
                                 train_losses.items()]))
            if self.scheduler is not None:
                if self.args.scheduler == 'ReduceLROnPlateau':
                    if (epoch >= self.args.start_validation and
                            epoch % self.args.validate_every == 0):
                        self.scheduler.step(
                            valid_metrics['valid_' + best_metric_name])
                else:
                    self.scheduler.step()
            learning_rate = self.optimizer.param_groups[0]['lr']

            self._write_jdv2_epoch_diagnostics(
                epoch, train_losses, epoch_valid_metrics,
                time.time() - start_time)

            if (self.net.jdv2_active and
                    self.args.training_stage == 'joint_goal' and
                    self.args.jdv2_latent_objective ==
                    'v2_marginal_responsibility' and
                    self.args.jdv2_early_collapse_gate and epoch >= 2):
                collapse, evidence = self._jdv2_early_collapse_signature(
                    self.last_train_joint_diagnostics)
                collapse_signature_epochs = (
                    collapse_signature_epochs + 1 if collapse else 0)
                if collapse:
                    print(
                        'JDV2 early collapse diagnostic hit '
                        f'{collapse_signature_epochs}/2: {evidence}')
                if collapse_signature_epochs >= 2:
                    should_stop = True
                    stop_reason = (
                        'Stage-A V2 E>0 latent collapse signature persisted '
                        'for two consecutive epochs')
                    with open(os.path.join(
                            self.args.model_dir,
                            'early_collapse_gate.json'), 'w') as handle:
                        json.dump({
                            'epoch': int(epoch),
                            'consecutive_epochs': collapse_signature_epochs,
                            'thresholds': {
                                'aggregate_soft_usage': '>0.99',
                                'entropy': '<1e-3',
                                'future_shuffle_l1': '<1e-6',
                                'hard_usage': '>=0.95',
                            },
                            'evidence': evidence,
                        }, handle, indent=2, sort_keys=True)

            # The explicit last checkpoint is written only after the epoch's
            # optimizer and scheduler work has completed.
            self._save_checkpoint(epoch, last_epoch=True)

            # save metrics to WandB
            if self.args.use_wandb and epoch >= self.args.start_validation:
                wandb.log({'learning_rate': learning_rate}, step=epoch)
                wandb.log(train_losses, step=epoch)
                # validation
                if epoch >= self.args.start_validation and epoch % self.args.validate_every == 0:
                    wandb.log(valid_metrics, step=epoch)
                    # best metrics
                    for k, v in self.best_metrics.items():
                        wandb.run.summary["best_" + k] = v
                    for k, v in self.best_metrics_epochs.items():
                        wandb.run.summary["best_epoch_" + k] = v
            if should_stop:
                print(f'Early stopping at epoch {epoch}: {stop_reason}.')
                break
        if self.args.trajectory_coupling == 'multiway_v4':
            summary_path = os.path.join(
                self.args.model_dir, 'training_summary.json')
            with open(summary_path, 'w') as handle:
                json.dump({
                    'selection_metric': best_metric_name,
                    'best_metrics': {
                        key: float(value)
                        for key, value in self.best_metrics.items()},
                    'best_metric_epochs': self.best_metrics_epochs,
                    'model_selection_split': (
                        self.args.model_selection_split),
                    'final_test_split': self.args.final_test_split,
                }, handle, indent=2, sort_keys=True)

    def _train_epoch(self, epoch):
        """
        Train one epoch of the model on the whole training set.
        """
        self.net.train()  # train mode
        if (self.args.training_stage == 'multiway_coupling' and
                self.args.freeze_upstream_generator):
            # Cached proposals are generated in eval mode.  Keep the frozen
            # online ablation numerically equivalent by preventing BatchNorm/
            # dropout state changes in the upstream family while leaving the
            # V4 coupler itself in train mode.
            for module in self.net._baseline_modules():
                module.eval()

        # INIT LOSSES and METRICS
        losses_epoch = self.net.init_losses()
        losses_coeffs = self.net.set_losses_coeffs()

        # Progress bar
        train_bar = tqdm(self.data_loaders['train'], ascii=True, ncols=100,
                         desc=f'Epoch {epoch}. Train batches')
        num_train_batches = len(self.data_loaders['train'])
        total_stage_optimizer_steps = max(
            self.args.num_epochs * num_train_batches, 1)
        accumulation_steps = (
            self.args.coupling_grad_accum_steps
            if self.args.training_stage == 'multiway_coupling' else 1)
        pending_backward = 0
        total_loss_epoch = 0.0
        diagnostic_sums = {}
        diagnostic_counts = {}
        self.optimizer.zero_grad()

        def optimizer_step(pending, rescale_partial=False):
            if pending == 0:
                return
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            if rescale_partial and pending < accumulation_steps:
                correction = float(accumulation_steps) / float(pending)
                for parameter in self.net.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            for name, parameter in self.net.named_parameters():
                if parameter.grad is not None and \
                        not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(
                        f'Non-finite gradient: module/parameter={name}, '
                        f'shape={tuple(parameter.grad.shape)}, '
                        f'dtype={parameter.grad.dtype}')
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), self.args.clip)
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.net.jdv2_active:
                self._stage_optimizer_steps_completed += 1
            self.optimizer.zero_grad()

        for batch_index, batch in enumerate(train_bar):
            if self.net.jdv2_active:
                # Curriculum state is a stage-local count of successful
                # optimizer steps. Cross-stage loading initializes this at 0.
                self.args.jdv2_stage_progress = min(
                    self._stage_optimizer_steps_completed /
                    total_stage_optimizer_steps, 1.0)
            if self.args.use_trajectory_bank_cache:
                inputs = self.net.prepare_cached_trajectory_pack(batch)
                with self._autocast_context():
                    losses, coupling = self.net.get_cached_multiway_loss(inputs)
                seq_list = None
                batch_id = None
                del batch, coupling
            else:
                batch_data, batch_id = batch
                inputs, seq_list = self.net.prepare_inputs(
                    batch_data, batch_id)
                del batch_data
                with self._autocast_context():
                    losses = self.net.get_loss(inputs, seq_list)
            if self.net.jdv2_active:
                # KL coefficients follow completed optimizer-step progress,
                # not an epoch-frozen approximation.
                losses_coeffs = self.net.set_losses_coeffs()
            loss = torch.zeros(1).to(self.device)
            for loss_name, loss_value in losses.items():
                loss += losses_coeffs[loss_name]*loss_value
                losses_epoch[loss_name] += loss_value.item()
            total_loss_epoch += float(loss.detach().cpu())

            if not torch.isfinite(loss).all():
                raise FloatingPointError(
                    f'Non-finite total loss; components={losses}')
            if self.args.joint_diagnostics and hasattr(
                    self.net, 'last_joint_diagnostics'):
                diag = self.net.last_joint_diagnostics
                postfix = {
                    key: f'{value:.3g}' for key, value in diag.items()
                    if key in {'num_edges', 'avg_degree', 'mode_entropy',
                               'relation_entropy', 'average_pairwise_energy',
                               'goal_candidate_entropy', 'L_goal', 'L_mode',
                               'L_PL', 'L_diff', 'L_rank', 'L_graph',
                               'L_refine', 'graph_candidate_edges',
                               'graph_active_edges', 'graph_mean_gate',
                               'continuous_refinement_mean_delta', 'L_align_raw',
                               'L_align_soft', 'alignment_keep_confidence',
                               'alignment_changed_fraction',
                               'alignment_mean_edge_gate', 'loss_total',
                               'loss_alignment', 'loss_pair_score',
                               'loss_pair_assignment', 'loss_no_harm',
                               'loss_perm_entropy', 'loss_relation_prior',
                               'changed_fraction', 'keep_confidence',
                               'mean_pair_gate', 'permutation_entropy',
                               'sinkhorn_row_error', 'sinkhorn_col_error',
                               'avg_num_agents', 'avg_num_edges',
                               'max_component_size'}
                }
                mode_usage = [value for key, value in sorted(diag.items())
                              if key.startswith('mode_usage_')]
                relation_usage = [value for key, value in sorted(diag.items())
                                  if key.startswith('relation_usage_')]
                if mode_usage:
                    postfix['mode_use'] = '/'.join(
                        f'{value:.2f}' for value in mode_usage)
                if relation_usage:
                    postfix['rel_use'] = '/'.join(
                        f'{value:.2f}' for value in relation_usage)
                train_bar.set_postfix(postfix)
                self._accumulate_joint_diagnostics(
                    diagnostic_sums, diagnostic_counts, diag)

            # Update network weights
            # An isolated/single-agent Sparse-Energy window has no pairwise
            # trainable signal while the baseline is frozen. It is a valid
            # degeneracy, so skip only that optimizer step instead of failing.
            if loss.requires_grad:
                scaled_loss = loss / accumulation_steps
                if self.scaler.is_enabled():
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                pending_backward += 1
                if pending_backward == accumulation_steps:
                    optimizer_step(pending_backward)
                    pending_backward = 0

            del inputs, batch_id
            del seq_list, losses, loss

        # Do not discard the last incomplete accumulation group.
        optimizer_step(pending_backward, rescale_partial=True)
        if self.net.jdv2_active:
            self.args.jdv2_stage_progress = min(
                self._stage_optimizer_steps_completed /
                total_stage_optimizer_steps, 1.0)

        # update losses
        for loss_name in losses_epoch.keys():
            # mean losses over batches
            losses_epoch[loss_name] = losses_epoch[loss_name] / num_train_batches
        losses_epoch = add_dict_prefix(losses_epoch, prefix='train')
        losses_epoch['train_loss_total'] = (
            total_loss_epoch / max(num_train_batches, 1))
        self.last_train_joint_diagnostics = \
            self._finalize_joint_diagnostics(
                diagnostic_sums, diagnostic_counts)

        return losses_epoch

    @torch.no_grad()
    def _evaluate_cached_epoch(self, epoch, mode='valid'):
        """Evaluate packed cached scenes without changing metric semantics."""
        self.net.eval()
        if mode == 'valid':
            self.data_loaders[mode].dataset.set_seed_index(0)
        metrics_epoch = self.net.init_test_metrics()
        scalar_sums = {name: 0.0 for name in V4_DIAGNOSTIC_NAMES}
        scalar_counts = {name: 0 for name in V4_DIAGNOSTIC_NAMES}
        buckets = {'N_size_buckets': {}, 'component_size_buckets': {}}
        evaluate_bar = tqdm(
            self.data_loaders[mode], ascii=True, ncols=100,
            desc=f'Epoch {epoch}. {mode.title()} packs')
        total_time = 0.0
        total_scenes = 0
        total_agents = 0
        total_edges = 0
        for pack in evaluate_bar:
            inputs = self.net.prepare_cached_trajectory_pack(pack)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            coupling = self.net.cached_multiway_coupling(inputs, hard=True)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            total_time += time.perf_counter() - started
            metric_values, aligned = self.net.cached_pack_metrics(
                inputs, coupling)
            for name, values in metric_values.items():
                metrics_epoch[name].extend(values)
            diagnostics = self.net.cached_pack_diagnostics(
                inputs, coupling, aligned)
            scene_count = int(inputs['scene_ptr'].numel() - 1)
            total_scenes += scene_count
            total_agents += int(inputs['raw_future_world'].shape[0])
            total_edges += int(inputs['edge_index'].shape[1])
            for name, value in diagnostics['scalar'].items():
                if name not in scalar_sums:
                    continue
                value = float(value.detach().cpu()) \
                    if torch.is_tensor(value) else float(value)
                scalar_sums[name] += value * scene_count
                scalar_counts[name] += scene_count
            for family in buckets:
                for label, values in diagnostics[family].items():
                    count = int(values.get('count', 0))
                    if count <= 0:
                        continue
                    target = buckets[family].setdefault(
                        label, {'count': 0, '_sums': {}})
                    target['count'] += count
                    for metric_name, metric_value in values.items():
                        if metric_name == 'count':
                            continue
                        target['_sums'][metric_name] = (
                            target['_sums'].get(metric_name, 0.0) +
                            float(metric_value) * count)
            evaluate_bar.set_postfix(
                scenes=scene_count,
                agents=int(inputs['raw_future_world'].shape[0]),
                edges=int(inputs['edge_index'].shape[1]),
                seed=int(inputs['cached_seed_indices'][0]))
            del pack, inputs, coupling, metric_values, aligned, diagnostics

        evaluate_metrics = {}
        for name, values in metrics_epoch.items():
            array = np.asarray(values, dtype=np.float64)
            evaluate_metrics[name] = float(array.mean()) if array.size else 0.0
        for name in V4_DIAGNOSTIC_NAMES:
            evaluate_metrics[name] = (
                scalar_sums[name] / scalar_counts[name]
                if scalar_counts[name] else 0.0)
        finalized_buckets = {}
        for family, family_buckets in buckets.items():
            finalized_buckets[family] = {}
            for label, values in family_buckets.items():
                count = values['count']
                finalized_buckets[family][label] = {
                    'count': count,
                    **{key: total / count
                       for key, total in values['_sums'].items()},
                }
        diagnostics_dir = os.path.join(self.args.model_dir, 'diagnostics')
        os.makedirs(diagnostics_dir, exist_ok=True)
        seed_index = self.data_loaders[mode].dataset.fixed_seed_index
        suffix = (f'_seed_{int(seed_index):02d}'
                  if seed_index is not None else '')
        diagnostic_path = os.path.join(
            diagnostics_dir,
            f'{mode}_epoch_{int(epoch):03d}{suffix}.json')
        with open(diagnostic_path, 'w') as handle:
            json.dump({
                'epoch': int(epoch),
                'split': mode,
                'cached_seed_index': seed_index,
                'K': int(self.args.num_samples),
                'seed_banks_merged': False,
                'metrics': evaluate_metrics,
                'throughput': {
                    'coupler_seconds': total_time,
                    'scenes_per_second': total_scenes / max(total_time, 1e-9),
                    'agents_per_second': total_agents / max(total_time, 1e-9),
                    'total_scenes': total_scenes,
                    'total_agents': total_agents,
                    'total_edges': total_edges,
                },
                **finalized_buckets,
            }, handle, indent=2, sort_keys=True)
        if total_scenes:
            print(
                f'Packed coupling throughput: '
                f'{total_scenes / max(total_time, 1e-9):.2f} scenes/s, '
                f'{total_agents / max(total_time, 1e-9):.2f} agents/s')
        return add_dict_prefix(evaluate_metrics, prefix=mode)

    @torch.no_grad()
    def _evaluate_epoch(self, epoch, mode='valid', evaluation_seed=None,
                        metric_names=None, external_edge_cache_root=None):
        """Evaluate with an isolated, reproducible stochastic RNG stream."""
        if evaluation_seed is None:
            if mode != 'valid':
                return self._evaluate_epoch_unseeded(
                    epoch, mode=mode, metric_names=metric_names,
                    external_edge_cache_root=external_edge_cache_root)
            evaluation_seed = self.args.validation_seed
        with isolated_random_seed(
                evaluation_seed, use_cuda=self.device.type == 'cuda'):
            return self._evaluate_epoch_unseeded(
                epoch, mode=mode, metric_names=metric_names,
                external_edge_cache_root=external_edge_cache_root)

    def _evaluate_epoch_unseeded(self, epoch, mode='valid', metric_names=None,
                                 external_edge_cache_root=None):
        """
        Loop over the validation or test set once. Compute metrics and save
        output trajectories.
        """
        if self.args.use_trajectory_bank_cache:
            return self._evaluate_cached_epoch(epoch, mode)
        self.net.eval()  # evaluation mode

        # INIT LOSSES and METRICS
        metrics_epoch = (self.net.init_test_metrics() if metric_names is None
                         else {name: [] for name in metric_names})
        v4_scalar_values = {name: [] for name in V4_DIAGNOSTIC_NAMES}
        v4_buckets = {'N_size_buckets': {}, 'component_size_buckets': {}}
        diagnostic_sums = {}
        diagnostic_counts = {}

        if self.args.use_wandb and wandb is None:
            raise ImportError('wandb is required only when --use_wandb True')
        if self.args.use_wandb and wandb.run is None:
            wandb.init(settings=wandb.Settings(start_method="thread"),
                       project="GDTS", config=self.args, entity="iadc_erwinsun",
                       group=f"{self.args.dataset}",
                       job_type=f"{self.args.test_set}",
                       tags=None, name=None)

        # Progress bar
        evaluate_bar = tqdm(self.data_loaders[mode], ascii=True, ncols=100,
                            desc=f'Epoch {epoch}. {mode.title()} batches')


        # loop over batches
        total_time = 0
        num_input = 0
        for batch_index, (batch_data, batch_id) in enumerate(evaluate_bar):

            inputs, seq_list = self.net.prepare_inputs(batch_data, batch_id)
            del batch_data
            
            # compute metric_mask
            metric_mask = compute_metric_mask(seq_list)
            st = time.time()
            with self._autocast_context():
                all_output, all_aux_outputs = self.net.forward(
                    inputs, if_test=True)
            if (self.net.jdv2_active and
                    self.args.training_stage == 'joint_goal' and
                    self.args.joint_diagnostics):
                # Validation predictions retain the deployed stochastic
                # policy. This separate no-gradient teacher pass records the
                # latent mixture health without changing sampled outputs.
                with self._autocast_context():
                    self.net._jdv2_goal_losses(inputs)
            if (external_edge_cache_root is not None and
                    'Relative_Motion_Error' in metrics_epoch and
                    'edge_index' not in all_aux_outputs):
                edge_record = load_cache_record(os.path.join(
                    external_edge_cache_root, mode,
                    f'{batch_index:06d}.pt'), allow_future_supervision=False)
                all_aux_outputs['edge_index'] = edge_record[
                    'edge_index'].to(self.device)
            if self.net.jdv2_active and self.args.joint_diagnostics:
                self._accumulate_joint_diagnostics(
                    diagnostic_sums, diagnostic_counts,
                    self.net.last_joint_diagnostics)
            total_time = total_time + time.time() - st
            num_input = num_input + inputs["abs_pixel_coord"].shape[1]

            # update metrics
            for metric_name in metrics_epoch.keys():
                # print('metric_name: ', metric_name)
                metrics_epoch[metric_name].extend(
                    self.net.compute_model_metrics(
                        metric_name=metric_name,
                        predictions=all_output,
                        metric_mask=metric_mask,
                        all_aux_outputs=all_aux_outputs,
                        inputs=inputs,
                        obs_length=self.args.obs_length,
                    ))

            if self.args.trajectory_coupling == 'multiway_v4':
                diagnostics = self.net.multiway_batch_diagnostics(
                    all_aux_outputs, inputs, metric_mask)
                for name, value in diagnostics['scalar'].items():
                    if name not in v4_scalar_values:
                        continue
                    if torch.is_tensor(value):
                        value = float(value.detach().cpu())
                    v4_scalar_values[name].append(float(value))
                for family in ('N_size_buckets',
                               'component_size_buckets'):
                    for label, values in diagnostics[family].items():
                        count = int(values.get('count', 0))
                        if count <= 0:
                            continue
                        target = v4_buckets[family].setdefault(
                            label, {'count': 0, '_sums': {}})
                        target['count'] += count
                        for metric_name, metric_value in values.items():
                            if metric_name == 'count':
                                continue
                            target['_sums'][metric_name] = (
                                target['_sums'].get(metric_name, 0.0) +
                                float(metric_value) * count)

            del inputs, batch_id, all_output, all_aux_outputs
            del seq_list, metric_mask


        # update metrics
        evaluate_metrics = {}
        for metric_name in metrics_epoch.keys():
            array_metric = np.array(metrics_epoch[metric_name])
            evaluate_metrics[metric_name] = array_metric.mean()
            if "goal" in metric_name:
                evaluate_metrics[metric_name + '_std'] = array_metric.std()
        if self.args.trajectory_coupling == 'multiway_v4':
            for name, values in v4_scalar_values.items():
                evaluate_metrics[name] = (
                    float(np.mean(values)) if values else 0.0)
            finalized_buckets = {}
            for family, buckets in v4_buckets.items():
                finalized_buckets[family] = {}
                for label, values in buckets.items():
                    count = values['count']
                    finalized_buckets[family][label] = {
                        'count': count,
                        **{
                            key: total / count
                            for key, total in values['_sums'].items()
                        },
                    }
            diagnostics_dir = os.path.join(
                self.args.model_dir, 'diagnostics')
            os.makedirs(diagnostics_dir, exist_ok=True)
            diagnostic_path = os.path.join(
                diagnostics_dir, f'{mode}_epoch_{int(epoch):03d}.json')
            with open(diagnostic_path, 'w') as handle:
                json.dump({
                    'epoch': int(epoch),
                    'split': mode,
                    'model_selection_split': (
                        self.args.model_selection_split),
                    'metrics': {
                        key: float(value)
                        for key, value in evaluate_metrics.items()},
                    **finalized_buckets,
                }, handle, indent=2, sort_keys=True)
        if self.net.jdv2_active:
            self.last_valid_joint_diagnostics = \
                self._finalize_joint_diagnostics(
                    diagnostic_sums, diagnostic_counts)
        evaluate_metrics = add_dict_prefix(evaluate_metrics, prefix=mode)
        # print('total_time (ms): ', total_time*1000)
        # print('num_input: ', num_input)
        if num_input:
            print('average_time for 1 input(ms/input): ',
                  total_time * 1000 / num_input)
        return evaluate_metrics

    def _set_device(self):
        """
        Set the device for the experiment. GPU if available, else CPU.
        """
        # torch.cuda.is_available() already checked in parsed args
        device = torch.device(self.args.device)
        print('\nUsing device:', device)

        # Additional info when using cuda
        if device.type == 'cuda':
            print('Number of available GPUs:', torch.cuda.device_count())
            print('GPU name:', torch.cuda.get_device_name(0))
            print('Cuda version:', torch.version.cuda)
        print()
        return device
