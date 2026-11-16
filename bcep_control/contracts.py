"""Kind-specific provenance; original BCEP checkpoint validation remains intact."""
import hashlib
from pathlib import Path
from bcep.contracts import WORKSPACE, code_identity, digest, require


def control_code_identity():
    paths = sorted((WORKSPACE/'bcep_control').glob('*.py'))
    paths += [WORKSPACE/'scripts/train_mdn_control.py', WORKSPACE/'scripts/bcep_postpilot_readonly.py']
    return dict(frozen_bcep=code_identity(), control={str(p.relative_to(WORKSPACE)): digest(p) for p in paths})


def checkpoint_identity(contract_path, split_path):
    return dict(contract=digest(contract_path), split=digest(split_path), source=control_code_identity(),
                model_kind='mdn8_full_covariance_v1', weights_origin='random_initialization', teacher_weights=False)


def validate_checkpoint(state, contract_path, split_path):
    require(state.get('schema') == 'mdn-control-pilot-v1', 'not an MDN control checkpoint')
    require(state.get('identity') == checkpoint_identity(contract_path, split_path), 'MDN identity mismatch')
    require(state.get('selection_metric_name') == 'dev_nll_final_agent_micro', 'MDN selector mismatch')
    require(state.get('completed_epoch', 0) >= 1 and state.get('update', 0) >= 1, 'no complete selected state')


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        x = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()
