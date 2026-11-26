"""Fixed identities and paths for the direct K20 experiment."""
from pathlib import Path

from bcep.contracts import canonical_hash, digest
from bcep_joint.common import read, write
from joint_flow_fourfold.common import immutable, source_hashes, status

SOURCE = Path(__file__).resolve().parents[1]
ASSETS = SOURCE.parent / 'RSJG_EBJD_assets/experiments'
RUN = ASSETS / 'direct_k20_joint_20261126'
DOC = SOURCE / 'docs/research/continuous_joint_flow/direct_k20_joint_20261126'
BASE = '3296fc5968463ed14bf43265c064b3856d55ed30'
MIX_REFERENCE = 'dfe8a4a700bf636c56178c4d7f2ae7fff7ca542b'
BRANCH = 'experiment/direct-k20-joint-20261126'
FOLDS = ('zara1', 'hotel', 'univ', 'eth', 'zara2')
SEEDS = tuple(range(2035, 2040))
PURPOSE = 'heldout_confirmation'
GROUPS = ('all', 'single', 'multi', 'N_ge_9')
METRICS = ('JADE', 'JFDE', 'minADE', 'minFDE')
MODELS = ('A', 'B', 'new')
ATOL = RTOL = 1e-6
ENVIRONMENT = dict(
    CUBLAS_WORKSPACE_CONFIG=':4096:8',
    OMP_NUM_THREADS='2',
    MKL_NUM_THREADS='2',
    OPENBLAS_NUM_THREADS='1',
    CUDA_VISIBLE_DEVICES='0',
)


def runtime_sources():
    return source_hashes() | {
        str(path.relative_to(SOURCE)): digest(path)
        for path in sorted((SOURCE / 'direct_k20').glob('*.py'))
    }


def window_key(split, index):
    return canonical_hash(dict(fold=split['fold'], member=split['members'][index]))


def groups_for(n):
    if n == 1:
        return ('all', 'single')
    return ('all', 'multi', 'N_ge_9') if n >= 9 else ('all', 'multi')
