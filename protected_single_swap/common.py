from pathlib import Path
from bcep.contracts import digest,canonical_hash
from bcep_joint.common import read,write
from joint_flow_fourfold.scoring import csvwrite
SOURCE=Path(__file__).resolve().parents[1]
ROOT=SOURCE.parent
RUN=ROOT/'RSJG_EBJD_assets/experiments/hotel_protected_single_swap_20261124'
CACHE=ROOT/'RSJG_EBJD_assets/experiments/hotel_endpoint_diagnostic_20261124'
DOC=SOURCE/'docs/research/protected_single_swap/hotel_20261124'
BASE='2b6bad80abf47770c4c98f9b1914e935d9626df4'
BRANCH='experiment/hotel-protected-single-swap-20261124'
BUDGET=.005
REFERENCE=dict(minFDE=.4497455630738098,JFDE=.7230634318033862)

def sources():
    return {str(p.relative_to(SOURCE)):digest(p) for folder in ('protected_single_swap','hotel_endpoint_diagnostic',
        'joint_flow_dependency','joint_marginal_residual','joint_flow_fourfold','bcep','bcep_v2',
        'bcep_joint','bcep_formal','ebjd','scene_pair_selector') for p in sorted((SOURCE/folder).rglob('*.py'))}
