from pathlib import Path
from bcep.contracts import digest,canonical_hash
from bcep_joint.common import read,write
from joint_flow_fourfold.scoring import csvwrite
SOURCE=Path(__file__).resolve().parents[1]
ROOT=SOURCE.parent
RUN=ROOT/'RSJG_EBJD_assets/experiments/hotel_endpoint_diagnostic_20261124'
DOC=SOURCE/'docs/research/hotel_endpoint_diagnostic/20261124'
BASE='aa7057389e71a46dbe4b89123a435dafb4ab70b3'
BRANCH='experiment/hotel-endpoint-selection-diagnostic-20261124'
REFERENCE=dict(raw256_minFDE=0.1736822056510894,selected20_minFDE=0.45528425735247885,
               corrected20_minFDE=0.449745562799765,corrected20_JFDE=0.723063430501159)
PRIOR_REPLAY=ROOT/'RSJG_EBJD_assets/experiments/joint_marginal_residual_20261124/hotel/head/observed_replay'

def sources():
    return {str(p.relative_to(SOURCE)):digest(p) for folder in ('hotel_endpoint_diagnostic','joint_flow_dependency',
        'joint_marginal_residual','joint_flow_fourfold','bcep','bcep_v2','bcep_joint','bcep_formal','ebjd','scene_pair_selector')
        for p in sorted((SOURCE/folder).rglob('*.py'))}
