import csv
from pathlib import Path
import numpy as np
from bcep.contracts import canonical_hash, digest
from bcep_joint.common import read, write
from joint_flow_fourfold.common import immutable

SOURCE = Path(__file__).resolve().parents[1]
ROOT = SOURCE.parent
OLD_RUN = ROOT/'RSJG_EBJD_assets/experiments/residual_threefold_frozen_test_20261124'
RUN = ROOT/'RSJG_EBJD_assets/experiments/residual_threefold_five_seed_20261124'
PRIOR = ROOT/'RSJG_EBJD_assets/experiments/joint_marginal_residual_20261124'
OLD_DOC = SOURCE/'docs/research/joint_marginal_residual/frozen_test_20261124'
DOC = SOURCE/'docs/research/joint_marginal_residual/five_seed_test_20261124'
DEV_DOC = SOURCE/'docs/research/joint_marginal_residual/20261124'
BASE = '09c52eddc7b35a03bd3c7d589181097e08779e88'
BRANCH = 'experiment/residual-threefold-five-seed-20261124'
FOLDS = ('zara1', 'zara2', 'hotel')
EXPECTED = {'zara1': (6, 21, 705, 2356, 28, 288),
            'zara2': (35, 16, 998, 5910, 214, 2197),
            'hotel': (19, 15, 445, 1197, 0, 0)}
PURPOSE = 'heldout_confirmation'
METRICS = ('JADE', 'JFDE', 'minADE', 'minFDE')
GROUPS = ('all', 'N_eq_1', 'N_2_to_8', 'N_ge_9')
SEEDS = (2035, 2036, 2037, 2038, 2039)
NEW_SEEDS = (2036, 2037, 2038, 2039)
PACKAGES = ('residual_frozen_test','joint_marginal_residual','bcep_v2','bcep_joint',
            'bcep_formal','bcep','ebjd','joint_flow_dependency','joint_flow_fourfold',
            'joint_flow_frozen_test','scene_pair_selector')


def sources():
    return {str(p.relative_to(SOURCE)): digest(p) for name in PACKAGES
            for p in sorted((SOURCE/name).rglob('*.py'))}


def groups_for(n):
    assert n > 0
    return ('all', 'N_eq_1' if n == 1 else 'N_2_to_8' if n < 9 else 'N_ge_9')


def csvwrite(path, rows):
    assert rows
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def historical(fold):
    base = SOURCE/'docs/research/continuous_joint_flow'
    if fold == 'zara2':return base/'zara2_frozen_test_20261122/PER_SEED.csv'
    return base/'zara1_hotel_frozen_test_20261124'/fold/'PER_SEED.csv'


def arrays(path, n):
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'B', 'Bstar', 'raw_ids', 'valid'}
        values = {k: z[k].copy() for k in z.files}
    for k in ('B','Bstar'):
        a = values[k]
        assert a.shape == (20,n,12,2) and a.dtype == np.float32 and np.isfinite(a).all()
    ids = values['raw_ids']
    assert ids.shape == (20,) and ids.dtype == np.int64
    assert len(np.unique(ids)) == 20 and (ids >= 0).all() and (ids < 256).all()
    assert values['valid'].shape == (n,) and values['valid'].dtype == np.bool_ and values['valid'].all()
    return values


def expected_row(fold, seed, position, index, split, config):
    assert seed in NEW_SEEDS
    member = split['members'][index]
    return dict(fold=fold, window=f'test_{position:04d}', index=index,
                n=len(member['agent_ids']), seed=seed,
                member_sha256=canonical_hash(member),
                file=f'{fold}/seed_{seed}/predictions/test_{position:04d}.npz',
                source_config_sha256=canonical_hash(config))


def validate_row(row, expected, root=RUN):
    assert {k: row[k] for k in expected} == expected
    path = root/row['file']
    assert digest(path) == row['sha256'], 'prediction checksum mismatch'
    arrays(path, row['n'])
    assert row['ddim_calls'] == 1 and row['ablation'] is False
