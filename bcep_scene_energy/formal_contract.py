"""Fixed single-run identity and pre-construction authorization gate."""
import contextlib,fcntl,hashlib,json,os
from pathlib import Path

SOURCE=Path(__file__).resolve().parents[1]
BASE='b51966c36ef2ff43246c9daac76666f6814bc2c8'
EXPERIMENT='zara2_ssec_energy_stage2_20261121'
AUDIT='zara2_ssec_energy_stage2_readiness_20261121'
READY='READY_FOR_SEPARATE_TRAINING_AUTHORIZATION'


def paths():
    assets=SOURCE.parent/'RSJG_EBJD_assets/experiments'
    return dict(run=assets/EXPERIMENT,audit=assets/AUDIT,formal=assets/'bcep_ebjd_v2_integrated',
        v3=assets/'zara2_bcep_v3_stage2_20261120',candidate_cost=assets/'zara2_costproxy_stage2_20261119',
        sharedworld=assets/'zara2_sharedworld_stage2_onerun_20261119')


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_manifest():
    names=['bcep','ebjd','bcep_v2','bcep_v3','bcep_scene_energy','bcep_joint','bcep_formal','bcep_confirmation','selector_control']
    return {str(p.relative_to(SOURCE)):file_sha(p) for folder in names for p in sorted((SOURCE/folder).rglob('*.py'))}


def require_authorization(command,frozen_sha,roots=None):
    """Only stdlib/file reads here: reject before data/model/optimizer construction."""
    if command!='train':return None
    if os.environ.get('RUN_ONE_ZARA2_SSEC')!='true' or not frozen_sha:
        raise PermissionError('require RUN_ONE_ZARA2_SSEC=true and --frozen-sha; no training authorized')
    roots=paths() if roots is None else roots
    path=roots['run']/'FROZEN_CONFIG.json'
    if not path.is_file() or file_sha(path)!=frozen_sha:
        raise PermissionError('frozen configuration FILE SHA256 mismatch; construction forbidden')
    config=json.loads(path.read_text())
    if (config.get('base_source'),config.get('experiment'),config.get('variant'))!=(BASE,EXPERIMENT,'energy_only'):
        raise PermissionError('not the single authorized Energy-only ZARA2 identity')
    if config['implementation_files']!=source_manifest():raise PermissionError('frozen source mismatch')
    ready_path=roots['audit']/'READINESS.json'
    if not ready_path.is_file():raise PermissionError('readiness receipt missing')
    ready=json.loads(ready_path.read_text())
    if (ready.get('status')!=READY or ready.get('frozen_file_sha256')!=frozen_sha
        or ready.get('real_optimizer_updates')!=0 or ready.get('synthetic_optimizer_updates')!=0):
        raise PermissionError('matching zero-update readiness required')
    return config


@contextlib.contextmanager
def locks(roots):
    roots['run'].mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        entries=[(roots['run']/'run.lock','a'),(roots['formal']/'controller.lock','r'),
            (roots['v3']/'run.lock','r'),(roots['candidate_cost']/'trial.lock','r'),(roots['sharedworld']/'run.lock','r')]
        for path,mode in entries:
            f=stack.enter_context(path.open(mode));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def model_hash(model):
    from bcep.contracts import canonical_hash
    return canonical_hash({n:hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n,t in model.state_dict().items()})


def freeze(inputs):
    from bcep.contracts import canonical_hash
    from bcep_joint.common import seed_all
    from .model import build
    seed_all(3201);model,origin=build(inputs.config,variant='energy_only')
    manifest=source_manifest()
    frozen=dict(schema='ssec-energy-only-zara2-stage2-v1',base_source=BASE,experiment=EXPERIMENT,variant='energy_only',
        allow_real_training=False,real_optimizer_updates=0,synthetic_optimizer_updates=0,auto_queue=False,
        config=inputs.config,recipe=inputs.recipe,implementation_files=manifest,implementation_hash=canonical_hash(manifest),
        provider=inputs.provider,split_sha256=canonical_hash(inputs.split),
        bank_sha256={b.purpose:b.hash for b in inputs.banks},counts=inputs.counts,initial_model_sha256=model_hash(model),
        candidate_seed=2035,training_seed=3201,K=20,DDIM_steps=20,max_real_updates_if_separately_authorized=16800,
        selector='all 590 selection_dev H-JADE only; exact ties earlier epoch',
        retention='all complete epochs + best + last + Top5 distinct epochs with hashes; no partial epoch marked complete',
        runtime=dict(precision='FP32',deterministic=True,TF32=False,SDPA='math'),
        candidate_identity='verified original bank slots; native augmentation/compaction preserve candidate axis; xy canonicalization inside SSEC',
        telemetry='same-forward I/H endpoint metrics, common-energy decisions, preclip gradients, post-step actual parameter changes; no new losses',
        activation='RUN_ONE_ZARA2_SSEC=true AND exact FROZEN_CONFIG.json file SHA256 AND matching readiness; separate authorization only',
        restrictions=['ZARA2 joint only','one seed3201','no test access','no old queue','no CSEE','no provider training','no follow-up launch'])
    frozen['identity']=dict(stage='joint',fold='zara2',experiment=EXPERIMENT,base=BASE,implementation=frozen['implementation_hash'],
        config=canonical_hash(inputs.config),recipe=canonical_hash(inputs.recipe),split=frozen['split_sha256'],banks=frozen['bank_sha256'],initial=frozen['initial_model_sha256'])
    return frozen,origin
