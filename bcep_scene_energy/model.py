"""Same random v2 bridge/S0 initialization; no trained-history initialization input."""
import hashlib
import torch
from bcep_v2.joint import build_joint
from .coupling import SceneEnergyCoupling


def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def build(config,*,variant='energy_only',initialization_seed=3201,rejection_gradient='proposal_st'):
    if variant not in ('energy_only','energy_csee'):raise ValueError('explicit energy_only or energy_csee required')
    if config.get('pair_objective','final_coordinate')!='final_coordinate':raise ValueError('final_coordinate only')
    if config.get('allow_real_training') is not False:raise PermissionError('this release permits zero training updates')
    model=build_joint(config)
    original={k:digest(v) for k,v in model.state_dict().items()}
    old=model.coupling.state_dict()
    removed={k for k in old if k.startswith(('project.','compatibility.'))}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(initialization_seed)
        new=SceneEnergyCoupling()
        new.load_state_dict({k:v for k,v in old.items() if k not in removed},strict=True)
        new.enable_energy(csee=variant=='energy_csee',rejection_gradient=rejection_gradient)
    model.coupling=new
    kept={k:v for k,v in original.items() if k not in {'coupling.'+x for x in removed}}
    assert all(digest(model.state_dict()[k])==v for k,v in kept.items())
    assert len(model.bridge.denoiser.blocks)==6 and not hasattr(model.bridge.denoiser,'grg_ctf')
    origin=dict(variant=variant,initialization='caller-seeded random v2 downstream + locally seeded phi/rho; no historical weights',
        caller_torch_initial_seed=torch.initial_seed(),
        new_module_seed=initialization_seed,rejection_gradient=rejection_gradient,old_tensor_hashes=kept,
        removed_old_feedback_keys=sorted('coupling.'+x for x in removed),
        new_tensors={k:dict(sha256=digest(v),shape=list(v.shape)) for k,v in model.state_dict().items() if k not in kept})
    return model,origin
