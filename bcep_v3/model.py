"""Strict v2 import, local new-parameter initialization, no permissive loads."""
import hashlib
import torch
from bcep_v2.joint import build_joint
from .coupling import HCRACoupling
from .gate import GeometryReliabilityGate

NEW_PREFIXES = ('coupling.csee.', 'coupling.hcra.', 'bridge.denoiser.grg_ctf.')


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def build(config, *, variant='v3', v2_state=None, initialization_seed=3201):
    if variant not in ('v2_compatible', 'v3'):
        raise ValueError('unknown architecture')
    # The anchor has no pair_objective switch; its effective default is final_coordinate.
    if config.get('pair_objective', 'final_coordinate') != 'final_coordinate':
        raise ValueError('v3 retains final_coordinate only')
    model = build_joint(config)
    if v2_state is not None:
        # Exact key/shape validation happens before allocating ANY added parameter.
        model.load_state_dict(v2_state, strict=True)
    prior = {k: tensor_sha(v) for k, v in model.state_dict().items()}
    if variant == 'v3':
        # New tensors have a local seed; old/global RNG and initial parameters unchanged.
        with torch.random.fork_rng(devices=[]):
            # New modules are constructed on CPU. manual_seed() would also reset
            # CUDA generators, which are deliberately outside this local fork.
            torch.random.default_generator.manual_seed(initialization_seed)
            coupling = HCRACoupling()
            coupling.load_state_dict(model.coupling.state_dict(), strict=True)
            coupling.enable_v3()
            model.coupling = coupling
            model.bridge.denoiser.grg_ctf = GeometryReliabilityGate()
    current = model.state_dict()
    assert all(tensor_sha(current[k]) == h for k, h in prior.items())
    extra = set(current) - set(prior)
    assert all(k.startswith(NEW_PREFIXES) for k in extra)
    origin = dict(variant=variant, old_tensor_count=len(prior), new_tensor_count=len(extra),
                  initialization_seed=initialization_seed, strict_legacy_load=True,
                  old={k:dict(sha256=h, shape=list(current[k].shape)) for k,h in prior.items()},
                  new={k:dict(sha256=tensor_sha(current[k]), shape=list(current[k].shape),
                             origin='local_seed_new_module_zero_residual_heads') for k in sorted(extra)})
    return model, origin
