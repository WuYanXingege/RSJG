"""Read checkpoint metadata; independent scalar counterexamples. No model imports/forwards.

Run from repository root with CUDA_VISIBLE_DEVICES='' and --emit.
Output is written by apply_patch by the review author, not by this script.
"""
import collections
import hashlib
import json
import math
import pathlib
import subprocess
import torch

ROOT = pathlib.Path.cwd()
PATH = ROOT / 'outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_a_no_z_full_seed2035/saved_models/best_model.pt'
ckpt = torch.load(PATH, map_location='cpu', weights_only=False)
state = ckpt['model_state_dict']
layers = collections.defaultdict(list)
for key, value in state.items():
    layers[key.rsplit('.', 1)[0]].append({'key': key, 'shape': list(value.shape), 'numel': value.numel(), 'dtype': str(value.dtype)})
buffers = ['var_sched.', 'diffnet.pos_emb.pe']
parameters = collections.Counter()
for key, value in state.items():
    if not any(key.startswith(p) for p in buffers):
        parameters[key.split('.')[0]] += value.numel()
lin = lambda a, b: (a + 1) * b
gru = lambda a, h: 3*h*a + 3*h*h + 6*h
v4 = {
    'trajectory_encoder': lin(7,128)+lin(128,128)+gru(128,128)+256+lin(128,128)+256,
    'relation_residual': lin(282,128)+lin(128,128)+lin(128,4),
    'candidate_energy_encoder': lin(128,128)+256,
    'edge_context_encoder': lin(270,128)+lin(128,128)+256,
    'relation_embedding': 4*128,
    'factor_head': lin(128,8),
    'pair_gate': lin(286,128)+lin(128,64)+lin(64,1),
    'dormant_direct_energy': lin(282,128)+lin(128,128)+lin(128,4),
    'dormant_edge_gate': lin(274,128)+lin(128,1),
    'component_confidence': lin(7,32)+lin(32,1),
}
# Jensen gap: target k=0; neighbor energy columns give logits difference 0 or -4.
softplus = lambda x: math.log1p(math.exp(x))
checks = {
    'mean_energy_CE': softplus(2),
    'expected_conditional_CE': .5*(softplus(0)+softplus(4)),
    'gauge_original_score': 3+5-7,
    'gauge_compensated_score': (3-2)+(5-4)-(7-2-4),
    'synchronous_cycle': {'cost': [[0,1],[1,0]], 'states': [[0,1],[1,0],[0,1]], 'joint_costs': [1,1,1]},
    'synchronous_worsening': {'cost': [[1,0],[0,2]], 'states': [[0,0],[1,1]], 'joint_costs': [1,2]},
    'trunk_denoiser_calls': len(range(100,30,-1)),
    'branch_denoiser_calls': 20*len(range(15,21)),
    'total_denoiser_calls': len(range(100,30,-1))+20*len(range(15,21)),
    'normal_draw_calls': 1+20*len(range(15,21)),
    'geometry6_added_parameters': lin(6,32)-lin(4,32),
    'chunk256_relation_fp32_bytes': 256*21*21*4*4,
    'v4_pair_hidden128_fp32_bytes_E256': 256*20*20*128*4,
}
assert checks['mean_energy_CE'] != checks['expected_conditional_CE']
assert checks['gauge_original_score'] == checks['gauge_compensated_score']
result = {
    'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'checkpoint_sha256': hashlib.sha256(PATH.read_bytes()).hexdigest(),
    'checkpoint_epoch': ckpt['epoch'],
    'checkpoint_state_layers': [{'name': key, 'tensors': value, 'stored_elements': sum(v['numel'] for v in value)} for key,value in layers.items()],
    'module_registered_parameters_excluding_buffers': dict(parameters),
    'stage_a_trainable': sum(parameters[k] for k in ['social_encoder','relation_inference','jdv2_future_teacher','jdv2_unary','jdv2_dynamic_relation','jdv2_joint_energy']),
    'base_registered_parameters': sum(parameters[k] for k in ['goal_module','registrar','diffnet']),
    'unary_head_nonzero': {k: int(torch.count_nonzero(v)) for k,v in state.items() if k.startswith('jdv2_unary.residual_head.2')},
    'corrector_head_nonzero': {k: int(torch.count_nonzero(v)) for k,v in state.items() if k.startswith('jdv2_corrector.output.2')},
    'v4_static_parameters': v4, 'v4_registered_total': sum(v4.values()),
    'v4_active_total': sum(v4.values())-v4['dormant_direct_energy']-v4['dormant_edge_gate'],
    'math_checks': checks,
    'execution': {'device':'cpu', 'model_imports':0, 'model_forwards':0, 'exact_payloads_read':0, 'solver_calls':0},
}
print(json.dumps(result, ensure_ascii=False, indent=2))
