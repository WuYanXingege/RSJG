"""Observed-only audit of native CPU versus canonical CUDA H0 permutations."""
import copy,time
import numpy as np
import torch
from bcep_joint.common import write
from bcep_v2.runtime import configure_cuda
from .run import fold_assets,observations_only,ORIGINAL_RUN


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    inputs,model,identity,_=fold_assets('zara1');gpu=copy.deepcopy(model).cuda()
    records=[]
    with observations_only():
        for pos,index in enumerate(inputs.split['selection_dev_indices'][:24]):
            batch=inputs.data.batch([index],'selection_dev',truth=False)
            goals=inputs.banks['selection_dev'].batch([index],batch)
            c=model.encode_context(batch.observed,batch.semantic_maps,batch.valid)
            _,cp,cs,ci=model.coupling.assign(c,goals,model.bridge.representation.goal_scale,straight_through=False,return_initial=True)
            configure_cuda();x=batch.to('cuda')
            c=gpu.encode_context(x.observed,x.semantic_maps,x.valid)
            _,gp,gs,gi=gpu.coupling.assign(c,goals.cuda(),gpu.bridge.representation.goal_scale,straight_through=False,return_initial=True)
            with np.load(ORIGINAL_RUN/'zara1/native_A'/f'{pos:04d}.npz',allow_pickle=False) as z:stored=z['H0_permutation'].copy()
            records.append(dict(window=f'dev_{pos+1:04d}',n=int(batch.valid.sum()),
                cpu_stored_equal=bool(np.array_equal(cp.numpy(),stored)),gpu_stored_equal=bool(np.array_equal(gp.cpu().numpy(),stored)),
                cpu_gpu_initial_slots_different=int((ci!=gi.cpu()).sum()),cpu_gpu_final_slots_different=int((cp!=gp.cpu()).sum()),
                final_score_max_abs=float((cs-gs.cpu()).abs().max())))
            torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_mem_efficient_sdp(True);torch.backends.cuda.enable_cudnn_sdp(True)
    # Do not replace the audit receipt referenced by a launched recovery.
    destination=ORIGINAL_RUN/'DEVICE_AUDIT_24.json'
    if destination.exists():raise FileExistsError(destination)
    write(destination,dict(truth_access=0,records=records,identity=identity,unix=time.time()))


if __name__=='__main__':main()
