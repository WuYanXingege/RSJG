"""Process-local CPU preparation guard; deliberately no torch import at import time."""
import os
import time
from collections import Counter
from pathlib import Path


def install():
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Explicit CUDA_VISIBLE_DEVICES empty required')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    counts=Counter();allowed={};started=time.monotonic()
    def reject(category):
        def denied(*a,**kw):
            counts[category+'_attempts']+=1
            counts[category+'_intercepted']+=1
            raise RuntimeError('CPU preparation guard: '+category)
        return denied
    names=['_lazy_init','init','is_available','device_count','synchronize','get_device_name',
           'get_device_properties','current_device','set_device','is_bf16_supported']
    for name in names:setattr(torch.cuda,name,reject('cuda'))
    native=[]
    for name in ('_cuda_getDeviceCount','_cuda_init','_cuda_setDevice','_cuda_getDeviceProperties'):
        if hasattr(torch._C,name):
            setattr(torch._C,name,reject('cuda_native'));native.append(name)
    torch.nn.Module.__init__=reject('model_constructor')
    torch.nn.Module._call_impl=reject('model_forward')
    torch.Tensor.backward=reject('backward')
    torch.autograd.backward=reject('backward')
    for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD):
        cls.__init__=reject('optimizer_constructor');cls.step=reject('optimizer_update')
    original_load=torch.load
    def load(p,*a,**kw):
        from src.p2_protocol import file_hash
        resolved=str(Path(p).resolve()) if isinstance(p,(str,Path)) else None
        if resolved not in allowed or kw.get('weights_only') is not True or str(kw.get('map_location'))!='cpu':
            return reject('unauthorized_deserialization')()
        if file_hash(resolved)!=allowed[resolved]:return reject('load_hash')()
        counts['authorized_observation_loads']+=1
        return original_load(p,*a,**kw)
    torch.load=load
    def permit(p,sha):allowed[str(Path(p).resolve())]=sha
    def report():
        return dict(pid=os.getpid(),ppid=os.getppid(),entry='CPU observation preparation',
            seconds=time.monotonic()-started,torch_threads=torch.get_num_threads(),
            counts=dict(counts),forwarded_cuda_calls=0,guarded_cuda_names=names,native_names=native,
            coverage='Python calls and listed torch C entrypoints after torch import',
            pre_torch_import_and_unlisted_native_cuda_coverage='UNKNOWN',
            real_model_calls=0,backward_calls=0,optimizer_updates=0,gpu_compute_calls=0)
    return permit,report
