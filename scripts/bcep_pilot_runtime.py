"""One-run FP32 precision policy, shared by gate and authorized pilot."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import importlib.util,json,platform,subprocess
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]

def configure():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.mha.set_fastpath_enabled(False)
    return dict(dtype='float32',amp=False,compile=False,tf32_matmul=False,tf32_cudnn=False,
        cudnn_benchmark=False,cudnn_deterministic=True,deterministic_algorithms=True,
        sdp='math',mha_fastpath=False,cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        cpu_threads=2,interop_threads=1)

def script_module(name):
    path=ROOT/name;spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def environment():
    return dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),
        gpu_identity=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip())

def assert_gpu_idle():
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if lines:raise RuntimeError('BLOCKED_RESOURCE: existing GPU compute processes: '+lines)
