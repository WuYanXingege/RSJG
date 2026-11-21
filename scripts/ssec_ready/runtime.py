from common import *
import scipy,platform
setup(cuda=True)
write(PUBLIC/'RUNTIME.json',dict(torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,scipy=scipy.__version__,
    python=platform.python_version(),device=torch.cuda.get_device_name(),total_device_bytes=torch.cuda.get_device_properties(0).total_memory,
    deterministic=torch.are_deterministic_algorithms_enabled(),TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
    TF32_cudnn=torch.backends.cudnn.allow_tf32,autocast=torch.is_autocast_enabled(),threads=torch.get_num_threads(),optimizer_guard=GUARD))
print('Runtime recorded; no model or optimizer constructed')
