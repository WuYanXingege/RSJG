"""Fixed FP32 policy; retain original CUDA pooling/grid-sample gradients."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bcep_pilot_runtime import configure as provider_configure,assert_gpu_idle,environment
import torch

def configure():
    policy=provider_configure()
    # Random U-Net/CTF training requires CUDA adaptive pooling and grid sampling
    # backwards. PyTorch has no strictly deterministic kernels for these ops.
    # Preserve their existing semantics; never substitute an approximate network.
    torch.use_deterministic_algorithms(True,warn_only=True)
    policy.update(deterministic_algorithms='warn_only',known_nondeterministic_backwards=['adaptive_avg_pool2d','grid_sampler_2d'],bitwise_cuda_resume_claim=False)
    return policy
