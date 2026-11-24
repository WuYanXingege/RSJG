"""FP64 native-endpoint distance batch; no candidate-head approximation."""
import torch


@torch.no_grad()
def directed_distance_batch(old, new, valid):
    """FP64 old-to-new max-min mean-agent distance, one value per new set.

    old: [20,N,2], new: [C,20,N,2]. Excludes padded agents. Kept here to
    avoid copying every rejected endpoint set to CPU during screening.
    """
    old = torch.as_tensor(old, device=new.device).to(torch.float64)
    valid = torch.as_tensor(valid, device=new.device, dtype=torch.bool)
    assert old.shape == new.shape[1:] and valid.any()
    old = old[:, valid]
    candidate = new[:, :, valid].to(torch.float64)
    distances = (old[None, :, None] - candidate[:, None]).norm(dim=-1).mean(-1)
    return distances.amin(dim=2).amax(dim=1)
