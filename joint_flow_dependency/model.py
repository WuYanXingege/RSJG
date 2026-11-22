"""One shared two-layer edge MLP; one factorization for likelihood and sampling."""
import torch
from torch import nn
from bcep_v2.joint import neighbors


def undirected_edges(observed):
    n = len(observed)
    valid = torch.ones(1, n, dtype=torch.bool, device=observed.device)
    _, i, j = neighbors(observed[None], valid)
    code = torch.unique(torch.minimum(i, j)*n+torch.maximum(i, j), sorted=True)
    return torch.stack((code//n, code % n), -1)


class EdgeDependence(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(260, 64), nn.SiLU(), nn.Linear(64, 4))
        nn.init.normal_(self.net[-1].weight, std=0.001)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([0.02, 0., 0., 0.02]))

    def loadings(self, observed, context, edges=None):
        edges = undirected_edges(observed) if edges is None else edges
        i, j = edges.unbind(-1)
        position = observed[:, -1]
        velocity = (observed[:, -1]-observed[:, -4])/1.2
        relative = torch.cat((position[j]-position[i], velocity[j]-velocity[i]), -1)
        left = torch.cat((context[i], context[j], relative), -1)
        right = torch.cat((context[j], context[i], -relative), -1)
        b = self.net(torch.stack((left, right), 1)).reshape(-1, 2, 2, 2)
        return edges, b

    def forward(self, observed, context, edges=None):
        """F satisfies z=F*[epsilon,xi], R=F F^T, with exact block-I in real arithmetic."""
        n = len(observed)
        edges, b = self.loadings(observed, context, edges)
        e = len(edges)
        identity = torch.eye(2, dtype=observed.dtype, device=observed.device)
        # Each (agent, edge) is unique; no unordered floating atomic reductions.
        loading = b.new_zeros(n, e, 2, 2)
        ids = torch.arange(e, device=observed.device)
        loading[edges[:, 0], ids] = b[:, 0]
        loading[edges[:, 1], ids] = b[:, 1]
        variance = identity + (loading @ loading.transpose(-1, -2)).sum(1)
        local_chol = torch.linalg.cholesky(variance)
        independent = torch.linalg.solve_triangular(local_chol, identity.expand(n, -1, -1), upper=False)
        shared = torch.linalg.solve_triangular(local_chol[:, None], loading, upper=False)
        factor = torch.cat((torch.block_diag(*independent.unbind(0)),
                            shared.permute(0, 2, 1, 3).reshape(2*n, 2*e)), 1)
        covariance = factor @ factor.T
        return dict(factor=factor, R=covariance, edges=edges, B=b, L=local_chol)


def delta_nll(covariance, latent_truth):
    """Relative to independent N(0,I), nat/person; no Flow Jacobian remains."""
    n = latent_truth.shape[0]
    z = latent_truth.reshape(-1, 1)
    chol = torch.linalg.cholesky(covariance)
    whitened = torch.linalg.solve_triangular(chol, z, upper=False)
    return (2*chol.diagonal().log().sum()+whitened.square().sum()-z.square().sum())/(2*n)


def sample_latents(factorization, epsilon, edge_noise):
    """Both noise arrays retain the same leading raw joint sample ID."""
    n = epsilon.shape[1]
    noise = torch.cat((epsilon.flatten(1), edge_noise.flatten(1)), 1)
    return (noise @ factorization['factor'].T).reshape(-1, n, 2)
