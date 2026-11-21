"""Detached FP64 complete-scene energy and arithmetic uncertainty, no learned threshold."""
import numpy as np


class EnergyVerifier:
    def __init__(self,state):
        self.z=state.z.detach().cpu().numpy().astype(np.float64)
        self.tables=state.tables.detach().cpu().numpy().astype(np.float64)
        self.edges=state.edges.detach().cpu().numpy();self.weights=state.weights.detach().cpu().numpy().astype(np.float64)
        self.n,self.k,self.r=self.z.shape;self.i,self.j=np.triu_indices(self.n,1)

    @staticmethod
    def gamma(operations):
        u=np.finfo(np.float64).eps/2
        return operations*u/(1-operations*u)

    def evaluate(self,pi):
        selected=self.z[np.arange(self.n)[:,None],pi]
        if self.n==1:return 0.,0.
        left,right=selected[self.i],selected[self.j]
        terms=np.square(left-right).sum(-1)/(self.n-1)
        global_value=float(terms.sum());absolute=float(np.abs(terms).sum())
        # Bound subtraction/squaring/dot-product errors using magnitudes BEFORE
        # subtraction; using only ||x-y|| would understate cancellation error.
        primitive=float(np.square(np.abs(left)+np.abs(right)).sum()/(self.n-1))
        local=np.empty((0,self.k))
        if len(self.edges):
            i,j=self.edges.T
            local=self.tables[np.arange(len(i))[:,None],pi[i],pi[j]]*self.weights[:,None]
        value=global_value+float(local.sum())
        absolute+=float(np.abs(local).sum())
        # Count ALL flattened summands, not just edges; this does not rely on
        # NumPy using pairwise reduction. Round the bound outward as well.
        primitive_upper=np.nextafter(primitive/(1-self.gamma(4+left.size)),np.inf)
        absolute_upper=np.nextafter(absolute/(1-self.gamma(self.r+terms.size+local.size+6)),np.inf)
        bound=np.nextafter(self.gamma(self.r+4)*primitive_upper+
                           self.gamma(terms.size+local.size+6)*absolute_upper,np.inf)
        return value,bound

    def compare(self,current,proposal):
        before,b0=self.evaluate(current);after,b1=self.evaluate(proposal)
        delta=after-before
        error=np.nextafter(b0+b1+self.gamma(1)*(abs(before)+abs(after)),np.inf)
        return dict(energy_before_fp64=before,energy_proposed_fp64=after,energy_delta_fp64=delta,
                    arithmetic_error_bound=error,reliable_decrease=bool(delta < -error))
