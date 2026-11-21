"""CPU mathematical prototype, NOT an installed/trainable network.

One common energy, exact conditional replacement scores, native SciPy Hungarian.
No optimizer, score scaling, score noise, temperature, or data-dependent fitting.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class SceneEnergy:
    z: np.ndarray                 # [N,K,R], candidate latents
    edges: np.ndarray             # [E,2], unique undirected i<j
    tables: np.ndarray            # [E,K,K], same potential used by both agents

    def __post_init__(self):
        self.z=np.asarray(self.z,np.float64)
        self.edges=np.asarray(self.edges,np.int64).reshape(-1,2)
        self.tables=np.asarray(self.tables,np.float64)
        self.n,self.k,self.r=self.z.shape
        assert self.tables.shape==(len(self.edges),self.k,self.k)
        assert len(set(map(tuple,self.edges)))==len(self.edges)
        assert all(0<=i<j<self.n for i,j in self.edges)
        degree=np.bincount(self.edges.ravel(),minlength=self.n)
        self.weights=np.array([1/np.sqrt(degree[i]*degree[j]) for i,j in self.edges])
        self.incidence=[[(e,int(j),False) if i==a else (e,int(i),True)
                         for e,(i,j) in enumerate(self.edges) if a in (i,j)] for a in range(self.n)]

    def validate(self,p):
        assert p.shape==(self.n,self.k)
        assert np.array_equal(np.sort(p,axis=1),np.broadcast_to(np.arange(self.k),p.shape))

    def selected(self,p):return self.z[np.arange(self.n)[:,None],p]

    def world_energy(self,p):
        # Also accepts a one-candidate replacement with duplicated candidate indices.
        selected=self.selected(p)
        energy=np.zeros(self.k)
        if self.n>1:
            energy=(self.n*np.square(selected).sum((0,2))-np.square(selected.sum(0)).sum(-1))/(self.n-1)
        for e,(i,j) in enumerate(self.edges):energy+=self.weights[e]*self.tables[e,p[i],p[j]]
        return energy

    def scores(self,p,i):
        """S[w,a] = -(E_w(p with i:=a)-E_w(p)), frozen other people."""
        selected=self.selected(p);old=selected[i];candidate=self.z[i]
        delta=np.zeros((self.k,self.k))
        if self.n>1:
            others=selected.sum(0)-old
            delta=np.square(candidate).sum(-1)[None]-np.square(old).sum(-1)[:,None]
            delta-=2/(self.n-1)*(others@candidate.T-np.sum(others*old,axis=-1)[:,None])
        for e,j,transpose in self.incidence[i]:
            table=self.tables[e].T if transpose else self.tables[e]
            cost=table[:,p[j]].T
            delta+=self.weights[e]*(cost-cost[np.arange(self.k),p[i]][:,None])
        return -delta

    def step(self,p,i):
        """One coordinate proposal. Exact ties abstain; no epsilon tie breaking."""
        self.validate(p);s=self.scores(p,i)
        _,q=linear_sum_assignment(-s)
        value=float(s[np.arange(self.k),q].sum())
        second=-np.inf
        for w,a in enumerate(q):
            forbidden=-s.copy();forbidden[w,a]=np.inf
            _,alt=linear_sum_assignment(forbidden)
            second=max(second,float(s[np.arange(self.k),alt].sum()))
        gap=value-second
        before=float(self.world_energy(p).sum());proposal=p.copy();proposal[i]=q
        after=float(self.world_energy(proposal).sum())
        # The unique optimum check is conservative. Ambiguous improvements abstain.
        accepted=bool(gap>0 and value>0 and after<before)
        return (proposal if accepted else p.copy()),dict(person=i,accepted=accepted,
            gap=gap,strict_tie=gap==0,score_gain=value,energy_before=before,
            proposed_energy=after,energy_after=after if accepted else before,
            changed_choices=int(np.count_nonzero(q!=p[i])) if accepted else 0)

    def sweep(self,p,order=None):
        self.validate(p);p=p.copy();trace=[]
        for i in (range(self.n) if order is None else order):
            if self.n==1:break
            p,record=self.step(p,i);trace.append(record)
        return p,trace


def synthetic(n=7,k=20,r=8,seed=3201,dense=False):
    """Fixed synthetic algebra inputs, never training examples or fitted parameters."""
    rng=np.random.default_rng(seed);z=rng.normal(size=(n,k,r))
    edges=np.array([(i,j) for i in range(n) for j in range(i+1,n) if dense or j-i<=3],dtype=int).reshape(-1,2)
    tables=rng.normal(size=(len(edges),k,k))
    p=np.stack([rng.permutation(k) for _ in range(n)])
    return SceneEnergy(z,edges,tables),p
