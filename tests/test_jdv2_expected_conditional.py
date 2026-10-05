"""Independent CPU mathematics, not model/rollout or performance certification."""
import itertools
import math
from fractions import Fraction

import pytest
import torch

from src.joint_goal_loss import (
    expected_conditional_composite,
    jdv2_no_z_pseudo_likelihood_from_local,
    jdv2_no_z_relation_kl_per_edge,
)


# Read by the archive runner; never touched by the library function.
AUDIT = {"forward_attempted": 0, "forward_completed": 0, "forward_rejected": 0,
         "loss_output_backward_hook_visits": 0, "grad_attempted": 0,
         "grad_completed": 0, "grad_failed": 0, "measurements": {}}
MC_SEED, MC_REPETITIONS, MC_S = 527015, 1024, 4


def loss(*args, **kwargs):
    AUDIT["forward_attempted"] += 1
    try:
        value = expected_conditional_composite(*args, **kwargs)
    except Exception:
        AUDIT["forward_rejected"] += 1
        raise
    AUDIT["forward_completed"] += 1
    if value.requires_grad:
        def backward_visit(g):
            AUDIT["loss_output_backward_hook_visits"] += 1
            return g
        value.register_hook(backward_visit)
    return value


def gradients(value, inputs, **kwargs):
    AUDIT["grad_attempted"] += 1
    try:
        result = torch.autograd.grad(value, inputs, **kwargs)
    except Exception:
        AUDIT["grad_failed"] += 1
        raise
    AUDIT["grad_completed"] += 1
    return result


def close(a, b, name=None):
    atol, rtol = ((1e-10, 1e-8) if a.dtype == torch.float64 else (1e-6, 1e-5))
    if name:
        AUDIT["measurements"][name] = {
            "max_abs_error": (a.detach()-b.detach()).abs().max().item(),
            "atol": atol, "rtol": rtol, "dtype": str(a.dtype)}
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def chunks(cost, cuts=None):
    cuts = [0, cost.shape[0]] if cuts is None else cuts
    for start, stop in zip(cuts, cuts[1:]):
        yield start, stop, cost[start:stop]


def tiny(dtype=torch.float64):
    # Unequal scene sizes; node2 receives three incident edges over two chunks;
    # node4 is isolated. Deterministic nonsymmetric, nonseparable edge costs.
    u = torch.tensor([[.2, -.5, 1.1], [.8, -.2, .1], [-.3, .7, .4],
                      [1., -.6, .2], [-.9, .1, .5]], dtype=dtype, requires_grad=True)
    q = torch.tensor([[1, 2, 3], [3, 1, 2], [2, 3, 1], [1, 3, 2], [2, 1, 3]],
                     dtype=dtype) / 6
    scenes = torch.tensor([11, 11, 11, 11, 9001])
    edges = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]])
    t = torch.arange(36, dtype=dtype).reshape(4, 3, 3)
    c = (torch.sin(.73*t) + .02*t).requires_grad_()
    z = torch.tensor([[0, 1, 2, 0, 1], [2, 2, 1, 1, 0],
                      [1, 0, 0, 2, 2], [0, 2, 1, 2, 1]])
    return u, q, scenes, edges, c, z


def binary():
    u = torch.tensor([[.2, -.4], [.7, -.1], [-.3, .8]],
                     dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[.75, .25], [.25, .75], [.5, .5]], dtype=torch.float64)
    scenes = torch.tensor([7, 7, 7])
    edges = torch.tensor([[0, 1], [1, 2]])
    c = torch.tensor([[[.1, 1.3], [-.6, .4]], [[.9, -.2], [.3, 1.1]]],
                     dtype=torch.float64, requires_grad=True)
    return u, q, scenes, edges, c


def explicit_conditional(u, q, scenes, edges, c, z, mask=None):
    """Scalar-loop reference: no gather, index_add or production CE helper."""
    n, k = u.shape
    valid = torch.ones_like(q, dtype=torch.bool) if mask is None else mask
    losses = []
    for i in range(n):
        candidates = [a for a in range(k) if bool(valid[i, a])]
        logits = []
        for a in candidates:
            energy = u.new_zeros(())
            for e in range(edges.shape[1]):
                src, dst = int(edges[0, e]), int(edges[1, e])
                if i == src:
                    energy = energy + c[e, a, int(z[dst])]
                elif i == dst:
                    energy = energy + c[e, int(z[src]), a]
            logits.append(u[i, a]-energy)
        log_normalizer = torch.logsumexp(torch.stack(logits), dim=0)
        losses.append(sum(q[i, a]*(log_normalizer-v)
                          for a, v in zip(candidates, logits)))
    scene_labels = sorted(set(scenes.tolist()))
    return torch.stack([torch.stack([losses[i] for i in range(n)
                        if int(scenes[i]) == b]).mean()
                        for b in scene_labels]).mean()


def exact_reference(u, q, scenes, edges, c, mask=None):
    """All valid states weighted by product q, including nonuniform labels."""
    n, k = q.shape
    valid = torch.ones_like(q, dtype=torch.bool) if mask is None else mask
    supports = [[a for a in range(k) if bool(valid[i, a])] for i in range(n)]
    total = u.new_zeros(())
    states = list(itertools.product(*supports))
    for z in states:
        probability = math.prod(float(q[i, z[i]]) for i in range(n))
        if probability:
            total = total + probability*explicit_conditional(
                u, q, scenes, edges, c, z, valid)
    return total, states


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_asymmetric_direction_dense_chunks_reference_gradients(dtype):
    u, q, scenes, edges, c, z = tiny(dtype)
    reference = torch.stack([explicit_conditional(u, q, scenes, edges, c, row)
                             for row in z]).mean()
    ref_grad = gradients(reference, (u, c))
    for cuts in ([0, 4], [0, 1, 3, 4], [0, 1, 2, 3, 4]):
        actual = loss(u, q, scenes, edges, chunks(c, cuts), z)
        close(actual, reference, f"chunk_value_{dtype}_{cuts}")
        for label, a, b in zip(("u", "C"), gradients(actual, (u, c)), ref_grad):
            close(a, b, f"chunk_gradient_{label}_{dtype}_{cuts}")
    wrong = torch.stack([explicit_conditional(
        u, q, scenes, edges, c.transpose(1, 2), row) for row in z]).mean()
    assert abs(float((wrong-reference).detach())) > .01  # detects axis reversal


@pytest.mark.parametrize("draws", [1, 4, 7])
def test_one_hot_matches_existing_fp32_helper_value_and_gradients(draws):
    u, q, scenes, edges, c, _ = tiny(torch.float32)
    ids = torch.tensor([0, 2, 1, 2, 0])
    q = torch.nn.functional.one_hot(ids, 3).float()
    src, dst = edges
    local = torch.zeros_like(u).index_add(
        0, src, torch.einsum("ekl,el->ek", c, q[dst]))
    local = local.index_add(0, dst, torch.einsum("ekl,ek->el", c, q[src]))
    old = jdv2_no_z_pseudo_likelihood_from_local(u, q, scenes, local)
    old_grad = gradients(old, (u, c))
    actual = loss(u, q, scenes, edges, chunks(c, [0, 2, 4]), ids[None].expand(draws, -1))
    close(actual, old, f"onehot_S{draws}_value")
    for label, a, b in zip(("u", "C"), gradients(actual, (u, c)), old_grad):
        close(a, b, f"onehot_S{draws}_{label}_gradient")


@pytest.mark.parametrize("n,k", [(1, 1), (1, 3), (5, 3)])
def test_empty_edges_unary_ce_and_unequal_scene_weight(n, k):
    u = torch.arange(n*k, dtype=torch.float64).reshape(n, k).div(7).requires_grad_()
    q = torch.full((n, k), 1/k, dtype=u.dtype)
    scenes = torch.tensor([-7]*(n-1)+[1000000])
    edge = torch.empty((2, 0), dtype=torch.long)
    z = torch.zeros((4, n), dtype=torch.long)
    actual = loss(u, q, scenes, edge, iter(()), z)
    reference = explicit_conditional(u, q, scenes, edge, u.new_empty((0,k,k)), z[0])
    close(actual, reference)
    close(gradients(actual, (u,))[0], gradients(reference, (u,))[0])
    assert torch.isfinite(actual)


def test_zero_cost_keeps_scene_mean_not_global_agent_mean():
    u, q, scenes, edges, c, z = tiny()
    c = torch.zeros_like(c, requires_grad=True)
    actual = loss(u, q, scenes, edges, chunks(c), z)
    per_agent = -(q*torch.log_softmax(u, -1)).sum(-1)
    reference = .5*(per_agent[:4].mean()+per_agent[4])
    close(actual, reference)
    assert abs(float((actual-per_agent.mean()).detach())) > .001
    gu, gc = gradients(actual, (u,c))
    assert torch.isfinite(gu).all() and torch.isfinite(gc).all()


def test_uniform_and_nonuniform_weighted_enumeration():
    u, q, scenes, edges, c = binary()
    for uniform in (True, False):
        labels = torch.full_like(q, .5) if uniform else q
        reference, states = exact_reference(u, labels, scenes, edges, c)
        outputs = [loss(u, labels, scenes, edges, chunks(c), torch.tensor([z]))
                   for z in states]
        weights = torch.tensor([math.prod(float(labels[i,z[i]]) for i in range(3))
                                for z in states], dtype=u.dtype)
        actual = (torch.stack(outputs)*weights).sum()
        close(actual, reference, f"weighted_enumeration_uniform_{uniform}")
        for label, a, b in zip(("u","C"), gradients(actual,(u,c)), gradients(reference,(u,c))):
            close(a,b,f"weighted_enum_grad_{label}_{uniform}")
        if not uniform:
            assert abs(float((torch.stack(outputs).mean()-reference).detach())) > .001


def test_three_candidate_chain_uniform_enumeration():
    u, q, scenes, edges, c, _ = tiny()
    u, q, scenes, edges, c = u[:3], torch.full_like(q[:3],1/3), scenes[:3], edges[:,[0,2]], c[[0,2]]
    reference, states = exact_reference(u,q,scenes,edges,c)
    actual = loss(u,q,scenes,edges,chunks(c),torch.tensor(states))
    close(actual, reference, "N3_K3_chain_27_states")
    for a,b in zip(gradients(actual,(u,c)),gradients(reference,(u,c))):
        close(a,b)


def test_rational_repeated_configurations_are_exact_not_unweighted():
    u,q,scenes,edges,c = binary()
    reference, states = exact_reference(u,q,scenes,edges,c)
    repeated = []
    for z in states:
        probability = math.prod(Fraction(float(q[i,z[i]])) for i in range(3))
        count = probability*32
        assert count.denominator == 1
        repeated.extend([z]*count.numerator)
    assert len(repeated)==32
    actual = loss(u,q,scenes,edges,chunks(c,[0,1,2]),torch.tensor(repeated))
    close(actual,reference,"rational_S32_value")
    for name,a,b in zip(("u","C"),gradients(actual,(u,c)),gradients(reference,(u,c))):
        close(a,b,"rational_S32_gradient_"+name)


def test_jensen_counterexample_is_a_target_difference():
    u=torch.zeros((2,2),dtype=torch.float64,requires_grad=True)
    q=torch.tensor([[1.,0.],[.5,.5]],dtype=u.dtype)
    scenes=torch.tensor([0,0]); edges=torch.tensor([[0],[1]])
    c=torch.tensor([[[0.,4.],[0.,0.]]],dtype=u.dtype,requires_grad=True)
    z=torch.tensor([[0,0],[0,1]])
    actual=loss(u,q,scenes,edges,chunks(c),z)
    node1_ce=torch.logsumexp(torch.tensor([0.,-4.],dtype=u.dtype),0)+2
    mean_node0=torch.nn.functional.softplus(u.new_tensor(2.))
    expected_node0=.5*(torch.nn.functional.softplus(u.new_tensor(0.))+
                       torch.nn.functional.softplus(u.new_tensor(4.)))
    close(actual,.5*(expected_node0+node1_ce))
    assert actual > .5*(mean_node0+node1_ce)
    AUDIT["measurements"]["Jensen"]={"mean_energy_node0":float(mean_node0),
        "expected_conditional_node0":float(expected_node0),"performance_result":False}


def sample_ids(q, generator, count):
    return torch.multinomial(q,count,replacement=True,generator=generator).T


def test_s4_mc_mean_and_gradient_against_exact_expectation():
    u,q,scenes,edges,c=binary()
    reference,states=exact_reference(u,q,scenes,edges,c)
    gu,gc=gradients(reference,(u,c))
    exact=torch.cat((reference.detach().reshape(1),gu.flatten(),gc.flatten()))
    generator=torch.Generator(device="cpu").manual_seed(MC_SEED)
    global_state=torch.random.get_rng_state().clone()
    saved_z=sample_ids(q,generator,MC_REPETITIONS*MC_S).reshape(MC_REPETITIONS,MC_S,3)
    assert torch.equal(global_state,torch.random.get_rng_state())
    samples=[]
    for z in saved_z:
        actual=loss(u,q,scenes,edges,chunks(c,[0,1,2]),z)
        gu,gc=gradients(actual,(u,c))
        samples.append(torch.cat((actual.detach().reshape(1),gu.flatten(),gc.flatten())))
    samples=torch.stack(samples)
    mean=samples.mean(0); se=samples.std(0,unbiased=True)/math.sqrt(MC_REPETITIONS)
    error=(mean-exact).abs(); bound=5*se+1e-10
    AUDIT["measurements"]["MC"]={"seed":MC_SEED,"repetitions":MC_REPETITIONS,
        "S":MC_S,"draw_vectors":MC_REPETITIONS*MC_S,"enumerated_states":len(states),
        "N":3,"K":2,"E":2,"dtype":"float64","exact":exact.tolist(),
        "mean":mean.tolist(),"standard_error":se.tolist(),"absolute_bias":error.tolist(),
        "bound":bound.tolist(),"all_components_pass":bool((error<=bound).all()),
        "component_order":"loss, u row-major (6), C row-major (8)"}
    assert (error<=bound).all()
    assert torch.equal(global_state,torch.random.get_rng_state())


def test_fp64_gradcheck_unary_and_asymmetric_cost():
    u,q,scenes,edges,c=binary()
    z=torch.tensor([[0,1,0],[1,0,1],[0,1,1],[1,1,0]])
    assert torch.autograd.gradcheck(
        lambda a,b: loss(a,q,scenes,edges,chunks(b,[0,1,2]),z),
        (u,c),eps=1e-6,atol=1e-5,rtol=1e-3,fast_mode=False)


def test_raw_m4_mixture_dual_ce_and_existing_kl_gradients():
    u,q,scenes,edges,_=binary()
    u=u.detach().float().requires_grad_(); q=q.float()
    def leaf(shape, scale, offset):
        x=torch.arange(math.prod(shape),dtype=torch.float32).reshape(shape)
        return (torch.sin(x*scale+offset)*.7).requires_grad_()
    teacher=leaf((2,4),.7,.1); deploy=leaf((2,2,2,4),.3,.9)
    left=leaf((2,4,2,2),.5,.3); right=leaf((2,4,2,2),.4,-.2)
    em=-torch.einsum("emkr,emlr->eklm",left,right)/math.sqrt(2)
    lq=torch.log_softmax(teacher,-1); lp=torch.log_softmax(deploy,-1)
    cpost=-torch.logsumexp(lq[:,None,None,:]-em,-1)
    cprior=-torch.logsumexp(lp-em,-1)
    z=torch.tensor([[0,1,0],[1,0,1],[0,1,1],[1,1,0]])
    post=loss(u,q,scenes,edges,chunks(cpost),z)
    prior=loss(u,q,scenes,edges,chunks(cprior),z)
    kl=jdv2_no_z_relation_kl_per_edge(lq,lp,q,edges).mean()
    refpost=torch.stack([explicit_conditional(u,q,scenes,edges,cpost,row) for row in z]).mean()
    refprior=torch.stack([explicit_conditional(u,q,scenes,edges,cprior,row) for row in z]).mean()
    refkl=u.new_zeros(())
    for e in range(2):
        for k in range(2):
            for j in range(2):
                for m in range(4):
                    refkl=refkl+q[edges[0,e],k]*q[edges[1,e],j]*lq[e,m].exp()*(
                        lq[e,m]-lp[e,k,j,m])/2
    leaves=(u,teacher,deploy,left,right)
    labels=("u","teacher","deploy","left_factor","right_factor")
    expected_paths={"post":(1,1,0,1,1),"prior":(1,0,1,1,1),"KL":(0,1,1,0,0)}
    for name,a,b in (("post",post,refpost),("prior",prior,refprior),("KL",kl,refkl),
                     ("total",.5*post+.5*prior+.07*kl,.5*refpost+.5*refprior+.07*refkl)):
        close(a,b,"mixture_"+name+"_value")
        ga=gradients(a,leaves,allow_unused=True,retain_graph=True)
        gb=gradients(b,leaves,allow_unused=True,retain_graph=True)
        norms={}
        for j,(label,va,vb) in enumerate(zip(labels,ga,gb)):
            if va is None:
                assert vb is None
                assert not expected_paths.get(name,(1,)*5)[j]
                norms[label]=None
            else:
                close(va,vb,"mixture_"+name+"_"+label)
                norms[label]=va.abs().max().item()
                assert norms[label]>1e-7
                assert expected_paths.get(name,(1,)*5)[j]
        AUDIT["measurements"]["gradient_paths_"+name]=norms


def test_packed_equals_equal_individual_scene_losses_and_gradients():
    u,q,scenes,edges,c,z=tiny()
    packed=loss(u,q,scenes,edges,chunks(c),z)
    first=loss(u[:4],q[:4],scenes[:4],edges,chunks(c),z[:,:4])
    second=loss(u[4:],q[4:],scenes[4:],edges[:,:0],iter(()),z[:,4:])
    separate=(first+second)/2
    close(packed,separate)
    for a,b in zip(gradients(packed,(u,c)),gradients(separate,(u,c))):
        close(a,b)


def test_agent_relabel_canonical_flip_transposes_cost_and_gradient():
    u,q,scenes,edges,c,z=tiny()
    base=loss(u,q,scenes,edges,chunks(c),z)
    permutation=torch.tensor([4,2,0,3,1]); inverse=torch.argsort(permutation)
    mapped=inverse[edges]; new_edges=[]; new_cost=[]
    for e in range(c.shape[0]):
        i,j=map(int,mapped[:,e])
        new_edges.append((min(i,j),max(i,j)))
        new_cost.append(c[e] if i<j else c[e].T)
    actual=loss(u[permutation],q[permutation],scenes[permutation],
                torch.tensor(new_edges).T,chunks(torch.stack(new_cost)),z[:,permutation])
    close(actual,base)
    for a,b in zip(gradients(actual,(u,c)),gradients(base,(u,c))):
        close(a,b)


def test_candidate_relabel_mask_and_gradient():
    u,q,scenes,edges,c,z=tiny()
    mask=torch.ones_like(q,dtype=torch.bool); mask[4,2]=False
    q=q.clone(); q[4]=torch.tensor([.25,.75,0.],dtype=q.dtype); z=z.clone(); z[:,4]=0
    base=loss(u,q,scenes,edges,chunks(c),z,mask)
    order=torch.tensor([[2,0,1],[1,2,0],[1,0,2],[2,1,0],[1,2,0]])
    inv=torch.argsort(order,-1)
    cnew=torch.stack([c[e][order[i]][:,order[j]] for e,(i,j) in enumerate(edges.T)])
    znew=inv.gather(1,z.T).T
    actual=loss(u.gather(1,order),q.gather(1,order),scenes,edges,chunks(cnew),
                znew,mask.gather(1,order))
    close(actual,base)
    for a,b in zip(gradients(actual,(u,c)),gradients(base,(u,c))):
        close(a,b)


def noncontiguous(t):
    if t.ndim==1:
        return torch.stack((t,t),-1)[:,0]
    return t.transpose(-1,-2).contiguous().transpose(-1,-2)


def test_noncontiguous_mask_rng_no_input_mutation_and_repeat_backward():
    u,q,scenes,edges,c,z=tiny()
    mask=torch.ones_like(q,dtype=torch.bool); mask[0,2]=False
    q=q.clone(); q[0]=torch.tensor([.2,.8,0.],dtype=q.dtype)
    z=z.clone(); z[:,0]=torch.tensor([0,1,1,0])
    tensors=[noncontiguous(t).detach() for t in (u,q,scenes,edges,c,z,mask)]
    u,q,scenes,edges,c,z,mask=tensors
    u.requires_grad_(); c.requires_grad_()
    assert all(not t.is_contiguous() for t in tensors)
    snapshots=[(t.clone(),t._version) for t in tensors]
    state=torch.random.get_rng_state().clone()
    outputs=[]; grads=[]
    for _ in range(2):
        value=loss(u,q,scenes,edges,chunks(c,[0,1,3,4]),z,mask)
        outputs.append(value.detach()); grads.append(gradients(value,(u,c)))
    assert torch.equal(outputs[0],outputs[1])
    assert all(torch.equal(a,b) for a,b in zip(*grads))
    assert all(torch.isfinite(g).all() for g in grads[0])
    assert grads[0][0][0,2]==0
    assert torch.equal(state,torch.random.get_rng_state())
    for tensor,(before,version) in zip(tensors,snapshots):
        assert torch.equal(tensor,before) and tensor._version==version


class OnePass:
    def __init__(self,cost):
        self.cost=cost; self.iterations=0; self.yields=0
    def __iter__(self):
        self.iterations+=1
        assert self.iterations==1
        for start,stop in ((0,1),(1,3),(3,4)):
            self.yields+=1
            yield start,stop,self.cost[start:stop]


def test_chunk_iterable_consumed_once():
    u,q,scenes,edges,c,z=tiny(); iterable=OnePass(c)
    value=loss(u,q,scenes,edges,iterable,z)
    gradients(value,(u,c))
    assert iterable.iterations==1 and iterable.yields==3


def test_high_degree_across_chunks_matches_independent_gradient():
    n,k=9,3
    u=(torch.arange(n*k,dtype=torch.float64).reshape(n,k)/17).requires_grad_()
    q=torch.tensor([[.2,.3,.5]],dtype=u.dtype).expand(n,-1)
    scenes=torch.zeros(n,dtype=torch.long)
    edges=torch.stack((torch.zeros(n-1,dtype=torch.long),torch.arange(1,n)))
    c=torch.cos(torch.arange((n-1)*k*k,dtype=u.dtype).reshape(n-1,k,k)/3).requires_grad_()
    z=torch.arange(4*n).reshape(4,n)%k
    actual=loss(u,q,scenes,edges,chunks(c,[0,2,3,6,8]),z)
    reference=torch.stack([explicit_conditional(u,q,scenes,edges,c,row) for row in z]).mean()
    close(actual,reference,"degree8_value")
    for name,a,b in zip(("u","C"),gradients(actual,(u,c)),gradients(reference,(u,c))):
        close(a,b,"degree8_gradient_"+name)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_masked_reference_and_one_valid_candidate_gradients(dtype):
    u,q,scenes,edges,c,z=tiny(dtype)
    mask=torch.ones_like(q,dtype=torch.bool)
    mask[0]=torch.tensor([False,True,False]); q[0]=torch.tensor([0.,1.,0.])
    mask[2,1]=False; q[2]=torch.tensor([.75,0.,.25])
    z[:,0]=1; z[:,2]=torch.tensor([0,2,2,0])
    actual=loss(u,q,scenes,edges,chunks(c,[0,1,4]),z,mask)
    reference=torch.stack([explicit_conditional(u,q,scenes,edges,c,row,mask) for row in z]).mean()
    close(actual,reference,"masked_reference_"+str(dtype))
    gu,gc=gradients(actual,(u,c))
    ru,rc=gradients(reference,(u,c))
    close(gu,ru); close(gc,rc)
    assert torch.equal(gu[0],torch.zeros_like(gu[0]))
    assert gu[2,1]==0


def test_label_tolerance_does_not_silently_renormalize():
    u=torch.tensor([[0.,1.]],dtype=torch.float64,requires_grad=True)
    q=torch.tensor([[.25,.75]],dtype=u.dtype)*(1+5e-13)
    edges=torch.empty((2,0),dtype=torch.long)
    actual=loss(u,q,torch.tensor([4]),edges,iter(()),torch.tensor([[1]]))
    explicit=-(q*torch.log_softmax(u,-1)).sum()
    normalized=-((q/q.sum())*torch.log_softmax(u,-1)).sum()
    assert torch.equal(actual,explicit)
    assert abs(float((actual-normalized).detach())) > 1e-14
    with pytest.raises(ValueError,match="normalized"):
        loss(u,q*1.00001,torch.tensor([4]),edges,iter(()),torch.tensor([[1]]))


@pytest.mark.parametrize("kind",[
    "missing","overlap","repeat","out_of_order","empty","too_far","shape",
    "wrong_dtype","bound_float","bound_bool","bad_tuple","cost_not_tensor",
    "nan","inf","negative_inf","meta_device"])
def test_invalid_chunks_rejected(kind):
    u,q,scenes,edges,c,z=tiny()
    entries=[(0,4,c)]; error=ValueError
    if kind=="missing": entries=[(0,2,c[:2])]
    elif kind=="overlap": entries=[(0,2,c[:2]),(1,4,c[1:])]
    elif kind=="repeat": entries=[(0,2,c[:2]),(0,2,c[:2])]
    elif kind=="out_of_order": entries=[(2,4,c[2:]),(0,2,c[:2])]
    elif kind=="empty": entries=[(0,0,c[:0]),(0,4,c)]
    elif kind=="too_far": entries=[(0,5,c)]
    elif kind=="shape": entries=[(0,4,c[:3])]
    elif kind=="wrong_dtype": entries=[(0,4,c.float())]; error=TypeError
    elif kind=="bound_float": entries=[(0.,4,c)]; error=TypeError
    elif kind=="bound_bool": entries=[(False,4,c)]; error=TypeError
    elif kind=="bad_tuple": entries=[(0,4)]
    elif kind=="cost_not_tensor": entries=[(0,4,None)]; error=TypeError
    elif kind in ("nan","inf","negative_inf"):
        bad=c.detach().clone(); bad[0,0,0]=float("-inf" if kind=="negative_inf" else kind)
        entries=[(0,4,bad)]
    elif kind=="meta_device": entries=[(0,4,torch.empty(c.shape,device="meta"))]
    with pytest.raises(error):
        loss(u,q,scenes,edges,iter(entries),z)


@pytest.mark.parametrize("kind",[
    "q_trainable","q_negative","q_zero","q_not_normalized","q_nan","q_inf",
    "u_nan","u_inf","u_half","q_dtype","scene_dtype","edge_dtype","z_dtype",
    "mask_dtype","mask_shape","mask_empty","q_off_support","z_off_support",
    "z_negative","z_too_large","edge_negative","edge_too_large","self_edge",
    "reverse_edge","duplicate","cross_scene","u_shape","q_shape","scene_shape",
    "edge_shape","z_shape","N_zero","K_zero","S_zero","not_tensor","meta_device"])
def test_invalid_inputs_rejected(kind):
    u,q,scenes,edges,c,z=tiny(); mask=torch.ones_like(q,dtype=torch.bool)
    error=ValueError
    if kind=="q_trainable": q.requires_grad_()
    elif kind=="q_negative": q[0,0]=-.1
    elif kind=="q_zero": q[0]=0
    elif kind=="q_not_normalized": q=q*2
    elif kind=="q_nan": q[0,0]=torch.nan
    elif kind=="q_inf": q[0,0]=torch.inf
    elif kind in ("u_nan","u_inf"):
        u=u.detach().clone(); u[0,0]=torch.nan if kind=="u_nan" else torch.inf
    elif kind=="u_half": u=u.half(); error=TypeError
    elif kind=="q_dtype": q=q.float(); error=TypeError
    elif kind=="scene_dtype": scenes=scenes.int(); error=TypeError
    elif kind=="edge_dtype": edges=edges.int(); error=TypeError
    elif kind=="z_dtype": z=z.int(); error=TypeError
    elif kind=="mask_dtype": mask=mask.long(); error=TypeError
    elif kind=="mask_shape": mask=mask[:1]
    elif kind=="mask_empty": mask[0]=False
    elif kind=="q_off_support": mask[0,2]=False
    elif kind=="z_off_support":
        mask[0,2]=False; q[0]=torch.tensor([.5,.5,0.]); z[0,0]=2
    elif kind=="z_negative": z[0,0]=-1
    elif kind=="z_too_large": z[0,0]=3
    elif kind=="edge_negative": edges[0,0]=-1
    elif kind=="edge_too_large": edges[1,0]=5
    elif kind=="self_edge": edges[1,0]=edges[0,0]
    elif kind=="reverse_edge": edges[:,0]=edges[:,0].flip(0)
    elif kind=="duplicate": edges[:,1]=edges[:,0]
    elif kind=="cross_scene": edges[1,0]=4
    elif kind=="u_shape": u=u[0]
    elif kind=="q_shape": q=q[0]
    elif kind=="scene_shape": scenes=scenes[:,None]
    elif kind=="edge_shape": edges=edges.T
    elif kind=="z_shape": z=z[:,:2]
    elif kind=="N_zero": u=u[:0]
    elif kind=="K_zero": u=u[:,:0]
    elif kind=="S_zero": z=z[:0]
    elif kind=="not_tensor": u=None; error=TypeError
    elif kind=="meta_device": u=torch.empty(u.shape,device="meta")
    with pytest.raises(error):
        loss(u,q,scenes,edges,chunks(c),z,mask)


def test_nonfinite_masked_values_also_rejected_and_overflow_explicit():
    u,q,scenes,edges,c,z=tiny()
    mask=torch.ones_like(q,dtype=torch.bool); mask[0,2]=False
    q[0]=torch.tensor([.5,.5,0.]); z[:,0]=0
    badu=u.detach().clone(); badu[0,2]=-torch.inf
    with pytest.raises(ValueError,match="all finite"):
        loss(badu,q,scenes,edges,chunks(c),z,mask)
    badc=c.detach().clone(); badc[0,2,2]=torch.nan
    with pytest.raises(ValueError,match="all finite"):
        loss(u,q,scenes,edges,chunks(badc),z,mask)
    big=torch.full_like(c,torch.finfo(c.dtype).max)
    with pytest.raises(FloatingPointError,match="overflow"):
        loss(u,q,scenes,edges,chunks(big),z,mask)
