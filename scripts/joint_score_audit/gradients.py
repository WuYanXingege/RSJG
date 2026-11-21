"""Native final_coordinate VJPs on 36 predeclared dev windows. NEVER optimize."""
import argparse, importlib, time
from common import *


def norm(x):
    return 0. if x is None else float(x.detach().double().norm())


def centered(x):
    return x-x.mean(-1,keepdim=True)-x.mean(-2,keepdim=True)+x.mean((-1,-2),keepdim=True)


def module_group(name):
    if name.startswith('coupling.'):
        for key in ['world_queries','scene_cross_attention','score','project','compatibility','csee','hcra']:
            if name.startswith('coupling.'+key):return key
        return 'endpoint_candidate_features'
    if name.startswith('bridge.encoder.'):return 'shared_encoder'
    return 'bridge_other_pair_unused'


def main(label):
    setup(label)
    from bcep_v2.joint import pair_loss
    from ebjd.unified import log_sinkhorn
    model,inputs=load_model(label);before=model_hash(model)
    # Native cuDNN GRU backward requires training mode. There is no nonzero
    # dropout or running-statistics normalization in this source-bound model.
    for module in model.modules():
        assert not isinstance(module,torch.nn.modules.batchnorm._BatchNorm)
        if isinstance(module,torch.nn.Dropout):assert module.p==0
        if isinstance(module,(torch.nn.MultiheadAttention,torch.nn.GRU)):assert module.dropout==0
    model.train()
    ids=read(OUT/'GRADIENT_WINDOWS.json');assert len(ids)==36
    native_module=importlib.import_module('bcep_v2.joint' if label.startswith('v2') else 'bcep_v3.coupling')
    native=native_module.assignment;calls=[];cap={}
    def assigned(scores,*a,**kw):
        result=native(scores,*a,**kw);calls.append((scores,result));return result
    native_module.assignment=assigned
    handles=[model.coupling.candidate_norm.register_forward_hook(lambda m,a,y:cap.update(candidate=y)),
             model.coupling.scene_cross_attention.register_forward_hook(lambda m,a,y:cap.update(world=y[0]))]
    if label=='v3_e12':
        handles.append(model.coupling.hcra.delta.register_forward_pre_hook(lambda m,a:cap.update(hcra_world=a[0][:,:,:,0,192:384],hcra_candidate=a[0][:,:,0,:,:192])))
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    rows=[];pr=[];hr=[];start=time.time()
    for index in ids:
        begin=time.time();torch.cuda.reset_peak_memory_stats();cap.clear();calls.clear()
        batch=inputs.data.batch([index],'selection_dev');g=inputs.banks[1].batch([index],batch).cuda();x=batch.to('cuda')
        context=model.encode_context(x.observed,x.semantic_maps,x.valid)
        h,pi,scores=model.coupling.assign(context,g,model.bridge.representation.goal_scale,straight_through=True)
        loss,metrics=pair_loss(h,x.future[:,:,-1],x.valid)
        assert len(calls)==2
        s0,(_,initial,soft0)=calls[0];sf,(_,_,softf)=calls[1]
        named=[('final_score',sf),('initial_score',s0),('final_soft',softf),('initial_st',initial),('world',cap['world']),('candidate',cap['candidate'])]
        active=[(n,t) for n,t in named+params if t.requires_grad]
        grads=torch.autograd.grad(loss,[t for n,t in active],allow_unused=True)
        gd=dict(zip([n for n,t in active],grads))
        z=lambda name,t:torch.zeros_like(t) if gd.get(name) is None else gd[name]
        gs=z('final_score',sf);gp=z('final_soft',softf);g0=z('initial_score',s0);gi=z('initial_st',initial)
        indirect=g0-gs;n=int(x.valid.sum());cg=centered(gp)
        # Independently apply the ORIGINAL Sinkhorn VJP to the measured initial-ST adjoint.
        leaf=s0.detach().requires_grad_(True)
        pp=log_sinkhorn(leaf.float(),20)
        movable=x.valid & (x.valid.sum(1,keepdim=True)>1)
        pp=torch.where(movable[...,None,None],pp,torch.eye(20,device=pp.device))
        expected,=torch.autograd.grad(pp,leaf,grad_outputs=gi)
        uni_prediction=cg/20 if n>1 else torch.zeros_like(cg)
        frozen=score_record(label,index)
        row=dict(model=label,index=index,n_agents=n,people_bin=people(n),pair_loss_m=float(loss.detach()),
            hard_JFDE_m=float(metrics['pair_exact_hard_jfde_m']),
            initial_pi_changed_vs_inference=int(np.count_nonzero(array(calls[0][1][0])[0]!=frozen['pi0'])),
            final_pi_changed_vs_inference=int(np.count_nonzero(array(pi)[0]!=frozen['pi'])),
            initial_score_max_change_vs_inference=float(np.max(np.abs(array(s0)[0]-frozen['S0']))),
            final_score_max_change_vs_inference=float(np.max(np.abs(array(sf)[0]-frozen['final']))),
            g_soft_norm=norm(gp),g_soft_effective_norm=norm(cg),g_final_score_norm=norm(gs),
            native_sinkhorn_gain=norm(gs)/norm(cg) if norm(cg) else None,
            uniform_jacobian_relative_residual=norm(gs-uni_prediction)/norm(gs) if norm(gs) else None,
            g_initial_score_total_norm=norm(g0),g_initial_st_adjoint_norm=norm(gi),
            g_initial_feedback_only_norm=norm(expected),
            feedback_to_direct_ratio=norm(expected)/norm(gs) if norm(gs) else None,
            g_initial_feedback_fp32_subtraction_norm=norm(indirect),
            initial_chain_residual_relative_to_direct=norm(indirect-expected)/norm(gs) if norm(gs) else 0.,
            initial_chain_residual_relative_to_indirect=norm(indirect-expected)/norm(expected) if norm(expected) else 0.,
            g_world_features_norm=norm(gd.get('world')),g_candidate_features_norm=norm(gd.get('candidate')),
            seconds=time.time()-begin,peak_allocated_MB=torch.cuda.max_memory_allocated()/2**20)
        rows.append(row)
        for group in sorted({module_group(name) for name,p in params}):
            members=[(name,p,gd.get(name)) for name,p in params if module_group(name)==group]
            pr.append(dict(model=label,index=index,n_agents=n,people_bin=people(n),module=group,
                parameter_tensors=len(members),used_tensors=sum(gv is not None for _,_,gv in members),
                nonzero_tensors=sum(norm(gv)>0 for _,_,gv in members),
                parameter_norm=float(np.sqrt(sum(norm(p)**2 for _,p,_ in members))),
                gradient_norm=float(np.sqrt(sum(norm(gv)**2 for _,_,gv in members))),
                finite=all(gv is None or bool(torch.isfinite(gv).all()) for _,_,gv in members)))
        if label=='v3_e12':
            hr.append(dict(index=index,n_agents=n,**{'world_'+k:v for k,v in feature_stats(array(cap['hcra_world'])).items()},
                           **{'candidate_'+k:v for k,v in feature_stats(array(cap['hcra_candidate'])).items()}))
        assert all(gv is None or bool(torch.isfinite(gv).all()) for gv in grads)
        assert all(p.grad is None for _,p in params)
        print(label,index,n,'gscore',norm(gs),'gindirect',norm(indirect),flush=True)
        del context,h,pi,scores,loss,grads,gd,active,cap,gs,gp,g0,gi,indirect,cg,expected,pp,leaf,named,s0,sf,softf,soft0,initial
        cap={};calls.clear()
    for h in handles:h.remove()
    native_module.assignment=native
    assert model_hash(model)==before and GUARD==dict(optimizer_constructions=0,optimizer_step_calls=0)
    csvwrite(DOC/f'GRADIENTS_{label}.csv',rows);csvwrite(DOC/f'PARAMETER_GRADIENTS_{label}.csv',pr)
    if hr:csvwrite(DOC/'HCRA_FEATURES_SAMPLE.csv',hr)
    write(DOC/f'GRADIENT_PROOF_{label}.json',dict(model=label,windows=ids,model_state_hash=before,unchanged=True,
        optimizer_guard=GUARD,gradient_mode='native train mode (cuDNN GRU backward) + ST + final_coordinate pair loss ONLY; one dev window, no augmentation or clipping; no dropout/BatchNorm',
        no_claim_to_replay_training_batch_gradient=True,seconds=time.time()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list(MODELS),required=True);main(p.parse_args().model)
