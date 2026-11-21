"""Fill only missing scores/features. Existing v3 scores/Hungarian/paths are reused."""
import argparse,time
import numpy as np
import torch
from common import *


@torch.no_grad()
def main(label):
    setup(label);dest=OUT/label;assert not dest.exists();(dest/'windows').mkdir(parents=True)
    model,inputs=load_model(label);before=model_hash(model)
    capture={};handles=[]
    handles.append(model.coupling.candidate_norm.register_forward_hook(lambda m,a,y:capture.update(candidate=array(y))))
    handles.append(model.coupling.scene_cross_attention.register_forward_hook(lambda m,a,y:capture.update(world=array(y[0]))))
    if label.startswith('v2'):
        import bcep_v2.joint as joint
        native=joint.assignment;calls=[]
        def assigned(scores,*a,**kw):
            result=native(scores,*a,**kw);calls.append((array(scores),array(result[0])));return result
        joint.assignment=assigned
    else:
        class FeatureCaptured(Exception):pass
        def stop_score(*a):raise FeatureCaptured()
        handles.append(model.coupling.score.register_forward_pre_hook(stop_score))
    np.savez_compressed(dest/'STATIC_FEATURES.npz',queries=array(model.coupling.world_queries))
    receipts=[];start=time.time();matched=0;differing=0
    for pos,index in enumerate(inputs.split['selection_dev_indices']):
        batch=inputs.data.batch([index],'selection_dev');goals=inputs.banks[1].batch([index],batch)
        x=batch.to('cuda');g=goals.cuda();capture.clear()
        context=model.encode_context(x.observed,x.semantic_maps,x.valid)
        if label.startswith('v2'):
            calls.clear();h,pi,scores=model.coupling.assign(context,g,model.bridge.representation.goal_scale,straight_through=False)
            assert len(calls)==2
            saved=dict(S0=calls[0][0][0],final=calls[1][0][0],pi0=calls[0][1][0],pi=calls[1][1][0])
            with np.load(OUT/'references/assignments'/str(MODELS[label]['epoch'])/f'{index}.npz') as ref:
                equal=np.array_equal(saved['pi'],ref['pi_H'][0]) and np.array_equal(saved['pi0'],ref['pi_initial'][0])
                matched+=int(equal);differing+=int(not equal)
        else:
            # Execute unchanged native feature path and stop BEFORE score MLP.
            try:model.coupling.forward_features(context,g,model.bridge.representation.goal_scale)
            except FeatureCaptured:pass
            else:raise AssertionError('score interception failed')
            saved={}
        saved.update(candidate=capture['candidate'][0],world=capture['world'][0])
        p=dest/'windows'/f'{index}.npz';np.savez_compressed(p,**saved)
        receipts.append(dict(index=index,n_agents=int(x.valid.sum()),file=str(p.relative_to(OUT)),sha256=sha(p)))
        if (pos+1)%100==0:print(label,pos+1,'seconds',round(time.time()-start,2),flush=True)
    assert model_hash(model)==before and GUARD==dict(optimizer_constructions=0,optimizer_step_calls=0)
    for h in handles:h.remove()
    write(dest/'FILES.json',receipts)
    write(dest/'PROOF.json',dict(model=label,model_sha256=before,checkpoint_sha256=MODELS[label]['sha'],
        source_commit=OLD_COMMIT if label.startswith('v2') else BASE,weights_unchanged=True,windows=590,
        historical_pi_exact_windows=matched,historical_pi_different_windows=differing,
        score_forward_windows=590 if label.startswith('v2') else 0,feature_forward_windows=590,
        DDIM_calls=0,optimizer_guard=GUARD,seconds=time.time()-start))
    print(label,'COMPLETE',matched,differing,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list(MODELS),required=True);main(p.parse_args().model)
