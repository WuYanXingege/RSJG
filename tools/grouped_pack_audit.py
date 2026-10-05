#!/usr/bin/env python3
"""All real native packs via explicit preparation provider, NOT production eval."""
import argparse,json,sys,time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.p2_grouped_training import *
from src.p2_grouped_artifacts import make_grant
from src.models.model import GDTS
from src.models.goal_pretrain import Goal_Pretrain

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--production',action='store_true',help='Requires genuine qualification; no preparation grant')
    a=ap.parse_args();D=a.archive.resolve()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    m=json.loads((D/'QUALIFIED_INPUT_REVIEW_MANIFEST.json').read_text());reg=Registry(m,m['manifest_hash'])
    pm=json.loads((D/'PREPARATION_MANIFEST.json').read_text());preg=Registry(pm,pm['manifest_hash']);grant=PreparationGrant(preg,make_grant(preg))
    ds=geometry_dataset(reg);owner=SimpleNamespace(args=SimpleNamespace(obs_length=8,seq_length=20,down_factor=8),device=torch.device('cpu'),dataset=ds,active_goal_model_type='independent')
    results={};start=time.monotonic()
    for role in ('train','inner_valid'):
        reader=PackReader(reg,role,preparation=None if a.production else grant);begin=time.monotonic();exposures=0;visited=[]
        for i,refs in enumerate(reader.packs):
            scene=ds.scenes[reg.by_id[refs[0][0]]['source']['scene_family']]
            if role=='train':
                data,ids,obs,target=reader.training(i)
                # Actual pure method body, no nn.Module constructor/forward/backward.
                x,seq=GDTS.prepare_inputs(owner,data,ids)
                gx,_,gs=Goal_Pretrain.prepare_inputs(owner,data,ids)
                assert x['x_augmented'].shape==(20,len(refs),8) and gs.shape==(20,len(refs))
                ox=observation_inputs(obs,scene,'cpu')
                assert torch.equal(x['x_augmented'][:8],ox['x_augmented'])
                assert torch.equal(gx['input_traj_maps'][:,:8],ox['input_traj_maps'])
            else:
                obs=reader.observation(i);x=observation_inputs(obs,scene,'cpu')
                assert x['x_augmented'].shape==(8,len(refs),8)
                # Explicit target artifact validation after obs-only adapter; no claim
                # that this adapter is a prediction or a qualified metric evaluation.
                target=reader._read(i,'target');metric=metric_inputs(obs,target,scene,'cpu')
                assert metric['abs_pixel_coord'].shape==(20,len(refs),2)
            assert torch.isfinite(x['x_augmented']).all()
            exposures+=len(refs);visited.extend(refs)
            if (i+1)%50==0:print(json.dumps(dict(role=role,packs=i+1,exposures=exposures)),flush=True)
        expected_packs,expected_exposures=(175,11122) if role=='train' else (391,24955)
        assert len(reader)==expected_packs and exposures==expected_exposures and len(set(visited))==exposures
        results[role]=dict(packs=len(reader),agent_window_exposures=exposures,packing_sha256=digest(reader.packs),
            reads=dict(reader.provider.counts),seconds=time.monotonic()-begin,
            actual_pure_prepare_inputs=role=='train',actual_observation_only_adapter=True)
    atomic_json(dict(status='PASS' if a.production else 'ARTIFACT_VALIDATION_ONLY_NOT_PRODUCTION',results=results,seconds=time.monotonic()-start,
        real_model_construct=0,model_forward=0,gradient=0,outer_targets=0,teacher=0,
        production_loader='PASS' if a.production else 'EXTERNAL_EVIDENCE_BLOCKED',inner_prediction_or_metrics=0),
        D/('PRODUCTION_LOADER_RESULT.json' if a.production else 'FULL_NATIVE_PACK_ARTIFACT_AUDIT.json'))
if __name__=='__main__':main()
