"""Identity-bound ZARA2 train/dev adapters over the unchanged formal v2 loader."""
from pathlib import Path
import subprocess
import torch
from bcep.contracts import canonical_hash,digest
from bcep_joint.common import read
from bcep_formal.data import FormalData
from bcep_v2.bank import Bank
from .contracts import validate_batch

SOURCE=Path(__file__).resolve().parents[1]
BASE='bfc51ba53b9ead6b5b1acd4996672c18e64905ef'
V2='468c5eddde48bdeec47c4bb3c39418352be16bb9'
EXPERIMENT='zara2_bcep_v3_stage2_20261120'
AUDIT='zara2_bcep_v3_stage2_readiness_20261120'
BANK_HASH={'train':'2883ba7a847c6dcf34e08159529d1b10803e9ff5626cbffbd9c8bd66aa8d605a','selection_dev':'b6aee909ed8bcdac6bb08b93fba554303829f94707cf54e87eccd399a07c1560'}


def paths():
    assets=SOURCE.parent/'RSJG_EBJD_assets/experiments'
    return dict(formal=assets/'bcep_ebjd_v2_integrated',candidate_cost=assets/'zara2_costproxy_stage2_20261119',
                sharedworld=assets/'zara2_sharedworld_stage2_onerun_20261119',run=assets/EXPERIMENT,audit=assets/AUDIT)


def check_architecture():
    # Only the Stage2 adapter/controller namespace may change from the accepted architecture.
    names=['bcep_v3/'+x+'.py' for x in ['model','coupling','evidence','geometry','gate','contracts']]
    names += subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'bcep','ebjd','bcep_v2','bcep_joint'],cwd=SOURCE,text=True).splitlines()
    for n in names:
        if n.endswith('.py'):
            original=subprocess.check_output(['git','show',BASE+':'+n],cwd=SOURCE)
            if original!=(SOURCE/n).read_bytes():raise ValueError('frozen architecture/runtime changed: '+n)
    return {n:digest(SOURCE/n) for n in names if n.endswith('.py')}


class TrainDevData(FormalData):
    def item(self,i,purpose,truth=True):
        if purpose not in ('train','selection_dev'):raise ValueError('only train/selection_dev are accessible')
        item=super().item(i,purpose,truth)
        item['metadata']=dict(item['metadata'],dt_seconds=.4,candidate_frame='world_xy',frame_mapping='observed8_future12')
        return item


class CheckedBank:
    def __init__(self,native,purpose):self.native=native;self.purpose=purpose;self.hash=native.hash
    def batch(self,ids,batch):
        goals=self.native.batch(ids,batch)
        validate_batch(batch,goals)
        return goals


class FormalInputs:
    def __init__(self):
        self.roots=paths();self.fixed_sources=check_architecture();up=self.roots['formal']
        required={'formal':'PAUSED_RESUMABLE','candidate_cost':'SESSION_BUDGET_PAUSED','sharedworld':'COMPLETE'}
        for name,state in required.items():
            if read(self.roots[name]/'STATUS.json')['status']!=state:raise ValueError('protected queue state changed: '+name)
        run=read(up/'FORMAL_RUN.json')
        if run['commit']!=V2:raise ValueError('wrong formal v2 source')
        plan=read(run['input_plan']);desc=plan['folds']['zara2']['split']
        if digest(desc['path'])!=desc['sha256'] or desc['sha256']!='b31703d83fce7aadc735897efaa5c3970c04f477c70afb05879a05669657a816':raise ValueError('split changed')
        self.split=read(desc['path']);s=self.split
        if (s['fold'],s['unit'],s['dt'],s['pred_length'])!=('zara2','m',.4,12):raise ValueError('split protocol mismatch')
        if canonical_hash(s)!='716e44710ff74faeffae1a6d7d814687443621e44623a0a9a1233cf8bf4f9bbb':raise ValueError('split identity changed')
        ids=s['train_indices']+s['selection_dev_indices']
        if (len(s['train_indices']),len(s['selection_dev_indices']))!=(2681,590):raise ValueError('wrong membership')
        for i in ids:
            if digest(s['records'][i]['path'])!=s['records'][i]['sha256']:raise ValueError('record digest mismatch')
        self.data=TrainDevData(s,'training',verify=False) # Explicit train/dev verification above; never load heldout files.
        self.banks=[]
        for purpose in ['train','selection_dev']:
            folder=up/'zara2/refined_banks'/f'{purpose}_2035';ident=read(folder/'IDENTITY.json')
            if (ident['seed'],ident['K'],ident['M'],ident['unit'],ident['purpose'],ident['test_only'])!=(2035,20,256,'m',purpose,False):raise ValueError('candidate contract changed')
            if ident['base']!=read(up/'zara2/density/FROZEN.json') or ident['head']!=read(up/'zara2/head/FROZEN.json'):raise ValueError('provider/bank provenance mismatch')
            bank=Bank(folder,ident,s)
            if bank.hash!=BANK_HASH[purpose]:raise ValueError('bank changed')
            self.banks.append(CheckedBank(bank,purpose))
        self.provider=[]
        for stage,epoch in [('density',19),('head',9)]:
            best=read(up/f'zara2/{stage}/TOP5.json')[0]
            if best['epoch']!=epoch or digest(up/f'zara2/{stage}'/best['file'])!=best['sha256']:raise ValueError('frozen provider weights changed')
            self.provider.append(dict(stage=stage,epoch=epoch,sha256=best['sha256'],file=best['file']))
        self.config=dict(read(up/'zara2/CONFIG.json')['joint'],scales=s['bridge_scales'],allow_real_training=False,pair_objective='final_coordinate')
        self.recipe=dict(seed=3201,lr=.0001,weight_decay=.0001,clip=5.,logical_scenes=16,max_epochs=100,minimum=30,patience=15,scheduler_horizon=100)
        self.counts={p:dict(windows=len(s[p+'_indices']),agents=sum(len(s['members'][i]['agent_ids']) for i in s[p+'_indices']),
                          max_agents=max(len(s['members'][i]['agent_ids']) for i in s[p+'_indices']),
                          N_ge_9=sum(len(s['members'][i]['agent_ids'])>=9 for i in s[p+'_indices'])) for p in ['train','selection_dev']}
        if self.counts['selection_dev']['N_ge_9']!=127 or self.counts['train']['max_agents']!=57:raise ValueError('population changed')
