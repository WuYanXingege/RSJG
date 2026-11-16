"""Explicit scene-macro objective and agent-micro endpoint likelihood."""
import copy

SELECTION_METRIC = 'dev_nll_final_agent_micro'

class LikelihoodAccumulator:
    def __init__(self,state=None):
        self.state=copy.deepcopy(state) if state is not None else dict(scenes=0,agents=0,loss_total_sum=0.,final_sum=0.,final_scene_mean_sum=0.,auxiliary={})
    def update(self,components):
        c=components;s=self.state;n=int(c['valid_agent_count'])
        if n<1 or not c['world_jacobian_included']:raise ValueError('valid world likelihood required')
        total=float(c['loss_total'].detach());final=float(c['nll_final_sum'].detach())
        import math
        if not math.isfinite(total) or not math.isfinite(final):raise ValueError('nonfinite loss')
        s['scenes']+=1;s['agents']+=n;s['loss_total_sum']+=total;s['final_sum']+=final;s['final_scene_mean_sum']+=final/n
        for h,x in c['auxiliary'].items():
            k=str(h);v=s['auxiliary'].setdefault(k,dict(sum=0.,agents=0,scene_mean_sum=0.));v['sum']+=float(x['nll_sum'].detach());v['agents']+=int(x['valid_agent_count']);v['scene_mean_sum']+=float(x['nll_sum'].detach())/int(x['valid_agent_count'])
    def metrics(self,prefix='dev'):
        s=self.state
        if not s['scenes']:raise ValueError('no likelihood samples')
        return {f'{prefix}_loss_total_scene_macro':s['loss_total_sum']/s['scenes'],f'{prefix}_nll_final_agent_micro':s['final_sum']/s['agents'],f'{prefix}_nll_final_scene_macro':s['final_scene_mean_sum']/s['scenes'],f'{prefix}_nll_final_sum':s['final_sum'],f'{prefix}_valid_agent_count':s['agents'],f'{prefix}_scene_count':s['scenes'],f'{prefix}_auxiliary':{h:dict(nll_sum=v['sum'],valid_agent_count=v['agents'],agent_micro=v['sum']/v['agents'],scene_macro=v['scene_mean_sum']/s['scenes']) for h,v in s['auxiliary'].items()},'world_jacobian_included':True}
