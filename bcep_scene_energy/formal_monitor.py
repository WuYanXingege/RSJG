"""Detached SSEC observations and native optimizer POST hooks; no update rule here."""
import contextlib,json,time
from pathlib import Path
import torch
from ebjd.unified import log_sinkhorn

PREFIXES=('coupling.energy.phi.','coupling.energy.rho.','bridge.encoder.')


def population(n):
    if n==1:return '1'
    if n<9:return '2-8'
    if n<20:return '9-19'
    if n<42:return '20-41'
    return '42-57'


@torch.no_grad()
def decision_statistics(detail,straight_through):
    rows=[]
    for scene in detail['scenes']:
        n=scene['valid_people'];trace=scene['trace']
        scores=torch.stack(scene['proposal_scores']).detach().double()
        effective=scores-scores.mean(-1,keepdim=True)-scores.mean(-2,keepdim=True)+scores.mean((-1,-2),keepdim=True)
        # Inference returns a hard permutation; this is the native TRAINING
        # relaxation at the measured scores, NOT the inference hard entropy.
        soft=log_sinkhorn(scores.float(),20).double() if n>1 else torch.eye(20,device=scores.device,dtype=torch.float64)[None]
        entropy=-(soft*soft.clamp_min(torch.finfo(torch.float64).tiny).log()).sum(-1).mean()
        gaps=[t['gap'] for t in trace]
        rows.append(dict(n_agents=n,population=population(n),attempted=len(trace),accepted=sum(t['accepted'] for t in trace),
            rejected=sum(not t['accepted'] for t in trace),strict_ties=sum(t['reason']=='strict_tie' for t in trace),
            reliable_decreases=sum(t['reliable_decrease'] for t in trace),roundoff_uncertain=sum(t['reason']=='roundoff_uncertain' for t in trace),
            native_not_exact_optimal=sum(not t['native_solution_exactly_optimal'] for t in trace),
            energy_before_fp64=trace[0]['energy_before_fp64'] if trace else 0.,
            energy_after_fp64=trace[-1]['energy_after_fp64'] if trace else 0.,
            accepted_energy_delta=sum(t['energy_delta_fp64'] for t in trace if t['accepted']),
            final_proposal_effective_score_RMS=float(effective.square().mean().sqrt()),
            proposal_matching_gap_mean=sum(gaps)/len(gaps) if gaps else 0.,proposal_matching_gap_min=min(gaps) if gaps else 0.,
            sinkhorn_row_entropy=float(entropy),sinkhorn_normalized_entropy=float(entropy)/__import__('math').log(20),
            soft_kind='native proposal relaxation; single-person identity',
            certified_monotone=all(t['energy_after_fp64']<=t['energy_before_fp64'] and (not t['accepted'] or (t['unique'] and t['reliable_decrease'])) for t in trace)))
    return rows


def gradient_statistics(named):
    out={}
    for prefix in PREFIXES:
        tensors=[p.grad for n,p in named.items() if n.startswith(prefix) and p.grad is not None]
        if any(not torch.isfinite(g).all() for g in tensors):raise FloatingPointError(prefix+' nonfinite gradient')
        out[prefix]=dict(gradient_l2=sum(float(g.detach().double().square().sum()) for g in tensors)**.5,
            present_tensors=len(tensors),nonzero_tensors=sum(bool(g.any()) for g in tensors))
    return out


class StepObserver:
    def __init__(self,model,path,start_update=0):
        self.model=model;self.path=Path(path);self.update=start_update
        self.named={n:p for n,p in model.named_parameters() if n.startswith(PREFIXES)}
        self.session=f'{__import__("os").getpid()}-{time.time_ns()}';self.groups=[];self.before={};self.metrics=None

    def capture_backward(self,metrics,seconds,peak_bytes):
        self.metrics=metrics;self.grad=gradient_statistics(self.named)
        self.before={n:p.detach().clone() for n,p in self.named.items()}
        self.seconds=seconds;self.peak_bytes=peak_bytes
        for prefix,v in self.grad.items():metrics[prefix+'gradient_l2']=v['gradient_l2']
        return self.grad

    def step_done(self,*args):
        if not self.before or self.metrics is None:raise RuntimeError('optimizer updated without observed grouped backward')
        self.update+=1;modules={}
        for prefix in PREFIXES:
            delta=[(p.detach()-self.before[n]).double() for n,p in self.named.items() if n.startswith(prefix)]
            value=dict(self.grad[prefix],parameter_delta_l2=sum(float(x.square().sum()) for x in delta)**.5,
                parameter_delta_max=max(float(x.abs().max()) for x in delta))
            modules[prefix]=value;self.metrics[prefix+'parameter_delta_l2']=value['parameter_delta_l2']
        row=dict(update=self.update,epoch=(self.update-1)//168+1,session=self.session,unix=time.time(),modules=modules,
            logical_group_forward_backward_seconds=self.seconds,allocated_peak_bytes=self.peak_bytes,physical_groups=self.groups)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        self.before={};self.metrics=None;self.groups=[]

    @contextlib.contextmanager
    def native_optimizer_hook(self):
        from bcep_v2 import state as native
        original=native.optimizer
        def factory(*a,**kw):
            opt=original(*a,**kw);opt.register_step_post_hook(self.step_done);return opt
        native.optimizer=factory
        try:yield
        finally:native.optimizer=original

    def epoch_summary(self,epoch):
        # Retain every journal attempt. Later replays supersede uncommitted
        # entries with the same logical update number after last.pt recovery.
        selected={}
        if self.path.exists():
            for line in self.path.open():
                r=json.loads(line)
                if (epoch-1)*168<r['update']<=epoch*168:selected[r['update']]=r
        if len(selected)!=168:raise RuntimeError('incomplete SSEC per-update journal for full epoch')
        rr=list(selected.values());modules={}
        for prefix in PREFIXES:
            modules[prefix]={key:sum(r['modules'][prefix][key] for r in rr)/len(rr) for key in ['gradient_l2','parameter_delta_l2','parameter_delta_max']}
        scenes=[s for r in rr for group in r['physical_groups'] for s in group['scenes']]
        decisions={key:sum(s[key] for s in scenes) for key in ['attempted','accepted','rejected','strict_ties','reliable_decreases','roundoff_uncertain','accepted_energy_delta']}
        costs=[]
        for name in ['1','2-8','9-19','20-41','42-57']:
            groups=[g for r in rr for g in r['physical_groups'] if population(max(g['valid_people']))==name]
            if groups:costs.append(dict(max_people_bin=name,physical_groups=len(groups),windows=sum(len(g['scenes']) for g in groups),coupling_forward_seconds=sum(g['coupling_forward_seconds'] for g in groups)))
        return dict(epoch=epoch,optimizer_updates=168,modules_mean_per_update=modules,decision_totals=decisions,
            coupling_cost_by_physical_group_max_people=costs,
            forward_backward_seconds=sum(r['logical_group_forward_backward_seconds'] for r in rr),
            allocated_peak_bytes=max(r['allocated_peak_bytes'] for r in rr),journal_update_first=min(selected),journal_update_last=max(selected))
