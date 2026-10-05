"""Pre-fetch bounded control, attached to the existing trainer update branches."""
from collections import Counter
import time
import math
from pathlib import Path
from src.p2_protocol import digest,path

class BoundedController:
    def __init__(self,attempts,native,total_steps,*,clock=time.monotonic,memory=lambda:0,
                 arm_seconds=900,stage_seconds=2700,reserved_bytes=10*1024**3,
                 stage_start=None,order=None,pair_failure_path=None):
        if min(attempts,native,total_steps)<=0 or attempts>native:raise ValueError('bounded positive cap/native')
        self.cap=attempts;self.native=native;self.total_steps=total_steps
        self.clock=clock;self.memory=memory;self.start=clock()
        self.stage_start=self.start if stage_start is None else stage_start
        self.arm_seconds=arm_seconds;self.stage_seconds=stage_seconds;self.reserved_bytes=reserved_bytes
        self.counts=Counter({k:0 for k in ['fetch_attempts','batch_fetch','loss_attempts','backward','optimizer_attempts','successful_updates','skipped','failed','unknown','loss_samples']})
        self.reason=None;self.exposure=[];self.order=order
        self.pair_failure_path=pair_failure_path
        self.last_loss=None;self.first_loss=None;self.peak_reserved=0
        self.first_components=None;self.last_components=None
        self.update_norms=[];self.gradient_norms=[];self.scene_sizes=[]
    def components(self,losses,diagnostics):
        values={k:(float(v.detach().cpu()) if math.isfinite(float(v.detach().cpu())) else str(float(v.detach().cpu())))
                for k,v in losses.items()}
        self.last_components=values
        if self.first_components is None:self.first_components=values
        self.scene_sizes.append({k:float(diagnostics[k]) for k in
            ('num_agents','num_edges','degree_positive_agent_count') if k in diagnostics})
    def before_optimizer(self,parameters):
        parameters=list(parameters)
        self.gradient_norms.append(sum(float(p.grad.detach().double().square().sum().cpu())
            for p in parameters if p.grad is not None)**.5)
        return [(p,p.detach().clone()) for p in parameters if p.requires_grad]
    def after_optimizer(self,before):
        self.update_norms.append(sum(float((p.detach()-old).double().square().sum().cpu())
            for p,old in before)**.5)
    def resources(self):
        now=self.clock();mem=self.memory();self.peak_reserved=max(self.peak_reserved,mem)
        if self.pair_failure_path and Path(self.pair_failure_path).exists():
            self.reason='PAIR_ALREADY_STOPPED';return False
        if now-self.start>=self.arm_seconds:self.reason='ARM_WALL_LIMIT'
        elif now-self.stage_start>=self.stage_seconds:self.reason='STAGE_WALL_LIMIT'
        elif mem>self.reserved_bytes:self.reason='MEMORY_LIMIT'
        return self.reason is None
    def batches(self,loader):
        iterator=None
        while self.reason is None:
            if self.counts['fetch_attempts']>=self.cap:self.reason='ATTEMPT_CAP';break
            if self.counts['batch_fetch']>=self.native:self.reason='NATIVE_EPOCH';break
            if not self.resources():break
            if iterator is None:iterator=iter(loader)
            self.counts['fetch_attempts']+=1
            try:batch=next(iterator)
            except StopIteration:self.reason='EMPTY_OR_SHORT_LOADER';self.counts['unknown']+=1;break
            self.counts['batch_fetch']+=1
            if self.order is not None:
                actual=batch[1]['p2_window_ids']
                if actual!=[self.order[self.counts['batch_fetch']-1]]:
                    self.fail('IDENTITY_MISMATCH');raise RuntimeError('P2 ordered batch identity')
                self.exposure.extend(actual)
            yield batch
        # Resource failures must stop the other arm as well.
        if self.reason not in {'ATTEMPT_CAP','NATIVE_EPOCH'}:self.publish_failure()
    def before_loss(self):self.counts['loss_attempts']+=1
    def loss(self,value):
        self.counts['loss_samples']+=1
        encoded=float(value) if math.isfinite(float(value)) else str(value)
        self.last_loss=encoded
        if self.first_loss is None:self.first_loss=encoded
    def outcome(self,success):
        self.counts['successful_updates' if success else 'skipped']+=1
        if not success:self.reason='SKIPPED_UPDATE';self.publish_failure()
        elif not self.resources():self.publish_failure()
    def fail(self,reason):
        self.reason=reason
        if not self.counts['failed']:self.counts['failed']+=1
        self.publish_failure()
    def publish_failure(self):
        if self.pair_failure_path:
            if Path(self.pair_failure_path).exists():
                return  # preserve the originating arm's failure, never overwrite it
            from src.joint_dependency_v2_cache import atomic_json_save
            atomic_json_save(self.ledger(),self.pair_failure_path)
    @property
    def complete_epoch(self):
        return self.counts['successful_updates']==self.native and self.reason in {'ATTEMPT_CAP','NATIVE_EPOCH',None}
    def ledger(self):
        return dict(counts=dict(self.counts),stop_reason=self.reason,native_epoch=self.native,
            epoch_complete=self.complete_epoch,partial_epoch=not self.complete_epoch,
            before_progress=0.,after_progress=self.counts['successful_updates']/self.total_steps,
            reference_total_steps=self.total_steps,exposure=self.exposure,exposure_hash=digest(self.exposure),
            batch_cursor=self.counts['batch_fetch'],first_loss=self.first_loss,last_loss=self.last_loss,
            first_components=self.first_components,last_components=self.last_components,
            gradient_norms=self.gradient_norms,update_norms=self.update_norms,scene_sizes=self.scene_sizes,
            loss_denominator=self.counts['loss_samples'],success_denominator=self.counts['successful_updates'],
            peak_reserved=self.peak_reserved,elapsed=self.clock()-self.start,
            limits_checked_at_boundaries_only=True,resumable=False)

def controller_for(owner):
    args=owner.args;reg=owner.p2_registry
    pair=reg.manifest['pair_run']
    import torch
    return BoundedController(args.p2_max_update_attempts,len(owner.data_loaders['train']),
        args.p2_reference_total_steps,arm_seconds=args.p2_arm_seconds,
        stage_seconds=args.p2_stage_seconds,reserved_bytes=args.p2_reserved_bytes,
        stage_start=pair['monotonic_start'],order=reg.manifest['train_order'],
        pair_failure_path=str(path(pair['failure_path'])),
        memory=lambda:torch.cuda.max_memory_reserved(owner.device) if owner.device.type=='cuda' else 0)

def finish_bounded(owner,error=None):
    from src.jdv2_objective_state import atomic_save
    from src.joint_dependency_v2_cache import atomic_json_save
    c=owner.p2_controller
    if error is not None:c.fail(type(error).__name__+': '+str(error))
    ledger=c.ledger()
    from src.p2_checkpoint import state_hash
    frozen_names={name for name,p in owner.net.named_parameters() if not p.requires_grad}
    final_frozen=state_hash({k:v for k,v in owner.net.state_dict().items() if k in frozen_names})
    ledger['frozen_before']=getattr(owner,'p2_frozen_before',None)
    ledger['frozen_after']=final_frozen
    ledger['frozen_equal']=ledger['frozen_before']==final_frozen
    if ledger['frozen_before'] is not None and not ledger['frozen_equal']:
        c.fail('FROZEN_WEIGHT_CHANGED');ledger.update(c.ledger())
    rng=getattr(owner,'jdv2_objective_rng',None)
    if rng is not None:
        ledger['mc']=dict(draw_calls=rng.draw_calls,sampled_agent_draws=rng.sampled_agent_draws,
            pending_backward=rng.pending_backward,successful_optimizer_updates=rng.successful_optimizer_updates)
    out=Path(owner.args.model_dir);out.mkdir(parents=True,exist_ok=True)
    atomic_json_save(ledger,str(out/'bounded_stop_ledger.json'))
    if error is None and c.reason in {'ATTEMPT_CAP','NATIVE_EPOCH'}:
        target=out/'bounded_stop_weights_only.pt'
        if target.exists():raise RuntimeError('bounded snapshot already exists; no overwrite')
        atomic_save(dict(artifact_role='bounded_stop_weights_only',resumable=False,
            model_state_dict=owner.net.state_dict(),ledger=ledger),str(target))
    return ledger
