"""P1/P2 optional diagnostic collection; all loss/backward/RNG order retained."""
import math
import torch

from bcep_v2.stages import augment, joint_callbacks as native_callbacks
from bcep_v2.objective import bridge_and_pair_loss
from bcep_v3.coupling import HCRACoupling


class ScalarJournal:
    """Transfer detached scalar values once; replay original Python arithmetic.

    No graph is kept alive by the journal. Nonfinite diagnostics are rejected
    before returning the logical backward, hence before clip/optimizer.step.
    """
    def __init__(self, deferred=False):
        self.deferred = deferred
        self.records = []
        self.metrics = {}
        self.host_transfers = 0

    def add(self, key, value, multiplier=1, divisor=1, *, count=False):
        if self.deferred:
            if torch.is_tensor(value):
                if value.numel() != 1:
                    raise ValueError('diagnostic must be scalar')
                value = value.detach().reshape(())
            self.records.append((key,value,multiplier,divisor,count))
        else:
            raw = value.detach() if torch.is_tensor(value) else value
            v = raw if count else float(raw)*multiplier/divisor
            self.metrics[key] = self.metrics.get(key,0) + v

    def finish(self):
        if not self.deferred:
            return self.metrics
        tensors = [v for _,v,_,_,_ in self.records if torch.is_tensor(v)]
        if tensors:
            host = iter(torch.stack([v.to(torch.float64) for v in tensors]).cpu().tolist())
            self.host_transfers += 1
        else:
            host = iter(())
        for key,value,multiplier,divisor,count in self.records:
            v = next(host) if torch.is_tensor(value) else value
            if not math.isfinite(v):
                raise FloatingPointError('nonfinite logical-update diagnostic: '+key)
            v = v if count else float(v)*multiplier/divisor
            self.metrics[key] = self.metrics.get(key,0) + v
        self.records.clear()
        return self.metrics


def compact_checked(batch, a, b, device, check_on_cpu=False):
    if check_on_cpu:
        if batch.valid.device.type != 'cpu':
            raise ValueError('P2 requires CPU collation; do not silently copy a CUDA mask')
        physical = batch.compact_scenes(a,b)
        n = physical.valid.shape[1]
        if not torch.equal(batch.valid[a:b,:n],physical.valid):
            raise ValueError('noncanonical collator padding')
        return physical.to(device)
    physical = batch.compact_scenes(a,b).to(device)
    n = physical.valid.shape[1]
    if not torch.equal(batch.valid[a:b,:n],physical.valid.cpu()):
        raise ValueError('noncanonical collator padding')
    return physical


def joint_callbacks(model, data, device, config, training, validation, *,
                    defer_diagnostics=False, cpu_mask_check=False):
    original_backward,validate = native_callbacks(model,data,device,config,training,validation)
    if not defer_diagnostics and not cpu_mask_check:
        return original_backward,validate
    if not isinstance(model.coupling,HCRACoupling):
        raise TypeError('optional audit path is limited to the v3 coupling')

    def backward(ids, order, views):
        journal = ScalarJournal(defer_diagnostics)
        for start in range(0,len(ids),config['batch_scenes']):
            micro = ids[start:start+config['batch_scenes']]
            batch = data.batch(micro,'train')
            batch,goals,_ = augment(batch,training.batch(micro,batch),order)
            counts = batch.valid.sum(1).tolist()  # CPU collation; not a CUDA synchronization.
            times = torch.randint(1,101,(len(micro),),device=device).float()/100
            noise = torch.randn(len(micro),1,batch.valid.shape[1],12,2,device=device)
            a = 0
            while a < len(micro):
                b = a+1
                while b < len(micro) and max(counts[a:b+1])*(b+1-a) <= config['group_max_slots']:
                    b += 1
                physical = compact_checked(batch,a,b,device,cpu_mask_check)
                n = physical.valid.shape[1]
                _,_,_,agent,scene = bridge_and_pair_loss(
                    model,physical,goals[a:b,:,:n].to(device),time_draw=times[a:b],noise=noise[a:b,:,:n])
                fraction = (b-a)/len(micro)
                loss = (agent*sum(counts[a:b])/sum(counts)+scene*fraction)/config['gradient_accumulation']
                loss.backward()
                journal.add('loss',loss.detach())
                for key,value in model.last_pair_metrics.items():
                    w = (b-a)/len(ids) if key != 'effective_edges' else 1
                    journal.add(key,value,w)
                c = model.last_loss_components
                for key in ('clean_path_raw_sum','clean_path_signal_sum'):
                    journal.add(key,c[key])
                journal.add('valid_scene_count',b-a,count=True)
                for key in ('clean_path_term','diffusion_term','geometry_term','pair_term','map_term'):
                    w = sum(counts[a:b])/sum(counts) if key in ('diffusion_term','map_term') else fraction
                    journal.add(key,c[key],w,config['gradient_accumulation'])
                a = b
        return journal.finish()
    return backward,validate
