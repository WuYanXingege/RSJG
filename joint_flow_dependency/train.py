"""One likelihood fit with real trajectory validation; native atomic Top5 writer."""
import torch
from bcep.contracts import canonical_hash, digest
from bcep_joint.common import read, write
from bcep_v2.state import run_stage
from .inputs import RUN, RECIPE
from .model import delta_nll
from .evaluation import evaluate_epoch


def train(inputs, model, provider, dependence, cache, identity, device):
    folder = RUN/'dependence'
    last = folder/'last.pt'
    train_records = {r['index']: r for r in cache['train']}
    identity = dict(identity, stage='joint', fold='zara2', experiment='continuous_joint_flow',
                    selector='all-dev B JADE only; exact ties earlier completed epoch')
    completed = [torch.load(last, map_location='cpu', weights_only=False)['completed_epoch'] if last.exists() else 0]

    def backward(group, order_rng, view_rng):
        del order_rng, view_rng
        total = 0.
        for index in group:
            record = train_records[index]
            observed_inputs = {k: v.to(device) for k, v in record['inputs'].items()}
            covariance = dependence(**observed_inputs)['R']
            loss = delta_nll(covariance, record['latent_truth'].to(device))
            assert torch.isfinite(loss)
            (loss/len(group)).backward()
            total += float(loss.detach())
        return dict(delta_nll_nat_per_person=total/len(group), equally_weighted_windows=len(group))

    def validate():
        epoch = completed[0]+1
        metrics = evaluate_epoch(inputs, model, provider, dependence, cache['selection_dev'], device, epoch)
        validate.metrics = metrics
        completed[0] = epoch
        return metrics['all']['JADE']

    # Only this dependency module is supplied to the native writer/optimizer.
    assert all(p.requires_grad for p in dependence.parameters())
    assert not any(p.requires_grad for p in list(model.parameters())+list(provider.parameters()))
    state = run_stage(folder, identity, RECIPE, dependence, list(train_records), backward, validate,
                      16800, resume=last.exists(), record_progress=True)
    if state['status'] != 'COMPLETE':
        raise RuntimeError('Recover this same run from last: '+state['status'])
    rank = min(state['history'], key=lambda row: (row['validation'], row['epoch']))
    path = folder/rank['file']
    saved = torch.load(path, map_location='cpu', weights_only=False)
    dependence.load_state_dict(saved['model'], strict=True)
    assert digest(path) == digest(folder/'best.pt')
    write(RUN/'TRAINING_RESULT.json', dict(completed_epochs=state['completed_epoch'], updates=state['update'],
        stop_reason=state['stop_reason'], rank1_epoch=rank['epoch'], rank1_sha256=digest(path),
        rank1_metrics=rank['metrics'], recipe=RECIPE, optimizer_parameter_names=state['trainable']))
    return dependence.eval()
