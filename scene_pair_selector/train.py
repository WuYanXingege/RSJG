"""One seed, one MSE fit; native complete-state writer and JADE selector."""
import time
import torch
from bcep.contracts import canonical_hash, digest
from bcep_joint.common import seed_all, read, write
from bcep_v2.state import run_stage
from .inputs import RUN, RECIPE
from .model import SceneScorer, choose_initial
from .cache import accumulate, fit_normalization


def move_features(record, device):
    return {k: v.to(device) for k, v in record['features'].items()}


@torch.no_grad()
def score_records(selector, records, device):
    switches, predictions = [], []
    selector.eval()
    for record in records:
        f = move_features(record, device)
        delta = selector.difference(**f)
        switch = choose_initial(delta, record['initial_pi'].to(device), record['feedback_pi'].to(device), f['valid'])
        switches.append(bool(switch.item()))
        predictions.append(delta.cpu())
    return switches, torch.cat(predictions)


def train(records, identity, device):
    seed_all(3201)
    selector = SceneScorer()
    normalization = fit_normalization(records['train'], selector)
    write(RUN/'NORMALIZATION.json', normalization)
    selector.to(device)
    train_records = {r['index']: r for r in records['train']}
    last = RUN/'selector/last.pt'
    identity = dict(identity, stage='joint', fold='zara2', experiment='scene_pair_selector',
                    selector='all dev selected JADE; exact ties earlier completed epoch')

    def backward(group, order_rng, view_rng):
        del order_rng, view_rng
        loss_sum = 0.
        for idx in group:
            record = train_records[idx]
            delta = selector.difference(**move_features(record, device))
            target = record['target'].to(device)
            loss = (delta-target).square().mean()
            (loss/len(group)).backward()
            loss_sum += float(loss.detach())
        return dict(two_output_mse=loss_sum/len(group), equally_weighted_windows=len(group))

    def validate():
        switches, predictions = score_records(selector, records['selection_dev'], device)
        groups = accumulate(records['selection_dev'], switches)
        validate.metrics = dict(all=groups['all'], N_ge_9=groups['N_ge_9'],
                                switches=sum(switches),
                                dense_switches=sum(s and int(r['features']['valid'].sum()) >= 9
                                                   for s, r in zip(switches, records['selection_dev'])))
        write(RUN/'LATEST_DEV.json', validate.metrics)
        return groups['all']['JADE']

    # run_stage constructs its optimizer from this module only. Frozen backbone
    # and Provider are absent from the optimizer and remain eval throughout.
    state = run_stage(RUN/'selector', identity, RECIPE, selector, list(train_records),
                      backward, validate, 16800, resume=last.exists(), record_progress=True)
    assert state['status'] == 'COMPLETE'
    rank = min(state['history'], key=lambda r: (r['validation'], r['epoch']))
    path = RUN/'selector'/rank['file']
    saved = torch.load(path, map_location='cpu', weights_only=False)
    selector.load_state_dict(saved['model'], strict=True)
    assert digest(path) == digest(RUN/'selector/best.pt')
    write(RUN/'TRAINING_RESULT.json', dict(completed_epochs=state['completed_epoch'],
        updates=state['update'], stop_reason=state['stop_reason'], rank1_epoch=rank['epoch'],
        rank1_sha256=digest(path), rank1_cache_metrics=rank['metrics'],
        optimizer_parameter_names=state['trainable'], recipe=RECIPE))
    return selector.eval()
