"""Final selected-only DDIM replay plus same-run independent I/H0 controls."""
from dataclasses import asdict
import csv
import hashlib
import time
import torch
from bcep_joint.common import write
from bcep_joint.inference import generate as native_generate
from ebjd.metrics import hard_metric_statistics
from .inputs import RUN, LIMITS
from .inference import generate
from .cache import cpu_statistics, accumulate
from .train import score_records


def csvwrite(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def paired_summary(rows, group):
    subset = [r for r in rows if group == 'all' or r['agents'] >= 9]
    selected = [r for r in subset if r['switch_to_I']]
    result = dict(group=group, windows=len(subset), switched=len(selected))
    for metric in ('JADE', 'JFDE'):
        changes = [r[f'actual_minus_H0_{metric}'] for r in selected]
        result.update({f'{metric}_improved': sum(v < 0 for v in changes),
                       f'{metric}_worsened': sum(v > 0 for v in changes),
                       f'{metric}_tied': sum(v == 0 for v in changes),
                       f'mean_switch_contribution_{metric}': sum(r[f'switch_contribution_{metric}'] for r in subset)/len(subset),
                       f'mean_replay_residual_{metric}': sum(r[f'replay_residual_{metric}'] for r in subset)/len(subset),
                       f'max_abs_replay_residual_{metric}': max(abs(r[f'replay_residual_{metric}']) for r in subset)})
    return result


@torch.no_grad()
def evaluate(inputs, model, selector, cache, device):
    cached_switches, cached_delta = score_records(selector, cache, device)
    records = []
    rows = []
    start = time.monotonic()
    for pos, idx in enumerate(inputs.split['selection_dev_indices']):
        batch, goals = inputs.batch('selection_dev', idx, device, truth=False)
        # Actual inference chooses before decoding, and has no future argument.
        selected, detail = generate(model, selector, batch.observed, batch.semantic_maps, batch.valid,
            batch.scene_ids, batch.agent_ids, goals)
        assert detail['ddim_calls'] == 1
        switch = bool(detail['switch'].item())
        assert switch == cached_switches[pos]
        torch.testing.assert_close(detail['delta'].cpu(), cached_delta[pos:pos+1], rtol=0, atol=0)
        # Independent forced controls. Native H0 is the original inference API.
        _, h0, hp, _ = native_generate(model, batch.observed, batch.semantic_maps, batch.valid,
            batch.scene_ids, batch.agent_ids, goals, 2035, mode='H')
        initial, di = generate(model, None, batch.observed, batch.semantic_maps, batch.valid,
            batch.scene_ids, batch.agent_ids, goals, force='I')
        assert torch.equal(hp, detail['feedback_pi'])
        assert torch.equal(di['initial_pi'], detail['initial_pi'])
        truth = inputs.data.batch([idx], 'selection_dev', truth=True).future.to(device)
        stats = {name: cpu_statistics(hard_metric_statistics(path, truth, batch.valid))
                 for name, path in [('I', initial), ('H0', h0), ('actual', selected)]}
        records.append(dict(statistics=stats))
        row = dict(window=f'dev_{pos+1:04d}', agents=int(batch.valid.sum()), switch_to_I=int(switch),
                   identical_permutations=int(torch.equal(detail['initial_pi'], detail['feedback_pi'])),
                   dA_hat=float(detail['delta'][0, 0]), dF_hat=float(detail['delta'][0, 1]))
        for metric, key in [('JADE', 'scene_jade'), ('JFDE', 'scene_jfde')]:
            for branch in stats:
                row[f'{branch}_{metric}'] = float(stats[branch][key])
            row[f'd{metric}_label'] = row[f'I_{metric}']-row[f'H0_{metric}']
            row[f'actual_minus_H0_{metric}'] = row[f'actual_{metric}']-row[f'H0_{metric}']
            row[f'switch_contribution_{metric}'] = switch*row[f'd{metric}_label']
            row[f'replay_residual_{metric}'] = row[f'actual_minus_H0_{metric}']-row[f'switch_contribution_{metric}']
            row[f'H0_minus_cache_{metric}'] = row[f'H0_{metric}']-float(cache[pos]['statistics']['H0'][key])
        rows.append(row)
        if (pos+1) % 50 == 0 or pos+1 == len(cache):
            progress = dict(phase='FINAL_SINGLE_ROUTE_REPLAY', windows=pos+1, total=len(cache), seconds=time.monotonic()-start)
            write(RUN/'PROGRESS.json', progress)
            print(progress, flush=True)
    groups = {branch: accumulate([dict(statistics={'H0': r['statistics'][branch]}) for r in records])
              for branch in ('I', 'H0', 'actual')}
    values = [groups['actual']['all']['JADE'], groups['actual']['all']['JFDE'], groups['actual']['N_ge_9']['JFDE']]
    passes = [values[0] <= LIMITS[0], values[1] < LIMITS[1], values[2] < LIMITS[2]]
    decision = dict(values=values, thresholds=LIMITS, differences=[x-y for x, y in zip(values, LIMITS)],
                    passes=passes, all_pass=all(passes),
                    decision='ZARA2 single-fold pass' if all(passes) else 'retain old coordinate; terminate route',
                    additional_seeds=0, additional_folds=0, threshold_search=False, test_access=False)
    summaries = [paired_summary(rows, group) for group in ('all', 'N_ge_9')]
    csvwrite(RUN/'DEV_WINDOWS.csv', rows)
    csvwrite(RUN/'PAIRED_SUMMARY.csv', summaries)
    write(RUN/'FINAL_METRICS.json', groups)
    write(RUN/'DECISION.json', decision)
    write(RUN/'FINAL_REPLAY.json', dict(windows=590, dense_windows=127, selected_ddim_calls=590,
        selected_steps_per_call=20, offline_control_ddim_calls=1180, exact_cached_decisions=True,
        frozen_model_eval=True, future_loaded_after_selection=True, seconds=time.monotonic()-start))
    return decision
