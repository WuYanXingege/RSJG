"""Pure reporting checks; no optimizer or official data loading."""
import copy,hashlib,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts/v3_trial'))
import report as trial


def reference():
    import csv,io
    blob=trial.gitblob(trial.REFERENCE_COMMIT,trial.REFERENCE_DIR+'OLD5_REFERENCE_WINDOWS.csv')
    meta=json.loads(trial.gitblob(trial.REFERENCE_COMMIT,trial.REFERENCE_DIR+'REFERENCE_REPLAY.json'))
    assert hashlib.sha256(blob).hexdigest()==meta['csv_sha256']
    return list(csv.DictReader(io.StringIO(blob.decode())))


def test_same_window_pairing_and_equal_counts():
    old=reference();new=[dict(r,G_JFDE=0.) for r in old]
    rows=trial.paired_rows(new,old);summary=trial.summaries(rows)
    assert len(rows)==590
    for r in summary:
        assert r['mean_delta_m']==r['median_delta_m']==0
        assert r['improved_windows']==r['regressed_windows']==0
        assert r['equal_windows']=={'all':590,'multi':501,'N_ge_9':127}[r['group']]
    bad=copy.deepcopy(new);bad[0]['minFDE']=float(bad[0]['minFDE'])+.01
    with pytest.raises(AssertionError):trial.paired_rows(bad,old)
    with pytest.raises(ValueError):trial.paired_rows(new[:-1],old)


def test_feedback_direction_and_decomposition():
    old=reference();new=[dict(r,G_JFDE=0.) for r in old]
    for r in new:
        r['I_JFDE']=float(r['I_JFDE'])+.03
        r['H_JFDE']=float(r['H_JFDE'])+.01
        r['feedback_gain']=float(r['feedback_gain'])+.02
    rows=trial.paired_rows(new,old);s={(r['group'],r['metric']):r for r in trial.summaries(rows)}
    assert s['all','H_JFDE']['mean_delta_m']==pytest.approx(.01)
    assert s['all','feedback_gain']['improved_windows']==590
    assert s['all','H_JFDE']['regressed_windows']==590


def test_resource_phase_bins_are_labeled_samples():
    h=[dict(epoch=1,update=168,train_seconds=10.,validation_seconds=5.)]
    e=[dict(epoch=1,unix=106.,pid=10)]
    t=[dict(update=168,unix=100.)]
    samples=[dict(pid=10,unix=x,gpu_process_mib=y) for x,y in [(91.,100),(99.,120),(102.,150),(105.,130)]]
    r=trial.resource_rows(h,e,samples,t)[0]
    assert r['train_gpu_process_sampled_peak_mib']==120
    assert r['validation_gpu_process_sampled_peak_mib']==150
    assert r['train_samples']==r['validation_samples']==2


def test_refuse_final_report_until_complete(monkeypatch):
    monkeypatch.setattr(trial,'read',lambda path:dict(status='RUNNING'))
    with pytest.raises(AssertionError):trial.main()
