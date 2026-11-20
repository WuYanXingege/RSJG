"""No GPU subprocess or CUDA initialization is performed by these guard tests."""
import importlib.util
from pathlib import Path
import pytest

SPEC=importlib.util.spec_from_file_location('perf_benchmark',Path(__file__).resolve().parents[1]/'scripts/perf_audit/benchmark.py')
bench=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(bench)


def test_busy_formal_refused_before_gpu_query(monkeypatch,tmp_path):
    monkeypatch.setattr(bench.subprocess,'check_output',lambda *a,**kw:pytest.fail('GPU query must not occur'))
    with pytest.raises(RuntimeError,match='GPU_PROFILE_PENDING'):
        bench.admission(tmp_path/'new',environ={'RUN_SYNTHETIC_PERF':'true'},status={'status':'RUNNING'})
    assert not (tmp_path/'new').exists()


def test_authorization_and_busy_device_fail_closed(tmp_path):
    with pytest.raises(PermissionError):bench.admission(tmp_path/'x',environ={},status={'status':'COMPLETE'},compute_apps='')
    with pytest.raises(PermissionError):
        bench.admission(tmp_path/'x',environ={'RUN_SYNTHETIC_PERF':'true','RUN_ONE_ZARA2_V3':'true'},status={'status':'COMPLETE'},compute_apps='')
    with pytest.raises(RuntimeError,match='occupied'):
        bench.admission(tmp_path/'x',environ={'RUN_SYNTHETIC_PERF':'true'},status={'status':'COMPLETE'},compute_apps='123')


def test_only_new_performance_output_allowed(monkeypatch,tmp_path):
    monkeypatch.setattr(bench,'PERF_ROOT',tmp_path/'performance');(tmp_path/'performance').mkdir()
    kwargs=dict(environ={'RUN_SYNTHETIC_PERF':'true'},status={'status':'COMPLETE'},compute_apps='')
    assert bench.admission(tmp_path/'performance/new',**kwargs)==tmp_path/'performance/new'
    with pytest.raises(ValueError):bench.admission(tmp_path/'formal',**kwargs)
    (tmp_path/'performance/existing').mkdir()
    with pytest.raises(FileExistsError):bench.admission(tmp_path/'performance/existing',**kwargs)
