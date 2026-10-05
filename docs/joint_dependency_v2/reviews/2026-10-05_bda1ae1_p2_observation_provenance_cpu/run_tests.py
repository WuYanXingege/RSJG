"""Guarded synthetic pytest recorder. No model/optimizer tests are selected."""
import json,sys,time,contextlib,io
from pathlib import Path
R=Path(__file__).resolve().parents[4];sys.path[:0]=[str(R),str(R/'tests')]
from src.p2_observation_guard import install
permit,report=install()
import pytest
import test_p2_observation_artifacts as fixture
fixture.PERMIT=permit
rows=[]
class Reporter:
    def pytest_runtest_logreport(self,report):
        r=report
        if r.when=='call' or r.failed:
            rows.append(dict(nodeid=r.nodeid,outcome=r.outcome,when=r.when,seconds=r.duration,
                error=str(r.longrepr) if r.failed else None))
out=Path(sys.argv[1])
if out.exists():raise RuntimeError('new test result file required')
start=time.monotonic();stream=io.StringIO()
with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
    try:
        code=pytest.main(['-q','-p','no:cacheprovider',*sys.argv[2:]],plugins=[Reporter()])
    except BaseException:
        import traceback
        traceback.print_exc();code=3
result=dict(exit_code=int(code),seconds=time.monotonic()-start,tests=rows,guard=report(),
    stdout=stream.getvalue(),command=sys.argv,real_targets=0,real_future_queries=0,
    synthetic_full20_and_target=True)
out.write_text(json.dumps(result,indent=2)+'\n')
from collections import Counter
print(json.dumps(dict(exit_code=int(code),outcomes=dict(Counter(x['outcome'] for x in rows)),guard=result['guard'],record=str(out))))
raise SystemExit(code)
