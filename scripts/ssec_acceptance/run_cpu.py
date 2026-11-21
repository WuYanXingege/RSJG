from common import *
setup()
import pytest

if __name__=='__main__':
    start=time.time()
    code=pytest.main(['-q','tests/test_bcep_scene_energy.py','--junitxml='+str(DOC/'CPU_TESTS.xml')])
    assert GUARD['optimizer_step_calls']==0
    write(DOC/'CPU_RECEIPT.json',dict(passed=code==0,seconds=time.time()-start,optimizer_guard=GUARD,real_updates=0,synthetic_updates=0))
    sys.exit(code)
