from common import *
import pytest,xml.etree.ElementTree as ET
setup();start=time.perf_counter()
code=pytest.main(['-q','tests/test_bcep_scene_energy.py','tests/test_ssec_identity.py','tests/test_ssec_formal.py','--junitxml='+str(PUBLIC/'CPU_TESTS.xml')])
tree=ET.parse(PUBLIC/'CPU_TESTS.xml')
for e in tree.getroot().iter('testsuite'):e.attrib.pop('hostname',None)
tree.write(PUBLIC/'CPU_TESTS.xml',encoding='utf-8',xml_declaration=True)
write(PUBLIC/'CPU_RECEIPT.json',dict(passed=code==0,tests=sum(int(e.get('tests',0)) for e in tree.getroot().iter('testsuite')),
    seconds=time.perf_counter()-start,optimizer_guard=GUARD,real_optimizer_updates=0,synthetic_optimizer_updates=0,
    checkpoint_writer_test='actual native nested persist body; explicitly labelled writer fixtures, no training epochs',
    stop_rule_test='actual native AST expressions, no training loop/optimizer execution'))
raise SystemExit(code)
