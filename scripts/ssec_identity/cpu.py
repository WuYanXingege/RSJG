from common import *
import pytest,xml.etree.ElementTree as ET
setup();start=time.perf_counter()
code=pytest.main(['-q','tests/test_bcep_scene_energy.py','tests/test_ssec_identity.py','--junitxml='+str(DOC/'CPU_TESTS.xml')])
tree=ET.parse(DOC/'CPU_TESTS.xml')
for x in tree.getroot().iter('testsuite'):x.attrib.pop('hostname',None)
tree.write(DOC/'CPU_TESTS.xml',encoding='utf-8',xml_declaration=True)
write(DOC/'CPU_RECEIPT.json',dict(passed=code==0,seconds=time.perf_counter()-start,optimizer_guard=GUARD,
    old_tests_unchanged=True,tests=sum(int(x.get('tests',0)) for x in tree.getroot().iter('testsuite'))))
raise SystemExit(code)
