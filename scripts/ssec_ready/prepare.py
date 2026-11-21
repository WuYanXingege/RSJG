from common import *
from bcep_scene_energy.formal import main
setup();main(['prepare'])
frozen=read(ROOTS['run']/'FROZEN_CONFIG.json')
assert frozen['implementation_hash']==read(PUBLIC/'REAL_ACCEPTANCE.json')['implementation_hash']==read(PUBLIC/'RECOVERY_RECEIPT.json')['implementation_hash']
for name in ['FROZEN_CONFIG.json','INITIALIZATION_ORIGIN.json']:
    (PUBLIC/name).write_bytes((ROOTS['run']/name).read_bytes())
write(PUBLIC/'SOURCE_SHA.json',dict(base=BASE,implementation_hash=frozen['implementation_hash'],files=frozen['implementation_files']))
(PUBLIC/'CONFIG_SHA256.txt').write_text('file_sha256 '+file_sha(ROOTS['run']/'FROZEN_CONFIG.json')+'\ncanonical_sha256 '+canonical_hash(frozen)+'\n')
print('Frozen runtime source and configuration ready; no training launched',flush=True)
