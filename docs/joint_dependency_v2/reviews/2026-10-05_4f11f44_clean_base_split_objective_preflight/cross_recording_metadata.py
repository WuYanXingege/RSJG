"""Additional metadata-only check; no tensor or coordinate reads."""
import json
from pathlib import Path
import sys
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parents[3]))
from tools.jdv2_clean_objective_preflight import recording_overlap
manifest=json.loads((D/'SPLIT_ROLE_MANIFEST.json').read_text())
records=[json.loads(line) for part in manifest['record_files'] for line in (D/part['path']).read_text().splitlines()]
print(json.dumps({'pairs':recording_overlap(records),'metadata_files_read':1+len(manifest['record_files']),
    'coordinates_or_future_tensors_read':0,'independent_recordings':'UNKNOWN',
    'interpretation':'Source-qualified exact overlap is zero; these same-scene identifier collisions are not content duplicate proof. Obtain recording/ID namespace provenance without opening protected future labels.'},indent=2))
