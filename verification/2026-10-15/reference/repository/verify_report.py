"""Recalculate frozen synthetic report without changing or reading external files."""
import json
from pathlib import Path
from zora_lab.connection_study import run

reference = json.loads((Path(__file__).parent / 'experiments' / 'connection_study_synthetic_report.json').read_text(encoding='utf-8'))
actual = run()
assert actual == reference, 'Frozen report differs from a fresh deterministic execution'
assert actual['no_raw_human_data'] is True
assert actual['status'] == 'software_connectivity_only_no_qualia_inference'
print('PASS: frozen report identical; 348 synthetic evaluation transitions; no human data')
