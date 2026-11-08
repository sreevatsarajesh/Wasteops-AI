import json
from pathlib import Path
p=Path('reports/metrics.json')
if not p.exists():raise SystemExit('No reports/metrics.json. Run scripts/train.py first.')
metrics=json.loads(p.read_text())
print(json.dumps({k:v for k,v in metrics.items() if k!='classification_report'},indent=2))
