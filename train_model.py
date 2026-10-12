from pathlib import Path
import json
from src.model import train_and_evaluate
ROOT=Path(__file__).resolve().parent
metrics=train_and_evaluate()
report=ROOT/'reports'/'metrics.json'
report.parent.mkdir(exist_ok=True)
report.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in metrics.items() if k!='classification_report'},indent=2))
print(f"Full report saved to: {report}")
