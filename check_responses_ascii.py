import json
import re

with open("llm_responses.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for d in data:
    resp = d["raw_response"]
    print(f"\n==================== {d['id']} ====================")
    # Check lines with boxed, final, answer, or last 3 non-empty lines
    lines = [line.strip() for line in resp.split("\n") if line.strip()]
    relevant = [l for l in lines if any(w in l.lower() for w in ["final", "answer", "boxed", "result", "statistic"])]
    print("Relevant lines:")
    for l in relevant[-4:]:
        print("  ", l.encode('ascii', 'replace').decode())
    print("Last 2 lines:")
    for l in lines[-2:]:
        print("  ", l.encode('ascii', 'replace').decode())
