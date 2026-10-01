import json

with open("llm_responses.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for d in data:
    if d["final_answer"] == "":
        print(f"==================== {d['id']} ====================")
        print("Last 400 chars of raw_response:")
        print(d["raw_response"][-400:])
        print("\nAny line containing 'FINAL' or 'ANSWER':")
        for line in d["raw_response"].split("\n"):
            if "final" in line.lower() or "answer" in line.lower():
                print("  LINE:", line)
