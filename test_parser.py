import json
import re

with open("llm_responses.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def parse_answer_flexible(text):
    if not text:
        return ""
    # 1. Look for FINAL ANSWER: <something> or on subsequent lines
    m = re.findall(r"(?:FINAL ANSWER|Answer):\s*([^\n\r]+)", text, re.IGNORECASE)
    if m:
        candidate = m[-1].strip()
        candidate = re.sub(r"[\$\*\`]", "", candidate).strip()
        if candidate:
            return candidate
            
    # 2. Look for \boxed{...}
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        cand = boxed[-1].strip()
        cand = re.sub(r"[\$\*\`]", "", cand).strip()
        return cand

    # 3. Look for FINAL ANSWER followed by next non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for idx, l in enumerate(lines):
        if re.search(r"FINAL ANSWER", l, re.IGNORECASE):
            if idx + 1 < len(lines):
                next_l = lines[idx + 1]
                # clean math formatting
                next_l = re.sub(r"[\$\*\`\(\)\\]", "", next_l)
                next_l = re.sub(r"\\boxed\{([^}]+)\}", r"\1", next_l)
                return next_l.strip()
                
    return ""

for d in data:
    parsed = parse_answer_flexible(d["raw_response"])
    print(f"{d['id']}: '{parsed}' (expected {d.get('key', '')})")
