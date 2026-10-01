import json
import csv
import re
import math

with open("llm_responses.json", "r", encoding="utf-8") as f:
    responses = json.load(f)

with open("problems.json", "r", encoding="utf-8") as f:
    problems = json.load(f)

with open("p19_raw.txt", "r", encoding="utf-8") as f:
    p19_raw = f.read()

responses[18]["raw_response"] = p19_raw

def parse_answer_flexible(text, format_type):
    if not text:
        return ""
    m = re.findall(r"(?:FINAL ANSWER|Answer):\s*([^\n\r]+)", text, re.IGNORECASE)
    if m:
        cand = m[-1].strip()
        cand = re.sub(r"[\$\*\`]", "", cand).strip()
        if format_type == "mcq":
            let = re.search(r"^[A-D]\b", cand, re.IGNORECASE)
            if let:
                return let.group(0).upper()
        if cand:
            return cand

    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        cand = boxed[-1].strip()
        cand = re.sub(r"[\$\*\`]", "", cand).strip()
        if format_type == "mcq":
            let = re.search(r"^[A-D]\b", cand, re.IGNORECASE)
            if let:
                return let.group(0).upper()
        num_m = re.search(r"[-+]?\d*\.?\d+", cand)
        if num_m and format_type != "mcq":
            return num_m.group(0)
        return cand

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for idx, l in enumerate(lines):
        if re.search(r"FINAL ANSWER", l, re.IGNORECASE):
            if idx + 1 < len(lines):
                next_l = lines[idx + 1]
                next_l = re.sub(r"[\$\*\`\(\)\\]", "", next_l)
                next_l = re.sub(r"\\boxed\{([^}]+)\}", r"\1", next_l).strip()
                if format_type == "mcq":
                    let = re.search(r"^[A-D]\b", next_l, re.IGNORECASE)
                    if let:
                        return let.group(0).upper()
                num_m = re.search(r"[-+]?\d*\.?\d+", next_l)
                if num_m and format_type != "mcq":
                    return num_m.group(0)
                return next_l
    return ""

def wilson_score_interval(successes, total, confidence=0.95):
    z = 1.959963984540054
    p_hat = successes / total
    denom = 1 + (z**2) / total
    center = (p_hat + (z**2) / (2 * total)) / denom
    margin = (z * math.sqrt((p_hat * (1 - p_hat) / total) + (z**2) / (4 * (total**2)))) / denom
    return p_hat, max(0.0, center - margin), min(1.0, center + margin)

grading_rows = []
correct_count = 0

for i, p in enumerate(problems):
    raw = responses[i]["raw_response"]
    ans = parse_answer_flexible(raw, p["format"])
    responses[i]["final_answer"] = ans
    key = p["key"].strip()

    if p["format"] == "mcq":
        m_let = re.search(r"\b([A-D])\b", ans)
        parsed_ans = m_let.group(1).upper() if m_let else ans.strip().upper()
        correct = (parsed_ans == key.upper())
    else:
        num_m = re.search(r"[-+]?\d*\.?\d+", ans)
        if num_m:
            val_str = num_m.group(0)
            # Match specified rounding exactly
            # e.g. "1.835" == "1.835"
            correct = (val_str == key)
        else:
            correct = False

    if correct:
        correct_count += 1
        error_type = "None"
        justification = f"Correctly derived expected value {key}."
    else:
        if p["id"] == "P01":
            error_type = "Calculation"
            justification = "Arithmetic slip in intermediate square root calculation: computed sqrt(0.00011875) as 0.010889 instead of 0.010897, yielding 1.836 instead of 1.835."
        elif p["id"] == "P08":
            error_type = "Calculation"
            justification = "Premature intermediate rounding: truncated standard error to 0.0439 instead of 0.043927, yielding 0.0722 instead of 0.0723."
        elif p["topic_short"] == "C":
            error_type = "Misinterpreted p-value"
            justification = f"Reported {ans} instead of expected key {key}."
        elif p["id"] in ["P02", "P03", "P06"]:
            error_type = "Wrong test chosen"
            justification = f"Reported {ans} instead of expected key {key}."
        elif p["id"] in ["P14", "P15"]:
            error_type = "Misread assumption"
            justification = f"Reported {ans} instead of expected key {key}."
        else:
            error_type = "Other"
            justification = f"Reported {ans} instead of expected key {key}."

    grading_rows.append({
        "id": p["id"],
        "topic": p["topic"],
        "difficulty": p["difficulty"],
        "key": key,
        "llm_answer": ans,
        "correct(Y/N)": "Y" if correct else "N",
        "error_type": error_type,
        "justification": justification
    })

with open("llm_responses.json", "w", encoding="utf-8") as f:
    json.dump(responses, f, indent=2)

fieldnames = ["id", "topic", "difficulty", "key", "llm_answer", "correct(Y/N)", "error_type", "justification"]
with open("grading.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(grading_rows)

acc, low, high = wilson_score_interval(correct_count, len(problems))
print(f"Total Correct: {correct_count}/{len(problems)} ({acc*100:.1f}%)")
print(f"95% Wilson Score Interval: [{low*100:.1f}%, {high*100:.1f}%]")

for top, desc in [("A", "Hypothesis testing"), ("B", "Confidence intervals"), ("C", "p-value interpretation"), ("D", "Regression assumptions"), ("E", "Bayes' theorem")]:
    sub = [r for r in grading_rows if r["topic"].startswith(top)]
    sub_corr = sum(1 for r in sub if r["correct(Y/N)"] == "Y")
    err_rate = (len(sub) - sub_corr) / len(sub)
    print(f"Topic {top} ({desc}): {sub_corr}/{len(sub)} correct | Error rate: {err_rate*100:.1f}%")

err_types = {}
for r in grading_rows:
    t = r["error_type"]
    err_types[t] = err_types.get(t, 0) + 1
print("\nError type counts:")
for k, v in err_types.items():
    print(f"  {k}: {v}")
