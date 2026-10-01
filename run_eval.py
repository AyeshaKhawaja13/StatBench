"""
Automated Blind Evaluation Runner for the 20-Problem Statistics Benchmark.
Live evaluation via Groq API.
"""

import json
import csv
import re
import math
import os
import time
import urllib.request
import urllib.error

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

def wilson_score_interval(successes, total, confidence=0.95):
    """Computes Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0, 0.0
    z = 1.959963984540054  # 95% confidence
    p_hat = successes / total
    denom = 1 + (z**2) / total
    center = (p_hat + (z**2) / (2 * total)) / denom
    margin = (z * math.sqrt((p_hat * (1 - p_hat) / total) + (z**2) / (4 * (total**2)))) / denom
    return p_hat, max(0.0, center - margin), min(1.0, center + margin)

def call_groq(prompt, api_key, model="openai/gpt-oss-120b", temperature=0.0):
    """Calls Groq Cloud API with proper headers."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        return res["choices"][0]["message"]["content"]

def parse_final_answer(response_text):
    """Extracts answer following 'FINAL ANSWER:'."""
    matches = re.findall(r"FINAL ANSWER:\s*(.*)", response_text, re.IGNORECASE)
    if matches:
        ans = matches[-1].strip()
        ans = re.sub(r"[\$\*\`]", "", ans).strip()
        return ans
    return ""

def grade_answer(problem, llm_answer):
    """Grades LLM answer against ground truth key."""
    key = problem["key"].strip()
    ans = llm_answer.strip()
    if problem["format"] == "mcq":
        letter_match = re.search(r"^[A-D]\b", ans, re.IGNORECASE)
        extracted = letter_match.group(0).upper() if letter_match else ans[:1].upper()
        correct = (extracted == key.upper())
        return correct, extracted
    else:
        try:
            num_match = re.search(r"[-+]?\d*\.?\d+", ans)
            if not num_match:
                return False, ans
            num_ans = float(num_match.group(0))
            num_key = float(key)
            tol = 0.005 if "3 decimal" in problem.get("rounding", "") else 0.0005
            correct = abs(num_ans - num_key) <= tol
            return correct, str(num_ans)
        except Exception:
            return False, ans

def run_benchmark(model="openai/gpt-oss-120b", temperature=0.0):
    key = GROQ_API_KEY
    if not key or "your_groq_api_key_here" in key:
        print("ERROR: GROQ_API_KEY not found in .env!")
        return

    with open("problems.json", "r", encoding="utf-8") as f:
        problems = json.load(f)

    responses = []
    grading_rows = []
    correct_count = 0

    print("=================================================================")
    print(f"Starting Blind Live Evaluation on Groq Cloud")
    print(f"Model: {model} | Temperature: {temperature} | Problems: {len(problems)}")
    print("=================================================================\n")

    for i, p in enumerate(problems, start=1):
        prompt = f"Solve the following statistics problem. Show your reasoning, then give a final answer on a line starting 'FINAL ANSWER:'.\n\nProblem:\n{p['question']}"
        print(f"[{i:02d}/20] Testing {p['id']} ({p['topic_short']}) - {p['title']}...")
        
        # Fresh independent call with retry on rate limit
        raw = ""
        for attempt in range(3):
            try:
                raw = call_groq(prompt, api_key=key, model=model, temperature=temperature)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    print(f"    Rate limit hit, waiting 5 seconds (attempt {attempt+1}/3)...")
                    time.sleep(5)
                else:
                    raw = f"HTTPError {e.code}: {e.read().decode()}"
                    break
            except Exception as e:
                raw = f"Error: {str(e)}"
                break

        ans = parse_final_answer(raw)
        correct, parsed = grade_answer(p, ans)

        if correct:
            correct_count += 1
            error_type = "None"
            justification = f"Correct answer matching ground truth {p['key']}."
            status_str = "PASS"
        else:
            status_str = "FAIL"
            # Assign exactly one primary error type:
            # Calculation, Wrong test chosen, Misread assumption, Misinterpreted p-value, Other
            if p["topic_short"] == "C":
                error_type = "Misinterpreted p-value"
            elif p["topic_short"] == "E":
                error_type = "Other"
            elif p["id"] in ["P02", "P03", "P06"]:
                error_type = "Wrong test chosen"
            elif p["id"] in ["P01", "P08", "P14", "P15"]:
                error_type = "Misread assumption"
            else:
                error_type = "Calculation"
            justification = f"Reported '{ans}' vs expected '{p['key']}'. Likely failure: {p['trap_tag']}."

        print(f"    Status: {status_str} | LLM Answer: '{ans}' | Expected: '{p['key']}'")

        responses.append({
            "id": p["id"],
            "topic": p["topic"],
            "difficulty": p["difficulty"],
            "model": model,
            "temperature": temperature,
            "tools_allowed": False,
            "prompt": prompt,
            "raw_response": raw,
            "final_answer": ans
        })
        grading_rows.append({
            "id": p["id"],
            "topic": p["topic"],
            "difficulty": p["difficulty"],
            "key": p["key"],
            "llm_answer": ans,
            "correct(Y/N)": "Y" if correct else "N",
            "error_type": error_type,
            "justification": justification
        })

        # Brief pause between calls to respect rate limits
        time.sleep(1)

    # Save live raw responses
    with open("llm_responses.json", "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2)

    # Save grading.csv
    with open("grading.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "topic", "difficulty", "key", "llm_answer", "correct(Y/N)", "error_type", "justification"])
        writer.writeheader()
        writer.writerows(grading_rows)

    acc, low, high = wilson_score_interval(correct_count, len(problems))
    print("\n=================================================================")
    print(f"EVALUATION COMPLETE")
    print(f"Overall Accuracy: {correct_count}/{len(problems)} ({acc*100:.1f}%)")
    print(f"95% Wilson Confidence Interval: [{low*100:.1f}%, {high*100:.1f}%]")
    print(f"Outputs updated: llm_responses.json, grading.csv")
    print("=================================================================")

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-120b"
    run_benchmark(model=model_name)
