import os
import urllib.request
import json
import time
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GROQ_API_KEY")

with open("problems.json", "r", encoding="utf-8") as f:
    problems = json.load(f)

p19 = problems[18]
prompt = f"Solve the following statistics problem. Show your reasoning, then give a final answer on a line starting 'FINAL ANSWER:'.\n\nProblem:\n{p19['question']}"

headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0"
}
payload = {
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.0
}

print("Requesting P19...")
for attempt in range(5):
    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw = data["choices"][0]["message"]["content"]
            print("Successfully received P19:")
            print(raw)
            with open("p19_raw.txt", "w", encoding="utf-8") as out:
                out.write(raw)
            break
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {e}. Sleeping 10s...")
        time.sleep(10)
