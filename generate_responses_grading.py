import json
import csv

with open("problems.json", "r", encoding="utf-8") as f:
    problems = json.load(f)

# Save llm_responses.json documenting the blind prompt harness and the Phase 2 stop
llm_responses_data = {
    "status": "STOPPED_AT_PHASE_2",
    "reason": "GLOBAL RULE ENFORCEMENT: Never invent results. If you cannot actually call a separate LLM, stop at Phase 2 and say so. Do not simulate the LLM's answers.",
    "environment_audit": {
        "external_network_access": False,
        "ollama_installed": True,
        "ollama_models_available": [],
        "api_keys_present": False,
        "local_weights_found": False
    },
    "protocol": {
        "model_name": None,
        "model_version": None,
        "temperature": 0.0,
        "tools_allowed": False,
        "prompt_template": "Solve the following statistics problem. Show your reasoning, then give a final answer on a line starting 'FINAL ANSWER:'."
    },
    "prompts_prepared": [
        {
            "id": p["id"],
            "topic": p["topic"],
            "difficulty": p["difficulty"],
            "prompt": f"Solve the following statistics problem. Show your reasoning, then give a final answer on a line starting 'FINAL ANSWER:'.\n\nProblem:\n{p['question']}",
            "raw_response": None,
            "final_answer": None
        }
        for p in problems
    ]
}

with open("llm_responses.json", "w", encoding="utf-8") as f:
    json.dump(llm_responses_data, f, indent=2)
print("Saved llm_responses.json successfully.")

# Save grading.csv
# Columns: id, topic, difficulty, key, llm_answer, correct(Y/N), error_type, justification
fieldnames = ["id", "topic", "difficulty", "key", "llm_answer", "correct(Y/N)", "error_type", "justification"]
with open("grading.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for p in problems:
        writer.writerow({
            "id": p["id"],
            "topic": p["topic"],
            "difficulty": p["difficulty"],
            "key": p["key"],
            "llm_answer": "N/A (Stopped at Phase 2)",
            "correct(Y/N)": "N/A",
            "error_type": "None",
            "justification": "Execution stopped at Phase 2 per protocol: external/local LLM unavailable in air-gapped environment; simulation forbidden."
        })
print("Saved grading.csv successfully.")
