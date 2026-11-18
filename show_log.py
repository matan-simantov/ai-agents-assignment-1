# Prints a log file in a readable way: python show_log.py logs/scenarios_llm.json [scenario_id]
import json
import sys

entries = json.load(open(sys.argv[1], encoding="utf-8"))
only = sys.argv[2] if len(sys.argv) > 2 else None

for e in entries:
    if only and not e["scenario_id"].startswith(only):
        continue
    print("=" * 90)
    print(f"{e['scenario_id']}  mode={e['mode']}  actor={e['actor_model']}  critic={e['critic_model']}")
    print(f"QUESTION: {e['question']}")
    for a in e["attempts"]:
        if a.get("evaluation") != "completed":
            decision = "EVALUATION NOT COMPLETED (technical error)"
        else:
            decision = "APPROVED" if a["is_acceptable"] else "REJECTED"
        print(f"\n  --- attempt {a['attempt']} ({a.get('source')}) -> {decision}")
        print(f"  candidate: {a.get('candidate')}")
        print(f"  feedback:  {a.get('feedback') or a.get('error', '')}")
    print(f"\nFINAL STATUS: {e['final_status']}")
    print(f"SHOWN TO USER: {e['shown_to_user']}")
    if "matches_expectation" in e:
        print(f"EXPECTED: {e['expected']}  ->  matches: {e['matches_expectation']}")
