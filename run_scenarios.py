# Runs the 5 required scenarios and saves the logs.
#   Scenarios 1-4: REAL LLM (Gemini) for actor and critic -> logs/scenarios_llm.json
#   Scenario 5:    real actor + MOCK critic (always rejects) -> logs/scenario5_mock.json
# Usage: python run_scenarios.py          (all 5 scenarios)
#        python run_scenarios.py mock     (only scenario 5, the mock control flow test)
import sys
import time

from critic import mock_critic_always_reject
from loop import APPROVED, MAX_ATTEMPTS_REACHED, answer_question, save_log

PAUSE = 4  # seconds between questions, to stay under the free tier rate limit

# Scenario 4: a broken test draft with invented facts (Google, team lead, 5 years, MIT)
BAD_DRAFT = ("עבדתי 5 שנים בגוגל כראש צוות של 10 מפתחים, ולפני כן סיימתי תואר שני ב-MIT. "
             "היום אני Backend Developer ב-Lumio Data.")


def run_conversation(scenario_id, questions, **kwargs):
    """Runs one or more questions in the same conversation (history is kept like in the app)."""
    history, entries = [], []
    for i, q in enumerate(questions, start=1):
        answer, log = answer_question(q, history, scenario_id=f"{scenario_id}.{i}", **kwargs)
        print(f"\n[{log['scenario_id']}] Q: {q}\n  status: {log['final_status']} "
              f"after {len(log['attempts'])} attempt(s)\n  shown: {answer}")
        if log["final_status"] == APPROVED:
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]
        entries.append(log)
        time.sleep(PAUSE)
    return entries


def check(name, ok):
    print(f"  ==> {name}: {'MATCHES expectation' if ok else 'DOES NOT match expectation'}")
    return ok


def main():
    if "mock" not in sys.argv:
        run_llm_scenarios()
    run_mock_scenario()


def run_llm_scenarios():
    llm_logs = []

    print("=== Scenario 1: question about existing info (LLM) ===")
    s1 = run_conversation("S1", ["איפה למדת ומה היה פרויקט הגמר שלך?"])
    s1[0]["expected"] = "approved answer: University of Haifa, B.Sc. CS, BusNow project"
    s1[0]["matches_expectation"] = check("S1", s1[0]["final_status"] == APPROVED)
    llm_logs += s1

    print("\n=== Scenario 2: follow-up question (LLM) ===")
    s2 = run_conversation("S2", ["באיזו חברה אתה עובד היום ומה התפקיד שלך?",
                                 "ומה ההישג שאתה הכי גאה בו שם?"])
    s2[1]["expected"] = "second answer understands 'there' = Lumio Data (nightly report 40 -> 12 min)"
    s2[1]["matches_expectation"] = check("S2", all(e["final_status"] == APPROVED for e in s2))
    llm_logs += s2

    print("\n=== Scenario 3: detail that is not in the sources (LLM) ===")
    s3 = run_conversation("S3", ["מה היה הממוצע שלך בתואר?"])
    s3[0]["expected"] = "approved answer that says the GPA is not in the profile, no invented number"
    s3[0]["matches_expectation"] = check("S3", s3[0]["final_status"] == APPROVED)
    llm_logs += s3

    print("\n=== Scenario 4: rejection and improvement (injected bad draft, then LLM) ===")
    s4 = run_conversation("S4", ["ספר לי על הניסיון המקצועי שלך"], first_draft=BAD_DRAFT)
    first = s4[0]["attempts"][0]
    s4[0]["expected"] = "attempt 1 (injected draft) rejected with specific feedback, a later attempt approved"
    s4[0]["matches_expectation"] = check(
        "S4", first.get("is_acceptable") is False and s4[0]["final_status"] == APPROVED)
    llm_logs += s4

    path = save_log(llm_logs, "scenarios_llm.json")
    print(f"\nsaved {path}")


def run_mock_scenario():
    print("\n=== Scenario 5: attempt limit - MOCK critic (control flow test) ===")
    s5 = run_conversation("S5", ["אילו מיומנויות טכניות יש לך?"],
                          critic=mock_critic_always_reject, mode="mock_critic")
    s5[0]["expected"] = "3 rejected attempts, stop after the third, limit message shown"
    s5[0]["matches_expectation"] = check(
        "S5", s5[0]["final_status"] == MAX_ATTEMPTS_REACHED and len(s5[0]["attempts"]) == 3)
    path = save_log(s5, "scenario5_mock.json")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
