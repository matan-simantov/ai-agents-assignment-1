# The Actor -> Critic -> fix loop (evaluator-optimizer pattern), with a run log.
import json
from datetime import datetime

from actor import generate_answer
from context import ACTOR_MODEL, CRITIC_MODEL, LOGS_DIR, MAX_ATTEMPTS
from critic import EvaluationError, evaluate
from llm import LLMCallError

# final statuses
APPROVED = "approved"
MAX_ATTEMPTS_REACHED = "max_attempts_reached"
TECHNICAL_ERROR = "technical_error"

MSG_MAX_ATTEMPTS = ("Sorry, I could not produce an approved answer within the limit of "
                    f"{MAX_ATTEMPTS} attempts. Please try to rephrase the question.")
# technical error messages (Hebrew, shown in the chat). The exact reason is written to the log.
MSG_TECH_ERROR = {
    "daily_quota": "שגיאה טכנית: מכסת השימוש היומית החינמית של מודל השפה נוצלה, ולכן אין אפשרות לענות כרגע. "
                   "המכסה מתאפסת פעם ביום (אפשר להגדיר מודל אחר ב-ACTOR_MODEL / CRITIC_MODEL).",
    "rate_limit": "שגיאה טכנית: נשלחו יותר מדי בקשות למודל השפה בזמן קצר. נסה שוב בעוד דקה.",
    "unavailable": "שגיאה טכנית: מודל השפה עמוס כרגע ולא הגיב. נסה שוב בעוד כמה רגעים.",
    "timeout": "שגיאה טכנית: מודל השפה לא הגיב בזמן. נסה שוב בעוד כמה רגעים.",
    "invalid_evaluation": "שגיאה טכנית: התשובה נוצרה אבל הבדיקה שלה לא הושלמה (פלט המעריך לא היה תקין), "
                          "ולכן היא לא מוצגת. נסה לשאול שוב.",
}
MSG_TECH_ERROR_DEFAULT = "שגיאה טכנית בפנייה למודל השפה, והשאלה לא טופלה. נסה שוב מאוחר יותר."


def now():
    return datetime.now().isoformat(timespec="seconds")


def answer_question(question, history, scenario_id="chat", actor=generate_answer, critic=evaluate,
                    first_draft=None, mode="llm"):
    """
    Runs the loop for one user question.
    history     - the conversation shown so far (it is NOT changed here)
    actor/critic - functions, so a mock critic can be used for the control flow test
    first_draft - optional test draft that is used as attempt 1 instead of calling the actor
    Returns (message_to_show, log_entry). log_entry["final_status"] tells what happened.
    """
    log = {
        "scenario_id": scenario_id,
        "mode": mode,
        "question": question,
        "started_at": now(),
        "actor_model": ACTOR_MODEL,
        "critic_model": CRITIC_MODEL if critic is evaluate else critic.__name__,
        "attempts": [],
        "final_status": None,
        "shown_to_user": None,
    }

    rejected_answer, feedback = None, None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        record = {"attempt": attempt}
        log["attempts"].append(record)

        # 1. generate (or use the injected test draft on attempt 1)
        try:
            if attempt == 1 and first_draft is not None:
                candidate = first_draft
                record["source"] = "injected_test_draft"
            else:
                candidate = actor(question, history, rejected_answer, feedback)
                record["source"] = "actor"
        except LLMCallError as e:
            record["candidate"] = None
            record["evaluation"] = "not_completed"
            record["error"] = f"actor call failed: {e}"
            record["error_type"] = e.kind
            return finish(log, TECHNICAL_ERROR, MSG_TECH_ERROR.get(e.kind, MSG_TECH_ERROR_DEFAULT))
        record["candidate"] = candidate

        # 2. evaluate + validate with Pydantic
        try:
            evaluation = critic(question, candidate, history)
        except (LLMCallError, EvaluationError) as e:
            kind = e.kind if isinstance(e, LLMCallError) else "invalid_evaluation"
            record["evaluation"] = "not_completed"   # no content decision was received
            record["error"] = f"critic failed: {e}"
            record["error_type"] = kind
            return finish(log, TECHNICAL_ERROR, MSG_TECH_ERROR.get(kind, MSG_TECH_ERROR_DEFAULT))

        record["evaluation"] = "completed"
        record["is_acceptable"] = evaluation.is_acceptable
        record["feedback"] = evaluation.feedback

        # 3. approved -> done
        if evaluation.is_acceptable:
            return finish(log, APPROVED, candidate)

        # 4. rejected -> next attempt gets the rejected answer + feedback
        rejected_answer, feedback = candidate, evaluation.feedback

    # 5. all attempts rejected
    return finish(log, MAX_ATTEMPTS_REACHED, MSG_MAX_ATTEMPTS)


def finish(log, status, message):
    log["final_status"] = status
    log["shown_to_user"] = message
    log["finished_at"] = now()
    return message, log


def save_log(entries, filename):
    LOGS_DIR.mkdir(exist_ok=True)
    path = LOGS_DIR / filename
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
