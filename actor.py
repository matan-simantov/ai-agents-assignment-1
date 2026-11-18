# Actor: generates a candidate answer (first attempt, or a fix after the critic rejected it).
from context import ACTOR_MODEL
from llm import call_llm
from prompts import ACTOR_SYSTEM_PROMPT, actor_retry_message


def generate_answer(question, history, rejected_answer=None, feedback=None):
    """history = the shown conversation so far (user messages + approved answers only)."""
    messages = [{"role": "system", "content": ACTOR_SYSTEM_PROMPT}]
    messages += history

    if rejected_answer is None:
        messages.append({"role": "user", "content": question})
    else:
        # retry: the actor gets the rejected answer + the critic feedback
        messages.append({"role": "user", "content": actor_retry_message(question, rejected_answer, feedback)})

    return call_llm(ACTOR_MODEL, messages)
