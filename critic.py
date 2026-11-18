# Critic: evaluates a candidate answer and returns a validated Evaluation object.
from pydantic import BaseModel, ValidationError

from context import CRITIC_MODEL
from llm import call_llm
from prompts import EVALUATOR_SYSTEM_PROMPT, evaluator_user_message


class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str


class EvaluationError(Exception):
    """The critic output could not be parsed / validated. This is NOT an approval."""


def parse_evaluation(raw_text):
    """Validates the raw model output against the Evaluation schema."""
    text = raw_text.strip()
    # sometimes models wrap JSON in ```json ... ``` even when asked not to
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return Evaluation.model_validate_json(text, strict=True)
    except ValidationError as e:
        raise EvaluationError(f"invalid evaluator output: {e.errors()[0]['msg']} | raw: {raw_text[:200]}") from e


def evaluate(question, answer, history):
    messages = [
        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
        {"role": "user", "content": evaluator_user_message(question, answer, history)},
    ]
    # ask Gemini for JSON that follows the Evaluation schema (structured output)
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "Evaluation", "schema": Evaluation.model_json_schema()},
    }
    raw = call_llm(CRITIC_MODEL, messages, response_format=response_format)
    return parse_evaluation(raw)


def mock_critic_always_reject(question, answer, history):
    """MOCK critic for scenario 5 - control flow test only, no LLM is called."""
    return Evaluation(
        is_acceptable=False,
        feedback="[MOCK] Rejected on purpose to test the attempt limit (control flow test).",
    )
