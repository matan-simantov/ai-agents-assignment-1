# Offline tests of the control flow (MOCK actor/critic, no API calls).
# Run: pytest -q
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from critic import Evaluation, EvaluationError, mock_critic_always_reject, parse_evaluation
from llm import LLMCallError
from loop import APPROVED, MAX_ATTEMPTS_REACHED, TECHNICAL_ERROR, answer_question


def fake_actor(question, history, rejected_answer=None, feedback=None):
    if rejected_answer is None:
        return "first answer"
    return f"fixed answer (feedback was: {feedback})"


def test_approved_on_first_attempt():
    critic = lambda q, a, h: Evaluation(is_acceptable=True, feedback="ok")
    msg, log = answer_question("q", [], actor=fake_actor, critic=critic, mode="mock")
    assert log["final_status"] == APPROVED
    assert msg == "first answer"
    assert len(log["attempts"]) == 1


def test_reject_then_fix_passes_feedback_to_actor():
    calls = []

    def critic(q, a, h):
        calls.append(a)
        return Evaluation(is_acceptable=len(calls) == 2, feedback="remove the invented job")

    msg, log = answer_question("q", [], actor=fake_actor, critic=critic, mode="mock")
    assert log["final_status"] == APPROVED
    assert "remove the invented job" in msg   # the actor really got the feedback
    assert len(log["attempts"]) == 2


def test_stops_after_three_rejections():
    msg, log = answer_question("q", [], actor=fake_actor, critic=mock_critic_always_reject, mode="mock")
    assert log["final_status"] == MAX_ATTEMPTS_REACHED
    assert len(log["attempts"]) == 3
    assert "could not produce an approved answer" in msg


def test_invalid_critic_output_is_technical_error_not_approval():
    def bad_critic(q, a, h):
        return parse_evaluation('{"feedback": "looks fine"}')   # is_acceptable is missing

    msg, log = answer_question("q", [], actor=fake_actor, critic=bad_critic, mode="mock")
    assert log["final_status"] == TECHNICAL_ERROR
    assert log["attempts"][0]["evaluation"] == "not_completed"
    assert "is_acceptable" not in log["attempts"][0]


def test_llm_failure_is_technical_error():
    def broken_actor(*args):
        raise LLMCallError("429 after retries")

    msg, log = answer_question("q", [], actor=broken_actor, critic=mock_critic_always_reject, mode="mock")
    assert log["final_status"] == TECHNICAL_ERROR


def test_parse_evaluation():
    ev = parse_evaluation('```json\n{"is_acceptable": false, "feedback": "x"}\n```')
    assert ev.is_acceptable is False
    with pytest.raises(EvaluationError):
        parse_evaluation("The answer is acceptable.")
    with pytest.raises(EvaluationError):
        parse_evaluation('{"is_acceptable": "yes", "feedback": "x"}')   # wrong type


def test_429_retries_are_bounded(monkeypatch):
    import httpx
    import llm
    from openai import RateLimitError

    tries = []

    def always_429(**kwargs):
        tries.append(1)
        resp = httpx.Response(429, request=httpx.Request("POST", "http://test"))
        raise RateLimitError("rate limited", response=resp, body=None)

    monkeypatch.setattr(llm.client.chat.completions, "create", always_429)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    with pytest.raises(LLMCallError):
        llm.call_llm("some-model", [{"role": "user", "content": "hi"}])
    assert len(tries) == llm.MAX_RETRIES + 1
