# Small wrapper around the Gemini API (OpenAI-compatible endpoint) with bounded retries.
# - 503 / timeout / per-minute 429: retry a few times with growing waits
# - DAILY quota 429: no retry (it will not come back today), and the model is not called again
import os
import time

from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from context import GEMINI_BASE_URL

MAX_RETRIES = 4          # retries after the first try, only for 503 / timeout / per-minute 429
BASE_WAIT_SECONDS = 5    # 5, 10, 20, 40 seconds
TIMEOUT_SECONDS = 120    # one call should never hang forever

# max_retries=0 so the SDK does not retry by itself - we control the retries here
client = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"), base_url=GEMINI_BASE_URL,
                max_retries=0, timeout=TIMEOUT_SECONDS)

# models whose free daily quota is used up (until the app is restarted)
DAILY_QUOTA_EXHAUSTED = set()


class LLMCallError(Exception):
    """The model call failed. Treated as a technical error, never as an approval.
    kind: daily_quota / rate_limit / unavailable / timeout / connection / api_error / empty"""

    def __init__(self, message, kind="api_error"):
        super().__init__(message)
        self.kind = kind


def is_daily_quota(error):
    """Gemini puts the quota id in the 429 body, e.g. GenerateRequestsPerDayPerProjectPerModel-FreeTier."""
    body = getattr(getattr(error, "response", None), "text", "") or ""
    return "PerDay" in body


def call_llm(model, messages, **kwargs):
    """Returns the text of the model answer. Raises LLMCallError if it fails."""
    if model in DAILY_QUOTA_EXHAUSTED:
        raise LLMCallError(f"daily free quota of {model} already used up - not calling it again", "daily_quota")

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(model=model, messages=messages, **kwargs)
            text = response.choices[0].message.content
            if not text:
                raise LLMCallError("model returned an empty response", "empty")
            return text
        except RateLimitError as e:
            if is_daily_quota(e):
                DAILY_QUOTA_EXHAUSTED.add(model)
                raise LLMCallError(f"429 daily free quota of {model} used up "
                                   "(GenerateRequestsPerDay...-FreeTier), no retry", "daily_quota") from e
            if attempt < MAX_RETRIES:
                wait_and_log(429, attempt)
                continue
            raise LLMCallError(f"429 rate limit, gave up after {MAX_RETRIES} retries", "rate_limit") from e
        except (APIStatusError, APITimeoutError) as e:
            status = getattr(e, "status_code", None) or "timeout"
            if status in (503, "timeout") and attempt < MAX_RETRIES:
                wait_and_log(status, attempt)
                continue
            if status == "timeout":
                raise LLMCallError(f"timeout after {MAX_RETRIES} retries", "timeout") from e
            if status == 503:
                raise LLMCallError(f"503 model overloaded, gave up after {MAX_RETRIES} retries", "unavailable") from e
            raise LLMCallError(f"API error {status}: {e}", "api_error") from e
        except APIConnectionError as e:
            raise LLMCallError(f"connection error: {e}", "connection") from e


def wait_and_log(status, attempt):
    wait = BASE_WAIT_SECONDS * (2 ** attempt)
    print(f"[llm] got {status}, waiting {wait}s and retrying ({attempt + 1}/{MAX_RETRIES})")
    time.sleep(wait)
