# Small wrapper around the Gemini API (OpenAI-compatible endpoint) with bounded retries on 429 / 503.
import os
import time

from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from context import GEMINI_BASE_URL

MAX_RETRIES = 4          # retries after the first try, only for 429 / 503 / timeout
BASE_WAIT_SECONDS = 5    # 5, 10, 20, 40 seconds
TIMEOUT_SECONDS = 120    # one call should never hang forever

# max_retries=0 so the SDK does not retry by itself - we control the retries here
client = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"), base_url=GEMINI_BASE_URL,
                max_retries=0, timeout=TIMEOUT_SECONDS)


class LLMCallError(Exception):
    """The model call failed (after the allowed retries). Treated as a technical error."""


def call_llm(model, messages, **kwargs):
    """Returns the text of the model answer. Raises LLMCallError if it fails."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(model=model, messages=messages, **kwargs)
            text = response.choices[0].message.content
            if not text:
                raise LLMCallError("model returned an empty response")
            return text
        except (RateLimitError, APIStatusError, APITimeoutError) as e:
            status = getattr(e, "status_code", None) or "timeout"
            retryable = isinstance(e, (RateLimitError, APITimeoutError)) or status == 503
            if retryable and attempt < MAX_RETRIES:
                wait = BASE_WAIT_SECONDS * (2 ** attempt)
                print(f"[llm] got {status}, waiting {wait}s and retrying ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            if retryable:
                raise LLMCallError(f"rate limit / unavailable ({status}), gave up after {MAX_RETRIES} retries") from e
            raise LLMCallError(f"API error {status}: {e}") from e
        except APIConnectionError as e:
            raise LLMCallError(f"connection error: {e}") from e
