# Assignment 1 – Personal chat with an answer evaluator and an improvement loop

AI Systems course (Dr. Paul Avrahami), lecture 2, slides 19–21.

**Student:** Matan Siman Tov, ID 206356172

A Gradio chat that answers as a person, based on a LinkedIn profile and a personal summary.
Every answer is written by an **Actor** (LLM), checked by a **Critic** (separate LLM call with its own
prompt) that returns structured output validated with **Pydantic**, and if it is rejected the Actor
gets the rejected answer + the feedback and tries again (max 3 attempts per question).

> The profile in `me/` is a **sample, fictional profile** ("Daniel Mor") created for this assignment.

## Models

| What | Actor | Critic |
|------|-------|--------|
| Scenarios 1–4 (real LLM, `logs/scenarios_llm.json`) | `gemini-3.5-flash` | `gemini-3.5-flash` |
| Scenario 5 (`logs/scenario5_mock.json`) | `gemini-3-flash-preview` | **MOCK** – always rejects |
| Live chat screenshots (first entries of `logs/app_log.json`) | `gemini-3-flash-preview` | `gemini-3-flash-preview` |
| **Live app – current default** | `gemini-3.5-flash-lite` | `gemini-3.5-flash-lite` |

All through the Google Gemini API, OpenAI-compatible endpoint, free tier. Default temperature, separate
system prompt and separate call for each role. The Critic uses structured output (`response_format` =
JSON schema of `Evaluation`) and the result is validated again with Pydantic (strict).
Client timeout 120 s. Retries only on 503 / timeout / per-minute 429: 4 retries with 5, 10, 20, 40 s waits.
A **daily** quota 429 (`...PerDay...-FreeTier`) is not retried, and that model is not called again until the
app is restarted, so no free quota is wasted on requests that cannot succeed.

The default model is **`gemini-3.5-flash-lite`** (both roles); change it in `.env` with `ACTOR_MODEL` / `CRITIC_MODEL`.
Why more than one model:
- `gemini-2.5-flash` (the model from the lecture) returns 404 "no longer available to new users".
- `gemini-3.8-flash` was overloaded (503 / 429) in my first run → `logs/run1_gemini-3.8-flash_overloaded/`.
- After scenarios 1–4 the **daily** free-tier quota of `gemini-3.5-flash` ran out (429
  `GenerateRequestsPerDayPerProjectPerModel-FreeTier`) → `logs/run2_scenario5_daily_quota_exhausted/`.
  So scenario 5 and the screenshots were run with `gemini-3-flash-preview`.
- During live testing the daily quota of `gemini-3-flash-preview` ran out as well (the free tier allows only a
  small number of requests per day per model, and every question needs at least 2 calls). The app default was
  therefore changed to `gemini-3.5-flash-lite`: tested with Actor + Critic + Pydantic structured output, two
  questions in a row with a follow-up (~4 s per question), and it still rejects the injected bad draft of scenario 4.

**Free-tier limitation:** each model has its own small daily quota (it resets once a day). If it runs out, the
chat shows a Hebrew technical-error message and nothing is approved. To continue, set another available model
in `.env` (for example `gemini-3.1-flash-lite`) and restart the app. No billing is enabled.

These failed runs are kept on purpose: they show the technical-error exit working with real API errors
(the question ends with an error message, nothing is approved).

## Project structure

```
me/linkedin.txt, me/summary.txt   background sources (sample profile)
context.py      loads the background, model names, MAX_ATTEMPTS = 3
llm.py          Gemini client + bounded retries (503/timeout/per-minute 429), no retry on daily quota
prompts.py      Actor system prompt, retry message, System Evaluator, User Evaluator
actor.py        Actor - generates / fixes an answer
critic.py       Critic - Evaluation(BaseModel), evaluation + validation, mock critic for scenario 5
loop.py         generate -> evaluate -> fix loop + run log
app.py          Gradio UI (Chat tab + Review log tab)
run_scenarios.py  runs the 5 required scenarios and saves the logs
show_log.py     prints a log file in a readable way (attempts, decisions, feedback)
tests/test_loop.py  offline tests of the control flow (mock, no API calls)
docs/architecture.png (+ .mmd source)   architecture diagram
logs/           run logs (JSON) and console output
screenshots/    screenshots of the chat and the review log
report.docx     short report
```

## Setup

Python **3.11** (tested with 3.11 on macOS).

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create the environment file (the key is never committed – `.env` is in `.gitignore`):

```bash
cp .env.example .env
# then edit .env and set GOOGLE_API_KEY (free key from https://aistudio.google.com/apikey)
```

## Run the chat

```bash
python app.py
```

Open the URL printed in the terminal (usually http://127.0.0.1:7860).
You can ask in Hebrew or English. The **Chat** tab shows only the questions and the final answers.
The **Review log** tab shows all attempts (candidate, critic decision, feedback). Every question is also
appended to `logs/app_log.json`.

## Run the scenarios and tests

```bash
python run_scenarios.py     # 5 scenarios, real LLM (1-4) + mock critic (5). Takes a few minutes.
python run_scenarios.py mock   # only scenario 5
python show_log.py logs/scenarios_llm.json S4   # read a log (here scenario 4)
pytest -q                   # offline tests of the loop (no API key needed)
```

Outputs:
- `logs/scenarios_llm.json` – scenarios 1–4, **real LLM results**
- `logs/scenario5_mock.json` – scenario 5, **mock critic (control flow test)**, the actor is the real LLM
- `logs/run_scenarios_output.txt` – console output of the run
- `logs/app_log.json` – questions asked in the Gradio app (the screenshot session)

## Results summary

| # | Scenario | Result | Matches expectation |
|---|---|---|---|
| 1 | Question on existing info | approved on attempt 1, grounded in linkedin.txt (Education) | yes |
| 2 | Follow-up question | both approved, "there" understood as Lumio Data | yes |
| 3 | Detail not in sources (GPA) | approved answer saying the info is missing | yes |
| 4 | Injected bad draft (Google, MIT) | attempt 1 rejected with specific feedback, attempt 2 fixed and approved | yes |
| 5 | Mock critic (control flow test) | 3 rejections, stopped, limit message shown | yes |

Screenshots are in `screenshots/`, the short report is `report.docx`.
Screenshots 01–03 show the real chat session from `logs/app_log.json` (gemini-3-flash-preview). After a small
fix to the review-log formatting they were re-rendered from that saved log in the same UI, without new API calls.

## How the loop works

1. The Actor gets: system prompt (role + style + grounding rules + background), the conversation history,
   and the question.
2. The Critic gets the question, the candidate answer, the background and the recent history, and must
   return JSON `{"is_acceptable": bool, "feedback": str}`. The JSON is validated with
   `Evaluation.model_validate_json` (strict). The decision is taken only from the validated boolean.
3. Approved → shown to the user and added to the history.
4. Rejected and attempts left → the Actor gets the question, background, history **plus the rejected
   answer and the feedback**, and writes a fixed version, which is evaluated again.
5. Three rejections → the message "could not produce an approved answer within the limit of 3 attempts".
   The rejected drafts are only in the log.
6. Technical error (API error, daily quota, 429/503 after the retries, timeout, or invalid critic output) →
   a Hebrew error message that says what happened (e.g. daily quota used up / model overloaded), the
   evaluation is logged as `not_completed` with `error_type`, and the loop stops. A failed validation is
   **never** treated as an approval, and a failed answer is **not** added to the conversation history.

Final statuses in the log: `approved`, `max_attempts_reached`, `technical_error`.
