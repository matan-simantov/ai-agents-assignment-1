# Gradio chat UI. The chat shows only user messages and final answers;
# drafts, critic decisions and feedback go to the separate "Review log" tab + logs/app_log.json
import json

import gradio as gr

from context import ACTOR_MODEL, CRITIC_MODEL, LOGS_DIR, NAME
from loop import APPROVED, answer_question

APP_LOG = LOGS_DIR / "app_log.json"


def append_to_app_log(entry):
    LOGS_DIR.mkdir(exist_ok=True)
    entries = json.loads(APP_LOG.read_text(encoding="utf-8")) if APP_LOG.exists() else []
    entries.append(entry)
    APP_LOG.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def respond(message, display, history, session_log):
    if not message.strip():
        return "", display, display, history, session_log, log_to_markdown(session_log), session_log
    question_id = f"chat-{len(session_log) + 1}"
    answer, log = answer_question(message, history, scenario_id=question_id)

    display = display + [{"role": "user", "content": message},
                         {"role": "assistant", "content": answer}]
    # only approved answers go into the history that the model sees next time
    if log["final_status"] == APPROVED:
        history = history + [{"role": "user", "content": message},
                             {"role": "assistant", "content": answer}]

    session_log = session_log + [log]
    append_to_app_log(log)
    return "", display, display, history, session_log, log_to_markdown(session_log), session_log


def log_to_markdown(session_log):
    """Readable version of the review log for the second tab."""
    if not session_log:
        return "No questions yet."
    lines = []
    for entry in session_log:
        lines.append(f"### {entry['scenario_id']}: {entry['question']}")
        lines.append(f"**Final status: `{entry['final_status']}`**\n")
        for a in entry["attempts"]:
            if a.get("evaluation") != "completed":
                decision = "evaluation NOT completed (technical error)"
            else:
                decision = "APPROVED" if a["is_acceptable"] else "REJECTED"
            lines.append(f"- **Attempt {a['attempt']}** ({a.get('source', '')}) - **{decision}**")
            candidate = (a.get("candidate") or "").replace("\n", " ")   # keep it inside the list item
            lines.append(f"    - Candidate: {candidate}")
            lines.append(f"    - Feedback: {a.get('feedback') or a.get('error', '')}")
        lines.append("")
    return "\n".join(lines)


CSS = """
.rtl-chat, .rtl-chat * { direction: rtl; text-align: right; }
"""

with gr.Blocks(title=f"Chat with {NAME}") as demo:
    gr.Markdown(f"## Chat with {NAME} (sample fictional profile)\n"
                f"You can ask in Hebrew or English. Actor: `{ACTOR_MODEL}`, Critic: `{CRITIC_MODEL}`, "
                "max 3 attempts per question.")
    display_state = gr.State([])   # what the user sees
    history_state = gr.State([])   # approved conversation, sent to the actor
    log_state = gr.State([])

    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(label="Conversation", height=760, rtl=True, elem_classes="rtl-chat")
        msg = gr.Textbox(label="Your question", placeholder="למשל: איפה למדת? / Where did you study?",
                         rtl=True)
        send = gr.Button("Send", variant="primary")
    with gr.Tab("Review log"):
        gr.Markdown("Every attempt of every question: candidate answer, critic decision and feedback.")
        log_md = gr.Markdown("No questions yet.")
        with gr.Accordion("Raw JSON (also saved to logs/app_log.json)", open=False):
            log_view = gr.JSON(label="Session log")

    inputs = [msg, display_state, history_state, log_state]
    outputs = [msg, chatbot, display_state, history_state, log_state, log_md, log_view]
    msg.submit(respond, inputs, outputs)
    send.click(respond, inputs, outputs)


if __name__ == "__main__":
    demo.launch(css=CSS)
