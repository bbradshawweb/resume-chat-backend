import json
import logging
import os
from typing import Any

import fitz  # PyMuPDF
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "1400"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "1400"))
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "25"))

DEFAULT_ALLOWED_ORIGINS = [
    "https://bradenbradshaw.com",
    "https://www.bradenbradshaw.com",
    "https://bbradshawweb.github.io",
    "http://localhost:8765",
    "http://127.0.0.1:8765",
]


app = Flask(__name__)
CORS(
    app,
    resources={
        r"/chat": {"origins": os.getenv("ALLOWED_ORIGINS", ",".join(DEFAULT_ALLOWED_ORIGINS)).split(",")},
        r"/chat/v2": {"origins": os.getenv("ALLOWED_ORIGINS", ",".join(DEFAULT_ALLOWED_ORIGINS)).split(",")},
        r"/context": {"origins": "*"},
        r"/health": {"origins": "*"},
        r"/": {"origins": "*"},
    },
)


def extract_resume_text(pdf_path: str = "resume.pdf") -> str:
    """Load resume text once at startup so each request can stay lightweight."""
    try:
        with fitz.open(pdf_path) as doc:
            return "\n".join(page.get_text().strip() for page in doc if page.get_text().strip())
    except Exception as exc:
        logger.exception("Error reading %s: %s", pdf_path, exc)
        return "Resume PDF text is currently unavailable."


def load_profile_context(path: str = "profile_context.json") -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.warning("%s not found; using resume PDF context only.", path)
    except json.JSONDecodeError as exc:
        logger.exception("Invalid JSON in %s: %s", path, exc)

    return {
        "version": "fallback",
        "positioning": "Data Scientist | Revenue Operations",
        "summary": "Braden Bradshaw works across data science, revenue operations, analytics, automation, and behavioral research.",
    }


resume_text = extract_resume_text()
profile_context = load_profile_context()


def get_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT_SECONDS)


def clean_text(value: Any, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())[:max_chars]


def normalize_history(raw_history: Any, current_message: str) -> list[dict[str, str]]:
    if not isinstance(raw_history, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in raw_history[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue

        content = clean_text(item.get("content"), MAX_HISTORY_CHARS)
        if content:
            normalized.append({"role": role, "content": content})

    # The current frontend adds the active user turn to history before sending.
    # Drop that duplicate so the final prompt contains the latest question once.
    if normalized and normalized[-1]["role"] == "user" and normalized[-1]["content"] == current_message:
        normalized.pop()

    return normalized


def summarize_page_context(raw_context: Any) -> str:
    if not isinstance(raw_context, dict):
        return ""

    safe_context = {
        str(key)[:40]: clean_text(value, 240)
        for key, value in raw_context.items()
        if isinstance(key, str) and isinstance(value, (str, int, float, bool))
    }
    if not safe_context:
        return ""

    return json.dumps(safe_context, ensure_ascii=True)


def build_system_prompt(page_context: str) -> str:
    context_json = json.dumps(profile_context, ensure_ascii=True, indent=2)

    prompt = f"""
You are Braden Bradshaw's interactive resume assistant embedded on bradenbradshaw.com.

Primary goal:
- Help visitors understand Braden's fit for data science, Revenue Operations, product analytics, marketing analytics, automation, and behavioral research roles.

Voice and behavior:
- Answer in first person as Braden when it feels natural.
- Be concise, specific, warm, and confident.
- Lead with Revenue Operations plus data science when relevant.
- Use concrete evidence from the provided context: metrics, projects, tools, roles, and research background.
- Do not invent employers, credentials, dates, technologies, outcomes, links, or contact details.
- If the answer is not in the context, say that you do not have that detail and pivot to the closest relevant known experience.
- For hiring or fit questions, connect evidence to business outcomes.
- Keep most answers to 2-4 short paragraphs unless the user asks for detail.

Priority context:
1. Structured portfolio and updated resume context below.
2. Extracted resume PDF text below.
3. Current page context, if provided.
4. Recent conversation messages.

Structured context:
{context_json}

Extracted resume PDF text:
{resume_text}
""".strip()

    if page_context:
        prompt += f"\n\nCurrent page context from the website:\n{page_context}"

    return prompt


def build_messages(user_message: str, history: list[dict[str, str]], page_context: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": build_system_prompt(page_context)}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


@app.route("/")
def index():
    return "Resume chat API is running."


@app.route("/context", methods=["GET"])
def context():
    return jsonify(
        {
            "version": profile_context.get("version", "unknown"),
            "positioning": profile_context.get("positioning"),
            "suggested_prompts": profile_context.get("suggested_prompts", []),
            "model": MODEL,
        }
    )


@app.route("/chat", methods=["POST"])
@app.route("/chat/v2", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_input = clean_text(payload.get("message"), MAX_MESSAGE_CHARS)
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    client = get_openai_client()
    if client is None:
        logger.error("OPENAI_API_KEY is not configured.")
        return jsonify({"error": "Chat is not configured"}), 503

    history = normalize_history(payload.get("history"), user_input)
    page_context = summarize_page_context(payload.get("page_context"))

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=build_messages(user_input, history, page_context),
            temperature=0.35,
            max_tokens=700,
        )
        answer = response.choices[0].message.content or ""
        return jsonify(
            {
                "response": answer,
                "model": MODEL,
                "history_used": len(history),
                "context_version": profile_context.get("version", "unknown"),
            }
        )

    except Exception as exc:
        logger.exception("Error during OpenAI chat completion: %s", exc)
        return jsonify({"error": "Server error"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model": MODEL,
            "context_version": profile_context.get("version", "unknown"),
            "resume_loaded": bool(resume_text and "unavailable" not in resume_text.lower()),
        }
    ), 200


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", port=int(os.getenv("PORT", "10000")), host="0.0.0.0")
