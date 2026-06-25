import json
import logging
import os
import time
from collections import defaultdict, deque
from hashlib import sha256
from threading import Lock
from typing import Any

import fitz  # PyMuPDF
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import APIConnectionError, APITimeoutError, RateLimitError
from openai import OpenAI


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "1400"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "1400"))
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "25"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "18"))
CLIENT_HASH_SALT = os.getenv("CLIENT_HASH_SALT", "resume-chat")

DEFAULT_ALLOWED_ORIGINS = [
    "https://bradenbradshaw.com",
    "https://www.bradenbradshaw.com",
    "https://bbradshawweb.github.io",
    "http://localhost:8765",
    "http://127.0.0.1:8765",
]


def configured_allowed_origins() -> list[str] | str:
    raw_origins = os.getenv("ALLOWED_ORIGINS")
    if not raw_origins:
        return DEFAULT_ALLOWED_ORIGINS

    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return "*" if origins == ["*"] else origins


ALLOWED_ORIGINS = configured_allowed_origins()
rate_limit_state: dict[str, deque[float]] = defaultdict(deque)
rate_limit_lock = Lock()


app = Flask(__name__)
CORS(
    app,
    resources={
        r"/chat": {"origins": ALLOWED_ORIGINS},
        r"/chat/v2": {"origins": ALLOWED_ORIGINS},
        r"/context": {"origins": "*"},
        r"/health": {"origins": "*"},
        r"/": {"origins": "*"},
    },
)


@app.after_request
def add_response_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Cache-Control", "no-store")
    return response


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


def client_ip() -> str:
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr or "unknown"


def client_hash() -> str:
    digest = sha256(f"{CLIENT_HASH_SALT}:{client_ip()}".encode("utf-8")).hexdigest()
    return digest[:12]


def check_rate_limit(key: str) -> tuple[bool, int]:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    with rate_limit_lock:
        attempts = rate_limit_state[key]
        while attempts and attempts[0] < window_start:
            attempts.popleft()

        if len(attempts) >= RATE_LIMIT_MAX_REQUESTS:
            retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - attempts[0])))
            return False, retry_after

        attempts.append(now)
        return True, 0


def log_chat_event(event: str, **metadata: Any) -> None:
    safe_metadata = {
        key: value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    logger.info("chat_event=%s %s", event, json.dumps(safe_metadata, sort_keys=True))


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


def pick_followups(user_message: str, answer: str) -> list[str]:
    prompt_groups = profile_context.get("follow_up_prompts", {})
    if not isinstance(prompt_groups, dict):
        return []

    combined = f"{user_message} {answer}".lower()
    group_name = "general"
    if any(term in combined for term in ("revenue", "revops", "lead", "salesforce", "routing", "handoff", "workcenter")):
        group_name = "revenue_operations"
    elif any(term in combined for term in ("model", "analytics", "attribution", "dashboard", "predictive", "statistics", "kpi")):
        group_name = "data_science"
    elif any(term in combined for term in ("research", "behavior", "psychology", "survey", "longitudinal", "publication")):
        group_name = "research"

    prompts = prompt_groups.get(group_name) or prompt_groups.get("general") or []
    return [prompt for prompt in prompts[:3] if isinstance(prompt, str)]


def build_system_prompt(page_context: str) -> str:
    context_json = json.dumps(profile_context, ensure_ascii=True, indent=2)

    prompt = f"""
You are Braden Bradshaw's interactive resume assistant embedded on bradenbradshaw.com.

Primary goal:
- Help visitors understand Braden's fit for data science, Revenue Operations, product analytics, marketing analytics, automation, and behavioral research roles.

Voice and behavior:
- The website is framed as Braden speaking through his resume. Treat second-person questions using "you" or "your" as questions directed to Braden, not to the AI model.
- Prefer first-person answers as Braden using "I", "my", and "me" when describing experience, projects, skills, and fit.
- Be concise, specific, warm, and confident.
- Lead with Revenue Operations plus data science when relevant.
- Use concrete evidence from the provided context: metrics, projects, tools, roles, and research background.
- Do not invent employers, credentials, dates, technologies, outcomes, links, or contact details.
- If the answer is not in the context, say that you do not have that detail and pivot to the closest relevant known experience.
- For hiring or fit questions, connect evidence to business outcomes.
- Keep most answers to 2-4 short paragraphs unless the user asks for detail.
- When a question is broad or vague, choose the strongest documented angle and answer with specific projects or metrics rather than generic traits.
- Prefer practical business language over academic jargon unless the user asks about statistical methods.
- Mention limitations plainly when needed: for example, the Master of Statistics is ongoing.

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
    follow_up_prompts = profile_context.get("follow_up_prompts", {})
    if not isinstance(follow_up_prompts, dict):
        follow_up_prompts = {}

    return jsonify(
        {
            "version": profile_context.get("version", "unknown"),
            "positioning": profile_context.get("positioning"),
            "summary": profile_context.get("summary"),
            "chat_intro": profile_context.get("chat_intro"),
            "initial_message": profile_context.get("initial_message"),
            "chat_config": profile_context.get("chat_config", {}),
            "prompt_chips": profile_context.get("prompt_chips", []),
            "suggested_prompts": profile_context.get("suggested_prompts", []),
            "follow_up_prompts": follow_up_prompts.get("general", []),
            "preferred_framing": profile_context.get("response_guidance", {}).get("preferred_framing", []),
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

    client_key = client_hash()
    allowed, retry_after = check_rate_limit(client_key)
    if not allowed:
        log_chat_event("rate_limited", client=client_key, retry_after=retry_after)
        response = jsonify(
            {
                "error": "Too many requests",
                "message": "The resume assistant is receiving a lot of traffic. Please try again shortly.",
                "retry_after": retry_after,
            }
        )
        response.status_code = 429
        response.headers["Retry-After"] = str(retry_after)
        return response

    client = get_openai_client()
    if client is None:
        logger.error("OPENAI_API_KEY is not configured.")
        log_chat_event("missing_api_key", client=client_key)
        return jsonify({"error": "Chat is not configured"}), 503

    history = normalize_history(payload.get("history"), user_input)
    page_context = summarize_page_context(payload.get("page_context"))
    started_at = time.perf_counter()
    log_chat_event(
        "request_started",
        client=client_key,
        message_chars=len(user_input),
        history_used=len(history),
        page_context=bool(page_context),
        origin=request.headers.get("Origin"),
    )

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
                "suggested_followups": pick_followups(user_input, answer),
            }
        )

    except RateLimitError as exc:
        logger.warning("OpenAI rate limit during chat completion: %s", exc)
        log_chat_event("upstream_rate_limited", client=client_key)
        response = jsonify(
            {
                "error": "Assistant is busy",
                "message": "The assistant is temporarily busy. Please try again in a moment.",
                "retry_after": 20,
            }
        )
        response.status_code = 429
        response.headers["Retry-After"] = "20"
        return response

    except (APITimeoutError, APIConnectionError) as exc:
        logger.warning("OpenAI connection issue during chat completion: %s", exc)
        log_chat_event("upstream_unavailable", client=client_key)
        return jsonify(
            {
                "error": "Assistant unavailable",
                "message": "The assistant is waking up or temporarily unreachable. Please try again in a moment.",
            }
        ), 503

    except Exception as exc:
        logger.exception("Error during OpenAI chat completion: %s", exc)
        log_chat_event("server_error", client=client_key)
        return jsonify({"error": "Server error"}), 500
    finally:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        log_chat_event("request_finished", client=client_key, elapsed_ms=elapsed_ms)


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
