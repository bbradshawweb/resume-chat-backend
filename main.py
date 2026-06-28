import json
import logging
import os
import re
import smtplib
import time
from collections import defaultdict, deque
from email.message import EmailMessage
from email.utils import formatdate
from hashlib import sha256
from threading import Lock
from typing import Any

import fitz  # PyMuPDF
import httpx
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import APIConnectionError, APITimeoutError, RateLimitError
from openai import OpenAI


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "1400"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "1400"))
MAX_CONTACT_FIELD_CHARS = int(os.getenv("MAX_CONTACT_FIELD_CHARS", "240"))
MAX_CONTACT_SUMMARY_CHARS = int(os.getenv("MAX_CONTACT_SUMMARY_CHARS", "1800"))
MAX_CONTACT_NOTE_CHARS = int(os.getenv("MAX_CONTACT_NOTE_CHARS", "1200"))
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "25"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "18"))
CLIENT_HASH_SALT = os.getenv("CLIENT_HASH_SALT", "resume-chat")
NOTIFICATION_ENABLED = env_bool("NOTIFICATION_ENABLED", False)
NOTIFICATION_TO_EMAIL = os.getenv("NOTIFICATION_TO_EMAIL", "bradshaw.braden@gmail.com")
NOTIFICATION_FROM_EMAIL = (
    os.getenv("NOTIFICATION_FROM_EMAIL")
    or os.getenv("SMTP_FROM_EMAIL")
    or os.getenv("SMTP_USERNAME")
)
NOTIFICATION_MIN_SCORE = int(os.getenv("NOTIFICATION_MIN_SCORE", "7"))
NOTIFICATION_COOLDOWN_SECONDS = int(os.getenv("NOTIFICATION_COOLDOWN_SECONDS", "21600"))
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RESEND_API_URL = os.getenv("RESEND_API_URL", "https://api.resend.com/emails")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL")
RESEND_USER_AGENT = os.getenv("RESEND_USER_AGENT", "resume-chat-backend/1.0 (+https://bradenbradshaw.com)")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_USE_TLS = env_bool("SMTP_USE_TLS", True)
SMTP_USE_SSL = env_bool("SMTP_USE_SSL", False)
SMTP_TIMEOUT_SECONDS = float(os.getenv("SMTP_TIMEOUT_SECONDS", "10"))
EMAIL_TIMEOUT_SECONDS = float(os.getenv("EMAIL_TIMEOUT_SECONDS", str(SMTP_TIMEOUT_SECONDS)))

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
notification_state: dict[str, float] = {}
notification_lock = Lock()


app = Flask(__name__)
CORS(
    app,
    resources={
        r"/chat": {"origins": ALLOWED_ORIGINS},
        r"/chat/v2": {"origins": ALLOWED_ORIGINS},
        r"/contact": {"origins": ALLOWED_ORIGINS},
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


def clean_multiline_text(value: Any, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    lines = [" ".join(line.split()) for line in value.strip().splitlines()]
    return "\n".join(line for line in lines if line)[:max_chars]


def clean_context_value(value: Any, max_chars: int = 700) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return clean_text(value, max_chars)
    if isinstance(value, list):
        cleaned_items = [clean_context_value(item, 180) for item in value[:8]]
        return " | ".join(item for item in cleaned_items if item)[:max_chars]
    if isinstance(value, dict):
        safe_items = {
            str(key)[:40]: clean_context_value(item, 220)
            for key, item in value.items()
            if isinstance(key, str)
        }
        return json.dumps(safe_items, ensure_ascii=True)[:max_chars]
    return ""


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


EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def clip_text(value: str, max_chars: int = 1200) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[: max_chars - 3]}..."


def matched_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    matches = set()
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", text):
            matches.add(term)
    return sorted(matches)


def extract_visitor_emails(transcript_text: str) -> list[str]:
    return sorted(set(EMAIL_PATTERN.findall(transcript_text)))[:5]


def is_valid_email(value: str) -> bool:
    return bool(EMAIL_PATTERN.fullmatch(value))


def notification_configured() -> bool:
    has_resend = bool(RESEND_API_KEY and (RESEND_FROM_EMAIL or NOTIFICATION_FROM_EMAIL))
    has_smtp = bool(NOTIFICATION_FROM_EMAIL and SMTP_HOST)
    return bool(NOTIFICATION_ENABLED and NOTIFICATION_TO_EMAIL and (has_resend or has_smtp))


def notification_provider() -> str:
    if RESEND_API_KEY and (RESEND_FROM_EMAIL or NOTIFICATION_FROM_EMAIL):
        return "resend"
    if NOTIFICATION_FROM_EMAIL and SMTP_HOST:
        return "smtp"
    return "none"


def notification_config_status() -> dict[str, bool | str]:
    return {
        "provider": notification_provider(),
        "enabled": NOTIFICATION_ENABLED,
        "to_email_present": bool(NOTIFICATION_TO_EMAIL),
        "resend_api_key_present": bool(RESEND_API_KEY),
        "resend_from_present": bool(RESEND_FROM_EMAIL),
        "notification_from_present": bool(NOTIFICATION_FROM_EMAIL),
        "smtp_host_present": bool(SMTP_HOST),
    }


def score_conversation_strength(
    user_message: str,
    answer: str,
    history: list[dict[str, str]],
) -> dict[str, Any]:
    user_turns = sum(1 for item in history if item.get("role") == "user") + 1
    transcript_text = " ".join(
        item.get("content", "")
        for item in [*history, {"role": "user", "content": user_message}, {"role": "assistant", "content": answer}]
    )
    visitor_text = " ".join(
        item.get("content", "")
        for item in [*history, {"role": "user", "content": user_message}]
        if item.get("role") == "user"
    ).lower()
    text = transcript_text.lower()
    score = 0
    reasons: list[str] = []

    contact_terms = (
        "call",
        "coffee chat",
        "connect on linkedin",
        "connect with",
        "contact",
        "email",
        "interview",
        "linkedin",
        "reach out",
        "schedule",
        "set up a call",
        "talk",
    )
    hiring_terms = (
        "candidate",
        "contract",
        "hire",
        "hiring",
        "job",
        "opening",
        "opportunity",
        "position",
        "recruiter",
        "recruiting",
        "talent",
    )
    evaluation_terms = (
        "analytics projects",
        "automation",
        "case study",
        "dashboard",
        "data science",
        "fit",
        "impact",
        "lead scoring",
        "portfolio",
        "project",
        "revenue growth",
        "revops",
        "salesforce",
        "strong fit",
    )

    visitor_emails = extract_visitor_emails(transcript_text)
    if visitor_emails:
        score += 5
        reasons.append("visitor shared an email address")

    contact_matches = matched_terms(visitor_text, contact_terms)
    if contact_matches:
        score += 4
        reasons.append(f"contact intent: {', '.join(contact_matches[:4])}")

    hiring_matches = matched_terms(visitor_text, hiring_terms)
    if hiring_matches:
        score += 4
        reasons.append(f"hiring intent: {', '.join(hiring_matches[:4])}")

    evaluation_matches = matched_terms(text, evaluation_terms)
    if evaluation_matches:
        score += min(4, len(evaluation_matches))
        reasons.append(f"role/project evaluation: {', '.join(evaluation_matches[:5])}")

    if user_turns >= 4:
        score += 3
        reasons.append(f"{user_turns} visitor turns")
    elif user_turns >= 2:
        score += 1
        reasons.append(f"{user_turns} visitor turns")

    if len(transcript_text) >= 1600:
        score += 2
        reasons.append("substantial transcript length")

    should_notify = score >= NOTIFICATION_MIN_SCORE and (
        user_turns >= 2 or bool(contact_matches or hiring_matches or visitor_emails)
    )
    return {
        "score": score,
        "should_notify": should_notify,
        "reasons": reasons,
        "user_turns": user_turns,
        "visitor_emails": visitor_emails,
        "contact_matches": contact_matches,
        "hiring_matches": hiring_matches,
        "evaluation_matches": evaluation_matches,
    }


def should_offer_contact_handoff(signal: dict[str, Any]) -> bool:
    if signal.get("contact_matches") or signal.get("hiring_matches") or signal.get("visitor_emails"):
        return True
    if signal.get("evaluation_matches") and signal.get("user_turns", 0) >= 3 and signal.get("score", 0) >= 5:
        return True
    return bool(signal.get("should_notify"))


def build_interest_summary(user_message: str, answer: str, history: list[dict[str, str]], signal: dict[str, Any]) -> str:
    visitor_messages = [
        item.get("content", "")
        for item in [*history, {"role": "user", "content": user_message}]
        if item.get("role") == "user" and item.get("content")
    ]
    latest_messages = visitor_messages[-3:]
    summary_parts = []
    if latest_messages:
        summary_parts.append("Visitor interest:")
        summary_parts.extend(f"- {clip_text(message, 260)}" for message in latest_messages)

    reasons = signal.get("reasons") or []
    if reasons:
        summary_parts.append("")
        summary_parts.append(f"Detected signals: {', '.join(reasons)}")

    if answer:
        summary_parts.append("")
        summary_parts.append(f"Latest assistant context: {clip_text(answer, 420)}")

    return "\n".join(summary_parts)[:MAX_CONTACT_SUMMARY_CHARS]


def build_contact_handoff(user_message: str, answer: str, history: list[dict[str, str]]) -> dict[str, Any] | None:
    signal = score_conversation_strength(user_message, answer, history)
    if not should_offer_contact_handoff(signal):
        return None

    visitor_emails = signal.get("visitor_emails") or []
    return {
        "enabled": True,
        "summary": build_interest_summary(user_message, answer, history, signal),
        "email": visitor_emails[0] if visitor_emails else "",
        "score": signal["score"],
        "reasons": signal.get("reasons", []),
        "cta": "Send to Braden",
    }


def reserve_notification_slot(client_key: str) -> tuple[bool, str]:
    if not NOTIFICATION_ENABLED:
        return False, "disabled"
    if not notification_configured():
        return False, "missing_email_config"

    now = time.time()
    with notification_lock:
        last_sent_at = notification_state.get(client_key)
        if (
            last_sent_at is not None
            and NOTIFICATION_COOLDOWN_SECONDS > 0
            and now - last_sent_at < NOTIFICATION_COOLDOWN_SECONDS
        ):
            return False, "cooldown"

        notification_state[client_key] = now
        return True, "reserved"


def format_transcript(history: list[dict[str, str]], user_message: str, answer: str) -> str:
    transcript = [*history]
    if user_message:
        transcript.append({"role": "user", "content": user_message})
    if answer:
        transcript.append({"role": "assistant", "content": answer})
    lines: list[str] = []
    for item in transcript[-10:]:
        role = "Visitor" if item.get("role") == "user" else "Assistant"
        content = clip_text(item.get("content", ""), 1600)
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def build_contact_submission_email(
    client_key: str,
    name: str,
    email: str,
    company: str,
    summary: str,
    additional_context: str,
    history: list[dict[str, str]],
    origin: str | None,
    user_agent: str | None,
) -> EmailMessage:
    display_name = name or "Portfolio visitor"
    subject_detail = company or email
    message = EmailMessage()
    message["To"] = NOTIFICATION_TO_EMAIL
    message["From"] = NOTIFICATION_FROM_EMAIL or ""
    message["Date"] = formatdate(localtime=True)
    message["Subject"] = f"Portfolio contact request from {display_name} ({subject_detail})"
    message["Reply-To"] = email

    body = f"""A visitor approved sending their portfolio-chat interest to you.

Name: {name or "Not provided"}
Email: {email}
Company / role: {company or "Not provided"}
Client hash: {client_key}
Origin: {origin or "unknown"}
User agent: {clip_text(user_agent or "unknown", 260)}

Generated conversation summary:
{summary or "No generated summary was provided. See transcript below."}

Visitor note / anything else to add:
{additional_context or "Not provided"}

Recent transcript:
{format_transcript(history, "", "")}
"""
    message.set_content(body)
    return message


def deliver_resend_email(message: EmailMessage, client_key: str, score: int) -> bool:
    from_email = RESEND_FROM_EMAIL or NOTIFICATION_FROM_EMAIL
    if not RESEND_API_KEY or not from_email:
        return False

    payload = {
        "from": from_email,
        "to": [NOTIFICATION_TO_EMAIL],
        "subject": message["Subject"],
        "text": message.get_content(),
    }
    if message.get("Reply-To"):
        payload["reply_to"] = message["Reply-To"]

    try:
        response = httpx.post(
            RESEND_API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": RESEND_USER_AGENT,
            },
            timeout=EMAIL_TIMEOUT_SECONDS,
        )
        if not 200 <= response.status_code < 300:
            logger.error("Resend email request failed status=%s body=%s", response.status_code, response.text)
            log_chat_event(
                "notification_failed",
                client=client_key,
                score=score,
                provider="resend",
                provider_status=response.status_code,
            )
            return False

        log_chat_event(
            "notification_sent",
            client=client_key,
            score=score,
            provider="resend",
            provider_status=response.status_code,
        )
        logger.info("resend_response=%s", response.text)
        return True
    except httpx.HTTPError as exc:
        logger.exception("Failed to send Resend notification email: %s", exc)
        log_chat_event("notification_failed", client=client_key, score=score, provider="resend")

    return False


def deliver_smtp_email(message: EmailMessage, client_key: str, score: int) -> bool:
    if not NOTIFICATION_FROM_EMAIL or not SMTP_HOST:
        return False

    try:
        smtp_cls = smtplib.SMTP_SSL if SMTP_USE_SSL else smtplib.SMTP
        with smtp_cls(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT_SECONDS) as server:
            if SMTP_USE_TLS and not SMTP_USE_SSL:
                server.starttls()
            if SMTP_USERNAME and SMTP_PASSWORD:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(message)

        log_chat_event("notification_sent", client=client_key, score=score, provider="smtp")
        return True
    except Exception as exc:
        logger.exception("Failed to send SMTP notification email: %s", exc)
        log_chat_event("notification_failed", client=client_key, score=score, provider="smtp")
        return False


def deliver_notification_email(message: EmailMessage, client_key: str, score: int) -> bool:
    if RESEND_API_KEY:
        return deliver_resend_email(message, client_key, score)

    log_chat_event("notification_provider_selected", client=client_key, provider="smtp", config=json.dumps(notification_config_status(), sort_keys=True))
    return deliver_smtp_email(message, client_key, score)


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
        str(key)[:60]: clean_context_value(value)
        for key, value in raw_context.items()
        if isinstance(key, str)
    }
    safe_context = {key: value for key, value in safe_context.items() if value}
    if not safe_context:
        return ""

    return json.dumps(safe_context, ensure_ascii=True)


def pick_followups(user_message: str, answer: str) -> list[str]:
    prompt_groups = profile_context.get("follow_up_prompts", {})
    if not isinstance(prompt_groups, dict):
        return []

    combined = f"{user_message} {answer}".lower()
    group_name = "general"
    if any(term in combined for term in ("revenue", "revops", "lead", "salesforce", "routing", "handoff", "workcenter", "manager", "console", "probation", "throttle", "capacity", "rep")):
        group_name = "revenue_operations"
    elif any(term in combined for term in ("marketing", "campaign", "source", "medium", "utm", "event", "lead quality", "cpa", "ad spend")):
        group_name = "marketing_analytics"
    elif any(term in combined for term in ("research", "behavior", "psychology", "survey", "longitudinal", "publication", "toxicity", "wikipedia", "biometric", "wearable", "loneliness", "social connection")):
        group_name = "research"
    elif any(term in combined for term in ("model", "analytics", "attribution", "dashboard", "predictive", "statistics", "kpi")):
        group_name = "data_science"

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
- Be proactive: end most answers with one short, useful question that helps qualify what the visitor cares about next. Ask about the role, team, business problem, data stack, timeline, or whether they want a concrete project example.
- Do not ask more than one question at a time, and do not ask a follow-up when the visitor clearly requested only a short factual answer.
- If a visitor expresses hiring, recruiting, consulting, interview, or contact intent, answer the question and ask whether they would like me to pass the conversation along to Braden. Do not ask them to review a summary, and do not promise an immediate response.
- Keep most answers to 2-4 short paragraphs unless the user asks for detail.
- Format answers for scanning: use short paragraphs, bullets for lists, and avoid dense blocks of text.
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
            "featured_projects": profile_context.get("projects", []),
            "preferred_framing": profile_context.get("response_guidance", {}).get("preferred_framing", []),
            "model": MODEL,
        }
    )


@app.route("/contact", methods=["POST"])
def contact():
    payload = request.get_json(silent=True) or {}
    client_key = client_hash()
    allowed, retry_after = check_rate_limit(f"contact:{client_key}")
    if not allowed:
        response = jsonify(
            {
                "error": "Too many contact attempts",
                "message": "Please wait a moment before sending another contact request.",
                "retry_after": retry_after,
            }
        )
        response.status_code = 429
        response.headers["Retry-After"] = str(retry_after)
        return response

    if not notification_configured():
        log_chat_event("contact_not_configured", client=client_key)
        return jsonify(
            {
                "error": "Contact email is not configured",
                "message": "Email sending is not fully configured yet. Please email me directly at bradshaw.braden@gmail.com for now.",
            }
        ), 503

    name = clean_text(payload.get("name"), MAX_CONTACT_FIELD_CHARS)
    email = clean_text(payload.get("email"), MAX_CONTACT_FIELD_CHARS).lower()
    company = clean_text(payload.get("company"), MAX_CONTACT_FIELD_CHARS)
    summary = clean_multiline_text(payload.get("summary"), MAX_CONTACT_SUMMARY_CHARS)
    additional_context = clean_multiline_text(payload.get("additional_context"), MAX_CONTACT_NOTE_CHARS)
    if not email or not is_valid_email(email):
        return jsonify({"error": "A valid email address is required"}), 400

    allowed_notification, reason = reserve_notification_slot(client_key)
    if not allowed_notification:
        status_code = 429 if reason == "cooldown" else 503
        return jsonify(
            {
                "error": "Contact request not sent",
                "message": "This contact request was not sent because email delivery is cooling down or unavailable.",
                "reason": reason,
            }
        ), status_code

    history = normalize_history(payload.get("history"), "")
    if not summary:
        summary = clip_text(format_transcript(history, "", ""), MAX_CONTACT_SUMMARY_CHARS)

    origin = request.headers.get("Origin")
    user_agent = clean_text(request.headers.get("User-Agent"), 260)
    message = build_contact_submission_email(
        client_key=client_key,
        name=name,
        email=email,
        company=company,
        summary=summary,
        additional_context=additional_context,
        history=history,
        origin=origin,
        user_agent=user_agent,
    )
    sent = deliver_notification_email(message, client_key, NOTIFICATION_MIN_SCORE)
    if not sent:
        with notification_lock:
            notification_state.pop(client_key, None)
        return jsonify(
            {
                "error": "Contact request failed",
                "message": "I could not send this through the site yet. Please email me directly at bradshaw.braden@gmail.com for now.",
                "provider": notification_provider(),
            }
        ), 502

    log_chat_event(
        "contact_request_sent",
        client=client_key,
        has_name=bool(name),
        has_company=bool(company),
        has_note=bool(additional_context),
    )
    return jsonify({"ok": True, "message": "Sent. Thanks - I will see this conversation, your contact info, and your note."})


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
    origin = request.headers.get("Origin")
    user_agent = clean_text(request.headers.get("User-Agent"), 260)
    started_at = time.perf_counter()
    log_chat_event(
        "request_started",
        client=client_key,
        message_chars=len(user_input),
        history_used=len(history),
        page_context=bool(page_context),
        origin=origin,
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=build_messages(user_input, history, page_context),
            temperature=0.35,
            max_tokens=700,
        )
        answer = response.choices[0].message.content or ""
        contact_handoff = build_contact_handoff(user_input, answer, history)
        return jsonify(
            {
                "response": answer,
                "model": MODEL,
                "history_used": len(history),
                "context_version": profile_context.get("version", "unknown"),
                "suggested_followups": [] if contact_handoff else pick_followups(user_input, answer),
                "contact_handoff": contact_handoff,
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
            "notifications_enabled": NOTIFICATION_ENABLED,
            "notifications_configured": notification_configured(),
            "notification_provider": notification_provider(),
            "notification_config": notification_config_status(),
        }
    ), 200


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", port=int(os.getenv("PORT", "10000")), host="0.0.0.0")
