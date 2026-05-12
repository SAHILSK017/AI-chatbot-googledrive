import datetime
import json
import logging
import os
import re
import threading
import time
from typing import Any

from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

from backend.google_drive import DriveConfigurationError, DriveRateLimitError, dedupe_files
from backend.tools import DriveSearchTool, SearchFolderContentsTool, parse_query, staged_search

logger = logging.getLogger(__name__)

_agent = None
_agent_lock = threading.RLock()
_memory = ConversationBufferWindowMemory(
    k=int(os.getenv("MEMORY_WINDOW", "3")),
    memory_key="chat_history",
    return_messages=True,
    output_key="output",
)

SEARCH_TERMS = {
    "assignment", "code", "doc", "docs", "document", "documents", "file", "files",
    "find", "folder", "folders", "image", "images", "internship", "invoice",
    "invoices", "java", "jpg", "jpeg", "latest", "list", "open", "pdf", "pdfs",
    "photo", "photos", "pic", "pics", "png", "recent", "resume", "sheet",
    "sheets", "show", "spreadsheet", "spreadsheets",
}
SEARCH_STARTERS = {"find", "show", "open", "list", "get", "search"}
GROQ_LIMIT_MARKERS = (
    "413",
    "request too large",
    "rate_limit_exceeded",
    "tokens per minute",
    "token limit",
    "rate limit",
    "429",
)


def _build_system_prompt() -> str:
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    return (
        f"Drive Assistant. Today: {today}. "
        "Use tools for Drive search. Be concise. Never invent files."
    )


def _get_agent():
    global _agent
    with _agent_lock:
        if _agent is not None:
            logger.debug("Reusing cached LangChain agent")
            return _agent

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY missing; cannot initialize chat model")
            raise ValueError("GROQ_API_KEY missing")

        logger.info("Initializing LangChain agent model=%s", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
        llm = ChatGroq(
            temperature=0.0,
            model_name=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            groq_api_key=api_key,
            request_timeout=float(os.getenv("AI_REQUEST_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.getenv("AI_MAX_RETRIES", "1")),
        )
        _agent = initialize_agent(
            [DriveSearchTool(), SearchFolderContentsTool()],
            llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=_memory,
            verbose=os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true",
            return_intermediate_steps=False,
            handle_parsing_errors=True,
            max_iterations=2,
            agent_kwargs={"system_message": _build_system_prompt()},
        )
        logger.info("LangChain agent initialized")
        return _agent


def _looks_like_search_query(message: str) -> bool:
    words = set(re.findall(r"[a-zA-Z0-9_+#.-]+", message.lower()))
    return len(words) == 1 or bool(words & SEARCH_TERMS)


def _is_simple_search(message: str) -> bool:
    words = re.findall(r"[a-zA-Z0-9_+#.-]+", message.lower())
    if not words:
        return False
    if len(words) == 1:
        return True
    return len(words) <= 4 and (
        words[0] in SEARCH_STARTERS or bool(set(words) & SEARCH_TERMS)
    )


def _is_groq_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in GROQ_LIMIT_MARKERS)


def _files_from_observation(observation: Any) -> list[dict[str, Any]]:
    if observation is None:
        return []

    payload = observation
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Tool returned invalid JSON observation=%r", payload[:300])
            return []

    if isinstance(payload, dict):
        files = payload.get("files", [])
        return files if isinstance(files, list) else []
    if isinstance(payload, list):
        return payload

    logger.warning("Unexpected tool observation type=%s", type(payload).__name__)
    return []


def _format_response(message: str, files: list[dict[str, Any]]) -> str:
    parsed = parse_query(message)
    count = len(files)
    if count == 0:
        return f"I couldn't find any files matching '{message}'."

    if parsed.wants_folders:
        label = "folder" if count == 1 else "folders"
        return f"I found {count} matching {label}."
    if parsed.folder_scoped and parsed.folder_terms:
        label = "item" if count == 1 else "items"
        folder_name = " ".join(parsed.folder_terms)
        return f"I found {count} matching {label} in the {folder_name} folder."
    if parsed.open_intent and count == 1:
        return "I found 1 matching file. Open it from the result below."

    label = "item" if count == 1 else "items"
    return f"I found {count} matching {label}."


def _agent_invoke(message: str) -> tuple[str, list[dict[str, Any]], bool]:
    started = time.monotonic()
    logger.info("AI response generation started prompt_length=%s", len(message or ""))
    agent = _get_agent()
    with _agent_lock:
        result = agent.invoke({"input": message})
    text = result.get("output", "") if isinstance(result, dict) else str(result)
    logger.info(
        "AI response generation complete duration_ms=%s response_length=%s",
        int((time.monotonic() - started) * 1000),
        len(text or ""),
    )
    return text, [], False


def _enhance_with_agent(message: str, files: list[dict[str, Any]]) -> str | None:
    if not files or os.getenv("AI_ENHANCE_RESULTS", "false").lower() != "true":
        return None
    if not os.getenv("GROQ_API_KEY"):
        logger.info("Skipping AI enhancement because GROQ_API_KEY is not configured")
        return None
    if _is_simple_search(message) or len(files) > int(os.getenv("AI_ENHANCE_MAX_FILES", "5")):
        return None

    names = [file.get("name", "file") for file in files[:3]]
    prompt = (
        f"Summarize Drive search result in one short sentence. "
        f"Query: {message}. Count: {len(files)}. Examples: {', '.join(names)}."
    )
    try:
        logger.info("AI enhancement started files=%s query=%r", len(files), message[:300])
        text, _, _ = _agent_invoke(prompt)
    except Exception as exc:
        if _is_groq_limit_error(exc):
            logger.warning("Skipping enhancement due to Groq token/rate limit: %s", exc)
        else:
            logger.exception("Agent enhancement failed; using deterministic response")
        return None

    lowered = text.lower()
    if not text.strip() or ("no file" in lowered and files):
        return None
    return text.strip()


def chat_with_agent(message: str):
    """Deterministic Drive search first, LangChain agent second."""
    started = time.monotonic()
    clean = (message or "").strip()
    logger.info("Chat pipeline started query=%r", clean[:500])
    if not clean:
        logger.info("Chat pipeline received empty query")
        return "Ask me what you want to find in Google Drive.", [], False

    is_search_query = _looks_like_search_query(clean)
    direct_files: list[dict[str, Any]] = []
    logger.info("Chat query classified search_query=%s", is_search_query)

    if is_search_query:
        try:
            logger.info("Deterministic Drive search started query=%r", clean[:500])
            direct_files = dedupe_files(staged_search(clean))
            logger.info("Deterministic Drive search complete count=%s query=%r", len(direct_files), clean[:500])
            if direct_files:
                if _is_simple_search(clean):
                    logger.info("Chat pipeline complete via deterministic simple search")
                    return _format_response(clean, direct_files), direct_files, True
                enhanced = _enhance_with_agent(clean, direct_files)
                logger.info("Chat pipeline complete via deterministic search enhanced=%s", bool(enhanced))
                return enhanced or _format_response(clean, direct_files), direct_files, True
        except DriveConfigurationError as exc:
            logger.error("Drive configuration error: %s", exc)
            return (
                "Google Drive search is not configured on the server. "
                "Set GOOGLE_SERVICE_ACCOUNT_JSON on Render or mount credentials.json.",
                [],
                True,
            )
        except DriveRateLimitError:
            logger.exception("Drive rate limit during deterministic search")
            return "Google Drive is rate-limiting searches right now. Please try again shortly.", [], True
        except Exception:
            logger.exception("Direct deterministic search failed message=%r", clean)

    try:
        logger.info("LangChain agent fallback started query=%r", clean[:500])
        text, agent_files, tool_used = _agent_invoke(clean)
        files = dedupe_files(direct_files + agent_files)

        if is_search_query and not tool_used:
            logger.warning("Agent skipped tool for search query=%r; forcing google_drive_search", clean)
            forced_files = _files_from_observation(DriveSearchTool()._run(clean))
            files = dedupe_files(files + forced_files)
            tool_used = True

        if files:
            if not text or ("no file" in text.lower() and files):
                text = _format_response(clean, files)
            logger.info("Chat pipeline complete with files=%s duration_ms=%s", len(files), int((time.monotonic() - started) * 1000))
            return text, files, True

        if is_search_query:
            logger.info("Chat pipeline complete no search results duration_ms=%s", int((time.monotonic() - started) * 1000))
            return _format_response(clean, []), [], True
        logger.info("Chat pipeline complete no files duration_ms=%s", int((time.monotonic() - started) * 1000))
        return text or "I can help search your Google Drive.", [], tool_used

    except Exception as exc:
        if _is_groq_limit_error(exc):
            logger.exception("LLM rate limit; returning deterministic fallback")
            if direct_files:
                return _format_response(clean, direct_files), direct_files, True
            if is_search_query:
                try:
                    fallback_files = dedupe_files(staged_search(clean))
                    return _format_response(clean, fallback_files), fallback_files, True
                except Exception:
                    logger.exception("Deterministic fallback after Groq limit failed")
            return "The AI service is busy right now. Please try again shortly.", [], False

        logger.exception("LangChain agent failed message=%r", clean)
        if is_search_query:
            try:
                fallback_files = dedupe_files(staged_search(clean))
                return _format_response(clean, fallback_files), fallback_files, True
            except Exception:
                logger.exception("Final deterministic fallback failed")
                return "I couldn't complete that Drive search. Please try a simpler search.", [], True

        return "I encountered a problem processing your request.", [], False
