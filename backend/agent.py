import datetime
import json
import logging
import os
import re
from typing import Any

from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

from backend.google_drive import DriveConfigurationError, DriveRateLimitError, dedupe_files
from backend.tools import DriveSearchTool, SearchFolderContentsTool, parse_query, staged_search

logger = logging.getLogger(__name__)

_agent = None
_memory = ConversationBufferMemory(
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


def _build_system_prompt() -> str:
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    return f"""Drive Assistant (Today: {today}).
You are a concise Google Drive assistant.
For any request about finding, showing, opening, listing, or filtering Drive files, use google_drive_search.
Use search_folder_contents only when the user asks for contents inside a named folder.
Never invent file names. If tools return no files, say no matching files were found.
Keep responses to one or two short sentences."""


def _get_agent():
    global _agent
    if _agent is not None:
        return _agent

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing")

    llm = ChatGroq(
        temperature=0.0,
        model_name=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        groq_api_key=api_key,
    )
    _agent = initialize_agent(
        [DriveSearchTool(), SearchFolderContentsTool()],
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=_memory,
        verbose=os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true",
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": _build_system_prompt()},
    )
    return _agent


def _looks_like_search_query(message: str) -> bool:
    words = set(re.findall(r"[a-zA-Z0-9_+#.-]+", message.lower()))
    return len(words) == 1 or bool(words & SEARCH_TERMS)


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


def _extract_files_from_intermediate_steps(steps: Any) -> list[dict[str, Any]]:
    if not isinstance(steps, list):
        logger.warning("intermediate_steps was %s, expected list", type(steps).__name__)
        return []

    files: list[dict[str, Any]] = []
    for step in steps:
        observation = None
        if isinstance(step, tuple) and len(step) >= 2:
            observation = step[1]
        elif isinstance(step, dict):
            observation = step.get("observation") or step.get("tool_output")
        else:
            observation = getattr(step, "observation", None)
        files.extend(_files_from_observation(observation))

    return dedupe_files(files)


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
    agent = _get_agent()
    result = agent.invoke({"input": message})
    text = result.get("output", "") if isinstance(result, dict) else str(result)
    steps = result.get("intermediate_steps", []) if isinstance(result, dict) else []
    files = _extract_files_from_intermediate_steps(steps)
    return text, files, bool(steps)


def _enhance_with_agent(message: str, files: list[dict[str, Any]]) -> str | None:
    if not files or os.getenv("AI_ENHANCE_RESULTS", "true").lower() == "false":
        return None
    if not os.getenv("GROQ_API_KEY"):
        return None

    preview = [
        {"name": file.get("name"), "mimeType": file.get("mimeType")}
        for file in files[:8]
    ]
    prompt = (
        f"User asked: {message}\n"
        f"Deterministic Drive search found {len(files)} files: {json.dumps(preview)}\n"
        "Respond with one concise sentence summarizing the result. "
        "Do not claim there are no files."
    )
    try:
        text, _, _ = _agent_invoke(prompt)
    except Exception:
        logger.exception("Agent enhancement failed; using deterministic response")
        return None

    lowered = text.lower()
    if not text.strip() or ("no file" in lowered and files):
        return None
    return text.strip()


def chat_with_agent(message: str):
    """Deterministic Drive search first, LangChain agent second."""
    clean = (message or "").strip()
    if not clean:
        return "Ask me what you want to find in Google Drive.", [], False

    is_search_query = _looks_like_search_query(clean)
    direct_files: list[dict[str, Any]] = []

    if is_search_query:
        try:
            direct_files = dedupe_files(staged_search(clean))
            logger.debug("Direct search count=%s message=%r", len(direct_files), clean)
            if direct_files:
                enhanced = _enhance_with_agent(clean, direct_files)
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
            return text, files, True

        if is_search_query:
            return _format_response(clean, []), [], True
        return text or "I can help search your Google Drive.", [], tool_used

    except Exception as exc:
        err = str(exc).lower()
        if "429" in err or "rate limit" in err:
            logger.exception("LLM rate limit; returning deterministic fallback")
            if direct_files:
                return _format_response(clean, direct_files), direct_files, True
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
