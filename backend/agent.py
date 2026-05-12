import datetime
import json
import logging
import os
import re
from typing import Any

from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

from backend.google_drive import DriveRateLimitError
from backend.tools import DriveSearchTool, SearchFolderContentsTool, staged_search

_agent = None
_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
logger = logging.getLogger(__name__)

SEARCH_TERMS = {
    "file", "files", "folder", "folders", "find", "get", "list", "locate",
    "open", "pic", "pics", "picture", "pictures", "photo", "photos", "image",
    "images", "png", "jpg", "jpeg", "doc", "docs", "document", "documents",
    "pdf", "sheet", "sheets", "spreadsheet", "video", "videos",
}


def _build_system_prompt() -> str:
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    return f"""Drive Assistant (Today: {today}).
Core rule: Google Drive search is deterministic and tool-backed. For any request about finding, listing, opening, or filtering files, always call `google_drive_search`.
Use `search_folder_contents` only when the user asks for files inside a named folder.
Keep answers concise. Do not invent files. If tools return no files, say no matching files were found."""


def _get_agent():
    global _agent
    if _agent is not None:
        return _agent

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing")

    llm = ChatGroq(temperature=0.0, model_name="llama-3.1-8b-instant", groq_api_key=api_key)

    _agent = initialize_agent(
        [DriveSearchTool(), SearchFolderContentsTool()],
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=_memory,
        verbose=False,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": _build_system_prompt()},
    )
    return _agent


def _looks_like_search_query(message: str) -> bool:
    words = set(re.findall(r"[a-zA-Z0-9_-]+", message.lower()))
    if len(words) == 1:
        return True
    return bool(words & SEARCH_TERMS)


def _dedupe_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for file in files:
        file_id = file.get("id")
        if not file_id or file_id in seen:
            continue
        seen.add(file_id)
        deduped.append(file)
    return deduped


def _files_from_observation(observation: Any) -> list[dict[str, Any]]:
    if observation is None:
        return []

    payload = observation
    if isinstance(observation, str):
        try:
            payload = json.loads(observation)
        except json.JSONDecodeError:
            logger.warning("Tool returned non-JSON observation: %r", observation[:300])
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
        logger.warning("intermediate_steps is not a list: %s", type(steps).__name__)
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

    return _dedupe_files(files)


def chat_with_agent(message: str):
    """Search Drive deterministically first; use the LLM only as a fallback/enhancer."""
    is_search_query = _looks_like_search_query(message)

    if is_search_query:
        try:
            files = staged_search(message)
            logger.debug("Direct search completed count=%s message=%r", len(files), message)
            if files:
                return f"I found {len(files)} matching items.", files, True
        except DriveRateLimitError:
            logger.exception("Direct Drive search hit rate limit")
            return "Google Drive is rate-limiting searches right now. Please try again shortly.", [], True
        except Exception:
            logger.exception("Direct Drive search failed message=%r", message)

    try:
        agent = _get_agent()
        result = agent({"input": message})
        text = result.get("output", "")
        steps = result.get("intermediate_steps", [])

        files = _extract_files_from_intermediate_steps(steps)
        tool_used = bool(steps)

        if is_search_query and not tool_used:
            logger.warning("Agent skipped tool usage for search query=%r", message)
            tool_output = DriveSearchTool()._run(message)
            files = _files_from_observation(tool_output)
            if files:
                return f"I found {len(files)} matching items.", _dedupe_files(files), True

        if is_search_query and not files:
            return f"I couldn't find any files matching '{message}'.", [], True

        return text or "Done.", files, tool_used
        
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "rate limit" in err_str:
            logger.exception("LLM rate limit; falling back to direct search")
            if is_search_query:
                try:
                    files = staged_search(message)
                    if files:
                        return f"The AI service is busy, but I found {len(files)} items with direct search.", files, True
                except Exception:
                    logger.exception("Direct fallback search also failed")
            return "The AI service is busy right now. Please try again shortly.", [], False
            
        logger.exception("Agent failure message=%r", message)
        return "I encountered a problem processing your request. Please try a simpler search.", [], False
