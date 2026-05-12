import datetime
import json
import os

from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

from backend.tools import DriveSearchTool, SearchFolderContentsTool, staged_search

_agent = None
_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def _build_system_prompt() -> str:
    # Aggressively shortened for token optimization
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    return f"""Drive Assistant (Today: {today}). 
Tools: `google_drive_search` (global), `search_folder_contents` (local).
Rules: Concise responses. Use history for context. Tools extract keywords/intents automatically."""


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
        verbose=False, # Reduced logs
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": _build_system_prompt()},
    )
    return _agent


def chat_with_agent(message: str):
    """Run AI agent with direct search fallback."""
    try:
        agent = _get_agent()
        result = agent({"input": message})
        text = result.get("output", "")
        steps = result.get("intermediate_steps", [])
        
        files = []
        seen = set()
        for _, obs in steps:
            try:
                for f in json.loads(obs).get("files", []):
                    if f.get("id") not in seen:
                        files.append(f)
                        seen.add(f["id"])
            except Exception:
                pass

        return text, files, len(steps) > 0
        
    except Exception as e:
        # CLEAN ERROR HANDLING: Hide raw 429s/keys
        err_str = str(e).lower()
        if "429" in err_str or "rate limit" in err_str:
            print(f"[fallback] API Limit hit. Entering Direct Search Mode.", flush=True)
            files = staged_search(message)
            if files:
                return f"The AI service is temporarily busy. I've switched to Direct Search Mode and found {len(files)} items.", files, True
            return "The AI service is busy and I couldn't find any matching files locally.", [], False
            
        print(f"[error] Agent failure: {e}", flush=True)
        return "I encountered a problem processing your request. Please try a simpler search.", [], False
