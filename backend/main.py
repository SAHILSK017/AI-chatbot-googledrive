import os
import re
from typing import Any, Dict, List

import requests as http
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from backend.agent import chat_with_agent
from backend.google_drive import get_drive_service
from backend.tools import staged_search

load_dotenv()

app = FastAPI(title="Drive Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    files: List[Dict[str, Any]] = []
    tool_used: bool = False


def _is_simple_query(text: str) -> bool:
    """Determine if a query is simple enough to bypass the LLM."""
    clean = text.lower().strip()
    words = clean.split()
    
    # 1. Single word searches
    if len(words) == 1:
        return True
        
    # 2. Basic "show/find" patterns
    patterns = [
        r"^(show|find|get|search)\s+(only\s+)?(images|pics|pdfs|spreadsheets|sheets|docs|documents|folders|recent|all)$",
        r"^(show|find|get|search)\s+all\s+files$",
        r"^(show|find|get|search)\s+(my\s+)?\w+$", # e.g. "find my invoices"
    ]
    if any(re.match(p, clean) for p in patterns):
        return True
        
    return False


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    msg = req.message
    
    # AGGRESSIVE TOKEN OPTIMIZATION: Bypassing LLM for deterministic queries
    if _is_simple_query(msg):
        print(f"[direct_search] Intercepted simple query: '{msg}'", flush=True)
        files = staged_search(msg)
        if files:
            count = len(files)
            return ChatResponse(
                response=f"I found {count} items matching your search (Direct Mode).",
                files=files,
                tool_used=True
            )
        else:
            return ChatResponse(
                response=f"I couldn't find any files matching '{msg}'.",
                files=[],
                tool_used=True
            )

    # Use LLM for semantic or complex reasoning
    text, files, tool_used = chat_with_agent(msg)
    return ChatResponse(response=text, files=files, tool_used=tool_used)


@app.get("/thumbnail/{file_id}")
async def thumbnail(file_id: str):
    """Proxy Drive thumbnails server-side to bypass browser auth requirements."""
    try:
        service = get_drive_service()
        meta = service.files().get(
            fileId=file_id,
            fields="thumbnailLink",
            supportsAllDrives=True,
        ).execute()

        url = meta.get("thumbnailLink", "")
        if not url:
            raise HTTPException(status_code=404, detail="No thumbnail available.")

        url = url.split("=s")[0] + "=s400"

        creds = service._http.credentials
        token = creds.token
        if not token:
            from google.auth.transport.requests import Request
            creds.refresh(Request(http.Session()))
            token = creds.token

        resp = http.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=8)
        if resp.status_code != 200:
            raise HTTPException(status_code=404, detail="Thumbnail fetch failed.")

        return Response(
            content=resp.content,
            media_type=resp.headers.get("Content-Type", "image/jpeg"),
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[thumbnail] {file_id}: {e}", flush=True)
        raise HTTPException(status_code=500, detail="Thumbnail service error")


@app.get("/health")
async def health():
    return {"status": "ok"}
