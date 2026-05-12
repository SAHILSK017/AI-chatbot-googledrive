import os
import logging
from typing import Any

import requests as http
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from backend.agent import chat_with_agent
from backend.google_drive import DriveConfigurationError
from backend.google_drive import get_drive_service

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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
    files: list[dict[str, Any]] = Field(default_factory=list)
    tool_used: bool = False


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        text, files, tool_used = chat_with_agent(req.message)
        return ChatResponse(response=text, files=files, tool_used=tool_used)
    except Exception:
        logger.exception("Unhandled /chat failure")
        return ChatResponse(
            response="I couldn't complete that request. Please try again.",
            files=[],
            tool_used=False,
        )


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
    except DriveConfigurationError:
        logger.exception("Thumbnail requested before Drive was configured")
        raise HTTPException(status_code=503, detail="Google Drive is not configured")
    except Exception as e:
        logger.exception("Thumbnail service error file_id=%s", file_id)
        raise HTTPException(status_code=500, detail="Thumbnail service error")


@app.get("/health")
async def health():
    return {"status": "ok"}
