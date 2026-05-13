import logging
import os
import time
from pathlib import Path
from typing import Any

import asyncio
import requests as http
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.agent import chat_with_agent, preload_agent
from backend.google_drive import DriveConfigurationError
from backend.google_drive import execute_drive_request, get_drive_service

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
CHAT_TIMEOUT_SECONDS = float(os.getenv("CHAT_TIMEOUT_SECONDS", "45"))
FRIENDLY_SERVER_ERROR = "Server temporarily unavailable. Please try again."

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
    error: str | None = None


def _drive_env_configured() -> bool:
    return bool(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        or os.getenv("GOOGLE_SERVICE_ACCOUNT_INFO")
        or os.getenv("GOOGLE_CREDENTIALS_JSON")
        or os.getenv("GOOGLE_SERVICE_ACCOUNT_BASE64")
        or os.getenv("GOOGLE_DRIVE_TOKEN")
        or Path("/app/credentials.json").exists()
        or (Path.cwd() / "credentials.json").exists()
        or (Path.cwd().parent / "credentials.json").exists()
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.monotonic()
    logger.info("Incoming request method=%s path=%s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Request failed method=%s path=%s", request.method, request.url.path)
        raise
    elapsed_ms = int((time.monotonic() - started) * 1000)
    logger.info(
        "Request complete method=%s path=%s status=%s duration_ms=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.on_event("startup")
async def startup_checks():
    started = time.monotonic()
    logger.info("Backend startup initialization begin")
    logger.info("Chat timeout seconds=%s", CHAT_TIMEOUT_SECONDS)
    logger.info("GROQ_API_KEY configured=%s", bool(os.getenv("GROQ_API_KEY")))
    logger.info("Google Drive env configured=%s", _drive_env_configured())
    logger.info("Vector DB configured=%s", bool(os.getenv("VECTOR_DB_PATH") or os.getenv("CHROMA_DB_PATH")))

    if _drive_env_configured() and os.getenv("PRELOAD_DRIVE_ON_STARTUP", "true").lower() == "true":
        drive_started = time.monotonic()
        try:
            await run_in_threadpool(get_drive_service)
            logger.info(
                "Google Drive service preloaded successfully duration_ms=%s",
                int((time.monotonic() - drive_started) * 1000),
            )
        except Exception:
            logger.exception("Google Drive preload failed")
    else:
        logger.warning("Google Drive preload skipped because Drive credentials are not configured")

    if os.getenv("GROQ_API_KEY") and os.getenv("PRELOAD_AGENT_ON_STARTUP", "true").lower() == "true":
        agent_started = time.monotonic()
        try:
            await asyncio.wait_for(
                run_in_threadpool(preload_agent),
                timeout=float(os.getenv("STARTUP_AGENT_TIMEOUT_SECONDS", "35")),
            )
            logger.info(
                "LangChain agent/model/tools preloaded duration_ms=%s",
                int((time.monotonic() - agent_started) * 1000),
            )
        except Exception:
            logger.exception("LangChain agent/model/tools preload failed")
    else:
        logger.warning("LangChain agent preload skipped because GROQ_API_KEY is not configured")

    logger.info("Backend startup initialization complete duration_ms=%s", int((time.monotonic() - started) * 1000))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    started = time.monotonic()
    message = (req.message or "").strip()
    logger.info("Chat request received length=%s query=%r", len(message), message[:500])
    try:
        text, files, tool_used = await asyncio.wait_for(
            run_in_threadpool(chat_with_agent, message),
            timeout=CHAT_TIMEOUT_SECONDS,
        )
        logger.info(
            "Chat request complete files=%s tool_used=%s response_length=%s duration_ms=%s",
            len(files),
            tool_used,
            len(text or ""),
            int((time.monotonic() - started) * 1000),
        )
        return ChatResponse(response=text, files=files, tool_used=tool_used)
    except asyncio.TimeoutError:
        logger.exception("Chat request timed out after %ss query=%r", CHAT_TIMEOUT_SECONDS, message[:500])
        return JSONResponse(
            status_code=504,
            content=ChatResponse(
                response=FRIENDLY_SERVER_ERROR,
                files=[],
                tool_used=False,
                error="chat_timeout",
            ).model_dump(),
        )
    except Exception:
        logger.exception("Unhandled /chat failure query=%r", message[:500])
        return JSONResponse(
            status_code=500,
            content=ChatResponse(
                response=FRIENDLY_SERVER_ERROR,
                files=[],
                tool_used=False,
                error="chat_failed",
            ).model_dump(),
        )


def _fetch_thumbnail_response(file_id: str) -> Response:
    try:
        service = get_drive_service()
        meta = execute_drive_request(
            service.files().get(
                fileId=file_id,
                fields="thumbnailLink",
                supportsAllDrives=True,
            ),
            description="thumbnail files.get",
        )

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
    except Exception:
        logger.exception("Thumbnail service error file_id=%s", file_id)
        raise HTTPException(status_code=500, detail="Thumbnail service error")


@app.get("/thumbnail/{file_id}")
async def thumbnail(file_id: str):
    return await run_in_threadpool(_fetch_thumbnail_response, file_id)


@app.get("/health")
async def health():
    return {"status": "ok"}
