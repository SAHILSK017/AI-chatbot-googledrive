import json
import logging
import os
from functools import lru_cache
from typing import Any

from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

DRIVE_FIELDS = (
    "files(id, name, mimeType, webViewLink, modifiedTime, size, thumbnailLink)"
)
DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"


class DriveRateLimitError(RuntimeError):
    """Raised when Google Drive returns a quota or rate-limit response."""


def _normalize_file(file: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": file.get("id", ""),
        "name": file.get("name", ""),
        "mimeType": file.get("mimeType", ""),
        "link": file.get("webViewLink", ""),
        "webViewLink": file.get("webViewLink", ""),
        "modified": file.get("modifiedTime", ""),
        "modifiedTime": file.get("modifiedTime", ""),
        "size": file.get("size"),
        "thumbnail": file.get("thumbnailLink"),
        "thumbnailLink": file.get("thumbnailLink"),
    }


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


def _load_credentials():
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.exists(credentials_path):
        return service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=[DRIVE_READONLY_SCOPE],
        )

    token_json = os.getenv("GOOGLE_DRIVE_TOKEN")
    if token_json:
        try:
            token_data = json.loads(token_json)
        except json.JSONDecodeError as exc:
            logger.exception("Invalid GOOGLE_DRIVE_TOKEN JSON")
            raise ValueError("GOOGLE_DRIVE_TOKEN contains invalid JSON") from exc

        return Credentials(
            token=token_data.get("access_token"),
            refresh_token=token_data.get("refresh_token"),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=token_data.get("client_id"),
            client_secret=token_data.get("client_secret"),
            scopes=[DRIVE_READONLY_SCOPE],
        )

    raise ValueError(
        "Google Drive credentials missing. Set GOOGLE_APPLICATION_CREDENTIALS "
        "or GOOGLE_DRIVE_TOKEN."
    )


@lru_cache(maxsize=1)
def get_drive_service():
    """Return a cached authenticated Google Drive v3 service."""
    credentials = _load_credentials()
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _with_default_filters(query: str | None, folder_id: str | None) -> str:
    clauses: list[str] = []
    cleaned_query = (query or "").strip()

    if cleaned_query:
        clauses.append(cleaned_query)
    if folder_id:
        clauses.append(f"'{folder_id}' in parents")
    if "trashed" not in cleaned_query.lower():
        clauses.append("trashed=false")

    return " and ".join(clauses) if clauses else "trashed=false"


def search_drive(
    query: str | None = None,
    folder_id: str | None = None,
    page_size: int = 30,
) -> list[dict[str, Any]]:
    """Execute a Drive files.list query and return normalized, deduped files."""
    q = _with_default_filters(query, folder_id)
    logger.debug("Drive search query=%r page_size=%s", q, page_size)

    try:
        response = (
            get_drive_service()
            .files()
            .list(
                q=q,
                pageSize=page_size,
                fields=DRIVE_FIELDS,
                orderBy="modifiedTime desc",
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )
    except HttpError as exc:
        status = getattr(exc.resp, "status", None)
        logger.exception("Google Drive API error status=%s query=%r", status, q)
        if status == 429:
            raise DriveRateLimitError("Google Drive rate limit exceeded") from exc
        raise
    except Exception:
        logger.exception("Google Drive search failed query=%r", q)
        raise

    files = [_normalize_file(file) for file in response.get("files", [])]
    deduped = _dedupe_files(files)
    logger.debug("Drive search returned %s files", len(deduped))
    return deduped


def _search_drive(q: str, page_size: int = 30) -> list[dict[str, Any]]:
    """Backward-compatible wrapper for older imports."""
    return search_drive(query=q, page_size=page_size)
