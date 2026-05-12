import base64
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
DRIVE_FIELDS = (
    "nextPageToken, files("
    "id, name, mimeType, webViewLink, modifiedTime, size, thumbnailLink, "
    "iconLink, parents, owners(displayName,emailAddress)"
    ")"
)
FOLDER_MIME = "application/vnd.google-apps.folder"


class DriveConfigurationError(RuntimeError):
    """Raised when Drive credentials are not configured."""


class DriveRateLimitError(RuntimeError):
    """Raised when Google Drive returns a quota or rate-limit response."""


def _normalize_file(file: dict[str, Any]) -> dict[str, Any]:
    web_link = file.get("webViewLink", "")
    modified = file.get("modifiedTime", "")
    thumb = file.get("thumbnailLink")
    return {
        "id": file.get("id", ""),
        "name": file.get("name", ""),
        "mimeType": file.get("mimeType", ""),
        "webViewLink": web_link,
        "link": web_link,
        "modifiedTime": modified,
        "modified": modified,
        "size": file.get("size"),
        "thumbnailLink": thumb,
        "thumbnail": thumb,
        "iconLink": file.get("iconLink", ""),
        "parents": file.get("parents", []),
        "owners": file.get("owners", []),
    }


def dedupe_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for file in files:
        file_id = file.get("id")
        if not file_id or file_id in seen:
            continue
        seen.add(file_id)
        deduped.append(file)
    return deduped


def _candidate_credential_paths() -> list[Path]:
    paths: list[Path] = []
    configured = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if configured:
        paths.append(Path(configured))

    paths.extend(
        [
            Path("/app/credentials.json"),
            Path.cwd() / "credentials.json",
            Path.cwd().parent / "credentials.json",
        ]
    )
    return paths


def _service_account_from_json(raw: str):
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DriveConfigurationError("Service account JSON is invalid.") from exc
    return service_account.Credentials.from_service_account_info(
        info,
        scopes=[DRIVE_READONLY_SCOPE],
    )


def _load_credentials():
    for path in _candidate_credential_paths():
        if path.exists():
            logger.info("Using Google Drive credentials file at %s", path)
            return service_account.Credentials.from_service_account_file(
                str(path),
                scopes=[DRIVE_READONLY_SCOPE],
            )

    raw_service_json = (
        os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        or os.getenv("GOOGLE_SERVICE_ACCOUNT_INFO")
        or os.getenv("GOOGLE_CREDENTIALS_JSON")
    )
    if raw_service_json:
        logger.info("Using Google Drive service account JSON from environment")
        return _service_account_from_json(raw_service_json)

    encoded_service_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_BASE64")
    if encoded_service_json:
        try:
            decoded = base64.b64decode(encoded_service_json).decode("utf-8")
        except Exception as exc:
            raise DriveConfigurationError("GOOGLE_SERVICE_ACCOUNT_BASE64 is invalid.") from exc
        logger.info("Using Google Drive service account JSON from base64 environment")
        return _service_account_from_json(decoded)

    token_json = os.getenv("GOOGLE_DRIVE_TOKEN")
    if token_json:
        try:
            token_data = json.loads(token_json)
        except json.JSONDecodeError as exc:
            raise DriveConfigurationError("GOOGLE_DRIVE_TOKEN contains invalid JSON.") from exc
        logger.info("Using Google Drive OAuth token from environment")
        return Credentials(
            token=token_data.get("access_token"),
            refresh_token=token_data.get("refresh_token"),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=token_data.get("client_id"),
            client_secret=token_data.get("client_secret"),
            scopes=[DRIVE_READONLY_SCOPE],
        )

    raise DriveConfigurationError(
        "Google Drive is not configured. On Render, set GOOGLE_SERVICE_ACCOUNT_JSON "
        "to the full service account JSON, or set GOOGLE_APPLICATION_CREDENTIALS to "
        "a mounted credentials file."
    )


@lru_cache(maxsize=1)
def get_drive_service():
    credentials = _load_credentials()
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def drive_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _with_default_filters(query: str | None, folder_id: str | None) -> str:
    clauses: list[str] = []
    cleaned_query = (query or "").strip()
    if cleaned_query:
        clauses.append(cleaned_query)
    if folder_id:
        clauses.append(f"'{drive_quote(folder_id)}' in parents")
    if "trashed" not in cleaned_query.lower():
        clauses.append("trashed=false")
    return " and ".join(clauses) if clauses else "trashed=false"


def search_drive(
    query: str | None = None,
    folder_id: str | None = None,
    page_size: int = 30,
    order_by: str = "modifiedTime desc",
) -> list[dict[str, Any]]:
    """Execute a Drive files.list query and return normalized, deduped files."""
    q = _with_default_filters(query, folder_id)
    logger.debug("Drive search q=%r page_size=%s", q, page_size)

    try:
        service = get_drive_service()
        files: list[dict[str, Any]] = []
        page_token = None
        remaining = max(page_size, 1)

        while remaining > 0:
            batch_size = min(remaining, 100)
            response = (
                service.files()
                .list(
                    q=q,
                    pageSize=batch_size,
                    pageToken=page_token,
                    fields=DRIVE_FIELDS,
                    orderBy=order_by,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                )
                .execute()
            )
            files.extend(_normalize_file(file) for file in response.get("files", []))
            page_token = response.get("nextPageToken")
            remaining = page_size - len(files)
            if not page_token:
                break

    except DriveConfigurationError:
        logger.error("Google Drive credentials are not configured")
        raise
    except HttpError as exc:
        status = getattr(exc.resp, "status", None)
        logger.exception("Google Drive API error status=%s q=%r", status, q)
        if status == 429:
            raise DriveRateLimitError("Google Drive rate limit exceeded") from exc
        raise
    except Exception:
        logger.exception("Google Drive search failed q=%r", q)
        raise

    deduped = dedupe_files(files)
    logger.debug("Drive search returned count=%s", len(deduped))
    return deduped


def list_folder_children(folder_id: str, page_size: int = 100) -> list[dict[str, Any]]:
    return search_drive(query=None, folder_id=folder_id, page_size=page_size)


@lru_cache(maxsize=128)
def collect_descendant_folder_ids(
    root_folder_id: str,
    max_folders: int = 60,
) -> list[str]:
    """Breadth-first traversal used for reliable scoped search."""
    folder_ids = [root_folder_id]
    queue = [root_folder_id]
    seen = {root_folder_id}

    while queue and len(folder_ids) < max_folders:
        current = queue.pop(0)
        children = search_drive(
            query=f"mimeType = '{FOLDER_MIME}'",
            folder_id=current,
            page_size=100,
            order_by="name",
        )
        for child in children:
            child_id = child.get("id")
            if child_id and child_id not in seen:
                seen.add(child_id)
                folder_ids.append(child_id)
                queue.append(child_id)
            if len(folder_ids) >= max_folders:
                break

    logger.debug("Collected descendant folders count=%s root=%s", len(folder_ids), root_folder_id)
    return folder_ids


def _search_drive(q: str, page_size: int = 30) -> list[dict[str, Any]]:
    """Backward-compatible wrapper for older imports."""
    return search_drive(query=q, page_size=page_size)
