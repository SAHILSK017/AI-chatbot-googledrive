import functools
import os
import time

from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")

_query_cache: dict = {}
_CACHE_TTL = 300  # seconds


@functools.lru_cache(maxsize=1)
def get_drive_service():
    """Load credentials from env var (production) or file (local dev)."""
    import json
    json_creds = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if json_creds:
        info = json.loads(json_creds)
        creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        print(f"[drive] Authenticated as: {info.get('client_email', 'unknown')}", flush=True)
        return build("drive", "v3", credentials=creds)

    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise FileNotFoundError(f"Credentials not found: {SERVICE_ACCOUNT_FILE}")
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)


def search_drive(query: str, folder_id: str = None, page_size: int = 20) -> list:
    cache_key = (query or "", folder_id or "")
    cached = _query_cache.get(cache_key)
    if cached is not None:
        result, ts = cached
        if time.time() - ts < _CACHE_TTL:
            return result
        del _query_cache[cache_key]

    service = get_drive_service()
    final_query = _build_query(query, folder_id)
    print(f"[drive] {final_query}", flush=True)

    try:
        kwargs = {
            "q": final_query,
            "pageSize": page_size,
            "fields": (
                "nextPageToken, files("
                "id, name, mimeType, modifiedTime, createdTime, "
                "webViewLink, webContentLink, iconLink, parents, "
                "size, owners, thumbnailLink, hasThumbnail"
                ")"
            ),
            "spaces": "drive",
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
        }
        # Google Drive API forbids orderBy when fullText is used
        if "fulltext contains" not in final_query.lower():
            kwargs["orderBy"] = "modifiedTime desc"

        data = service.files().list(**kwargs).execute()
        items = data.get("files", [])
        _query_cache[cache_key] = (items, time.time())
        return items
    except Exception as e:
        print(f"[drive error] {e}", flush=True)
        return []


def _build_query(query: str, folder_id: str = None) -> str:
    parts = []
    if query:
        parts.append(f"({query})")
    if folder_id:
        parts.append(f"'{folder_id}' in parents")
    parts.append("trashed = false")
    return " and ".join(parts)


def invalidate_cache():
    _query_cache.clear()
