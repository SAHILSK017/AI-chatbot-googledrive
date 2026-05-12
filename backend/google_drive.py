import functools
import os
import time

from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")

_query_cache: dict = {}
_folder_tree_cache: dict = {}
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


def get_all_folder_ids(root_id: str) -> list:
    """Recursively fetch all subfolder IDs under root_id, with a 5-min cache."""
    if not root_id:
        return []
        
    cached = _folder_tree_cache.get(root_id)
    if cached is not None:
        result, ts = cached
        if time.time() - ts < _CACHE_TTL:
            return result
        del _folder_tree_cache[root_id]
        
    service = get_drive_service()
    folder_ids = [root_id]
    queue = [root_id]
    
    try:
        while queue:
            current_id = queue.pop(0)
            q = f"'{current_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            res = service.files().list(q=q, fields="files(id)").execute()
            for f in res.get('files', []):
                if len(folder_ids) < 30: # Cap at 30 to avoid Google API query length limits
                    folder_ids.append(f['id'])
                    queue.append(f['id'])
                
        _folder_tree_cache[root_id] = (folder_ids, time.time())
        return folder_ids
    except Exception as e:
        print(f"[drive error] subfolders fetch: {e}", flush=True)
        return [root_id]


def _build_query(query: str, folder_id: str = None) -> str:
    parts = []
    if query:
        parts.append(f"({query})")
    if folder_id:
        all_ids = get_all_folder_ids(folder_id)
        parents_or = " or ".join([f"'{fid}' in parents" for fid in all_ids])
        parts.append(f"({parents_or})")
    parts.append("trashed = false")
    return " and ".join(parts)


def invalidate_cache():
    _query_cache.clear()
    _folder_tree_cache.clear()
