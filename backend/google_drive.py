"""
Fixed backend/tools.py
Key fixes:
  1. Image/media intent detection → MIME type filters (no more "pics" returning nothing)
  2. staged_search covers text, folder, and MIME-type paths
  3. Better error surfacing so you can see WHY a search fails
"""

import json
import re
from typing import Any

from langchain.tools import BaseTool

# ── MIME type map ─────────────────────────────────────────────────────────────
MIME_MAP = {
    # Images
    "image": "mimeType contains 'image/'",
    "images": "mimeType contains 'image/'",
    "pics": "mimeType contains 'image/'",
    "pictures": "mimeType contains 'image/'",
    "photos": "mimeType contains 'image/'",
    "jpg": "mimeType='image/jpeg'",
    "jpeg": "mimeType='image/jpeg'",
    "png": "mimeType='image/png'",
    "gif": "mimeType='image/gif'",
    # Docs
    "spreadsheet": "mimeType='application/vnd.google-apps.spreadsheet'",
    "spreadsheets": "mimeType='application/vnd.google-apps.spreadsheet'",
    "sheet": "mimeType='application/vnd.google-apps.spreadsheet'",
    "excel": "mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'",
    "doc": "mimeType='application/vnd.google-apps.document'",
    "docs": "mimeType='application/vnd.google-apps.document'",
    "document": "mimeType='application/vnd.google-apps.document'",
    "pdf": "mimeType='application/pdf'",
    "slide": "mimeType='application/vnd.google-apps.presentation'",
    "slides": "mimeType='application/vnd.google-apps.presentation'",
    "presentation": "mimeType='application/vnd.google-apps.presentation'",
    "folder": "mimeType='application/vnd.google-apps.folder'",
    "folders": "mimeType='application/vnd.google-apps.folder'",
    # Video / Audio
    "video": "mimeType contains 'video/'",
    "videos": "mimeType contains 'video/'",
    "audio": "mimeType contains 'audio/'",
    "mp3": "mimeType='audio/mpeg'",
    "mp4": "mimeType='video/mp4'",
}

# Keywords that signal the user wants ALL files of a type (no name search)
SHOW_ALL_KEYWORDS = {"show", "list", "display", "find", "get", "all", "only"}


def _parse_query(user_message: str) -> dict:
    """
    Returns:
        {
            "name_query": str | None,   # fullText / name search term
            "mime_filter": str | None,  # e.g. "mimeType contains 'image/'"
            "folder_id": str | None,    # if user specifies a folder
        }
    """
    msg = user_message.lower()
    words = set(re.findall(r"\w+", msg))

    # Detect MIME intent
    mime_filter = None
    for keyword, mime in MIME_MAP.items():
        if keyword in words:
            mime_filter = mime
            break

    # If the message is a "show only X" style, skip name search
    is_show_all = bool(words & SHOW_ALL_KEYWORDS) and mime_filter is not None
    name_query = None if is_show_all else _extract_name_query(user_message, mime_filter)

    return {
        "name_query": name_query,
        "mime_filter": mime_filter,
        "folder_id": None,
    }


def _extract_name_query(message: str, mime_filter) -> str | None:
    """Strip known type words and stop words; return remainder as name query."""
    stop = SHOW_ALL_KEYWORDS | set(MIME_MAP.keys()) | {
        "file", "files", "in", "my", "google", "drive", "the", "a", "an",
        "for", "me", "please", "can", "you", "could", "find", "search",
    }
    tokens = re.findall(r"\w+", message.lower())
    meaningful = [t for t in tokens if t not in stop and len(t) > 1]
    return " ".join(meaningful) if meaningful else None


def _build_drive_query(parsed: dict) -> str:
    """Build a Drive API q= string from parsed intent."""
    clauses = ["trashed=false"]

    if parsed["mime_filter"]:
        clauses.append(parsed["mime_filter"])

    if parsed["name_query"]:
        # Search both name and full text
        name = parsed["name_query"].replace("'", "\\'")
        clauses.append(f"(name contains '{name}' or fullText contains '{name}')")

    return " and ".join(clauses)


# ── Google Drive API helpers ──────────────────────────────────────────────────

def _get_drive_service():
    """Return an authenticated Drive v3 service. Adjust auth to your setup."""
    # Option A: Service account
    # from google.oauth2 import service_account
    # from googleapiclient.discovery import build
    # creds = service_account.Credentials.from_service_account_file(
    #     "service_account.json",
    #     scopes=["https://www.googleapis.com/auth/drive.readonly"]
    # )
    # return build("drive", "v3", credentials=creds)

    # Option B: OAuth2 token from env
    import os
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    token = os.getenv("GOOGLE_DRIVE_TOKEN")
    if not token:
        raise ValueError("GOOGLE_DRIVE_TOKEN env var not set")

    import json as _json
    creds_data = _json.loads(token)
    creds = Credentials(
        token=creds_data.get("access_token"),
        refresh_token=creds_data.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=creds_data.get("client_id"),
        client_secret=creds_data.get("client_secret"),
    )
    return build("drive", "v3", credentials=creds)


def _search_drive(q: str, page_size: int = 30) -> list[dict]:
    """Execute a Drive files.list query and return normalised file dicts."""
    service = _get_drive_service()
    results = service.files().list(
        q=q,
        pageSize=page_size,
        fields="files(id, name, mimeType, webViewLink, modifiedTime, size, thumbnailLink)",
        orderBy="modifiedTime desc",
    ).execute()

    files = results.get("files", [])
    return [
        {
            "id": f["id"],
            "name": f["name"],
            "mimeType": f.get("mimeType", ""),
            "link": f.get("webViewLink", ""),
            "modified": f.get("modifiedTime", ""),
            "size": f.get("size"),
            "thumbnail": f.get("thumbnailLink"),
        }
        for f in files
    ]


# ── LangChain Tools ───────────────────────────────────────────────────────────

class DriveSearchTool(BaseTool):
    name: str = "google_drive_search"
    description: str = (
        "Search Google Drive for files. "
        "Input: natural language query like 'find my invoices', 'show image files', 'pics'. "
        "Handles MIME type filtering automatically (images, PDFs, sheets, etc.)."
    )

    def _run(self, query: str) -> str:
        try:
            parsed = _parse_query(query)
            q = _build_drive_query(parsed)

            # Debug: uncomment to see the Drive query
            # print(f"[DriveSearchTool] q={q!r}", flush=True)

            files = _search_drive(q)
            if not files:
                hint = f" (Drive query used: {q})" if not files else ""
                return json.dumps({
                    "files": [],
                    "message": f"No files found.{hint}",
                    "query_used": q,
                })

            return json.dumps({"files": files, "count": len(files)})

        except Exception as e:
            return json.dumps({"files": [], "error": str(e)})

    async def _arun(self, query: str) -> str:
        return self._run(query)


class SearchFolderContentsTool(BaseTool):
    name: str = "search_folder_contents"
    description: str = (
        "List contents of a specific Google Drive folder. "
        "Input: folder ID or folder name."
    )

    def _run(self, folder_id_or_name: str) -> str:
        try:
            service = _get_drive_service()

            # If not a Drive ID, look up by name first
            folder_id = folder_id_or_name
            if not re.match(r"^[a-zA-Z0-9_-]{20,}$", folder_id_or_name):
                name = folder_id_or_name.replace("'", "\\'")
                res = service.files().list(
                    q=f"mimeType='application/vnd.google-apps.folder' and name contains '{name}' and trashed=false",
                    pageSize=5,
                    fields="files(id, name)",
                ).execute()
                folders = res.get("files", [])
                if not folders:
                    return json.dumps({"files": [], "error": f"Folder '{folder_id_or_name}' not found."})
                folder_id = folders[0]["id"]

            files = _search_drive(f"'{folder_id}' in parents and trashed=false")
            return json.dumps({"files": files, "count": len(files)})

        except Exception as e:
            return json.dumps({"files": [], "error": str(e)})

    async def _arun(self, folder_id_or_name: str) -> str:
        return self._run(folder_id_or_name)


# ── staged_search fallback (no LLM) ──────────────────────────────────────────

def staged_search(message: str) -> list[dict]:
    """
    Fallback: direct Drive search without the LLM agent.
    Called when API rate limits hit.
    """
    try:
        parsed = _parse_query(message)
        q = _build_drive_query(parsed)
        return _search_drive(q, page_size=20)
    except Exception as e:
        print(f"[staged_search] error: {e}", flush=True)
        return []