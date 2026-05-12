import datetime
import json
import os
from difflib import SequenceMatcher
from typing import Type

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool

from backend.google_drive import search_drive

# ---------------------------------------------------------------------------
# Constants & Helpers
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "all", "any", "containing", "contains", "contents", "directory", 
    "file", "files", "find", "folder", "folders", "for", "from", "get", "give", 
    "in", "inside", "list", "locate", "me", "my", "named", "of", "only", "open", 
    "please", "related", "search", "show", "that", "the", "with", "within", "called",
}

MIME_INTENTS = {
    "document": "application/vnd.google-apps.document",
    "documents": "application/vnd.google-apps.document",
    "doc": "application/vnd.google-apps.document",
    "docs": "application/vnd.google-apps.document",
    "spreadsheet": "application/vnd.google-apps.spreadsheet",
    "spreadsheets": "application/vnd.google-apps.spreadsheet",
    "sheet": "application/vnd.google-apps.spreadsheet",
    "sheets": "application/vnd.google-apps.spreadsheet",
    "excel": "application/vnd.google-apps.spreadsheet",
    "folder": "application/vnd.google-apps.folder",
    "folders": "application/vnd.google-apps.folder",
    "directory": "application/vnd.google-apps.folder",
    "pdf": "application/pdf",
    "pdfs": "application/pdf",
    "image": "image/",
    "images": "image/",
    "photo": "image/",
    "photos": "image/",
    "pic": "image/",
    "pics": "image/",
    "video": "video/",
    "videos": "video/",
}


def _clean_query(text: str) -> list:
    """Normalize, remove filler words, and return meaningful keywords."""
    words = text.lower().replace("'", "").replace('"', "").replace(",", " ").split()
    # Filter out stopwords AND words that match our MIME_INTENTS (they are handled by _detect_intent)
    return [w for w in words if w not in _STOPWORDS and w not in MIME_INTENTS and len(w) > 1]


def _detect_intent(text: str) -> dict:
    """Detect mimeType and date filters from raw text."""
    intent = {"mimeType": None, "modifiedTime": None}
    
    words = text.lower().replace("'", "").replace('"', "").replace(",", " ").split()
    now = datetime.datetime.utcnow()
    
    # Date mapping
    date_map = {
        "today": now.strftime("%Y-%m-%d"),
        "yesterday": (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
        "recent": (now - datetime.timedelta(days=7)).strftime("%Y-%m-%d"),
        "week": (now - datetime.timedelta(days=7)).strftime("%Y-%m-%d"),
        "month": (now - datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
    }

    for word in words:
        # Check dates
        for k, v in date_map.items():
            if k in word:
                intent["modifiedTime"] = f"{v}T00:00:00Z"
                break
        
        # Check MIME
        for k, v in MIME_INTENTS.items():
            if k == word or k + "s" == word or word.startswith(k):
                intent["mimeType"] = v
                break

    return intent


def _similarity(a: str, b: str) -> int:
    return int(SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100)


def staged_search(raw_query: str, folder_id: str = None) -> list:
    """
    Intelligent 6-stage search pipeline.
    """
    root = folder_id # Scoped if provided
    keywords = _clean_query(raw_query)
    intent = _detect_intent(raw_query)
    
    # Debug Logging
    print(f"\n[search_debug] Raw Input: '{raw_query}'", flush=True)
    print(f"[search_debug] Extracted Keywords: {keywords}", flush=True)
    print(f"[search_debug] Detected Intent: MIME={intent['mimeType']}, Date={intent['modifiedTime']}", flush=True)
    print(f"[search_debug] Scope: {'Folder=' + str(root) if root else 'Global'}", flush=True)

    def build_q(name_part=None, full_text_part=None):
        parts = []
        if name_part:
            parts.append(name_part)
        if full_text_part:
            parts.append(full_text_part)
        
        if intent["mimeType"]:
            if "/" in intent["mimeType"] and not intent["mimeType"].startswith("application"):
                parts.append(f"mimeType contains '{intent['mimeType']}'")
            else:
                parts.append(f"mimeType = '{intent['mimeType']}'")
        
        if intent["modifiedTime"]:
            parts.append(f"modifiedTime > '{intent['modifiedTime']}'")
            
        return " and ".join(parts) if parts else "trashed = false"

    # Stage 1: Exact Name Match (with keywords)
    if keywords:
        name_q = " and ".join([f"name contains '{k}'" for k in keywords])
        q = build_q(name_part=name_q)
        print(f"[search_stage] 1. Precise Name: {q}", flush=True)
        results = search_drive(q, folder_id=root)
        if results: 
            print(f"[search_result] Found {len(results)} items in Stage 1", flush=True)
            return results

    # Stage 2: Intent Only (e.g. "show images")
    # If no keywords remain (or even if they do), try an intent-only global search
    if intent["mimeType"] or intent["modifiedTime"]:
        q = build_q()
        print(f"[search_stage] 2. Intent Only: {q}", flush=True)
        # For general intent queries like "show images", we search globally (root=None) 
        # unless specifically asked for a folder.
        results = search_drive(q, folder_id=root if root else None)
        if results: 
            print(f"[search_result] Found {len(results)} items in Stage 2", flush=True)
            return results

    # Stage 3: Partial Token Match (relaxed)
    if keywords and len(keywords) > 1:
        for kw in keywords:
            q = build_q(name_part=f"name contains '{kw}'")
            print(f"[search_stage] 3. Token '{kw}': {q}", flush=True)
            results = search_drive(q, folder_id=root)
            if results: 
                print(f"[search_result] Found {len(results)} items in Stage 3", flush=True)
                return results

    # Stage 4: FullText Search
    if keywords:
        ft_q = " and ".join([f"fullText contains '{k}'" for k in keywords])
        q = build_q(full_text_part=ft_q)
        print(f"[search_stage] 4. FullText: {q}", flush=True)
        results = search_drive(q, folder_id=root)
        if results: 
            print(f"[search_result] Found {len(results)} items in Stage 4", flush=True)
            return results

    # Stage 5: Broad Fuzzy Fallback
    print(f"[search_stage] 5. Fuzzy Fallback", flush=True)
    extra_q = build_q()
    pool = search_drive(extra_q, folder_id=root, page_size=50)
    if pool and keywords:
        search_target = " ".join(keywords)
        scored = sorted(
            [(_similarity(search_target, f["name"]), f) for f in pool],
            key=lambda x: x[0],
            reverse=True,
        )
        matches = [f for score, f in scored if score >= 40]
        if matches:
            print(f"[search_result] Found {len(matches)} items in Stage 5", flush=True)
            return matches

    print(f"[search_result] No files found in any stage.", flush=True)
    return []


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class DriveSearchInput(BaseModel):
    q_parameter: str = Field(
        description="Natural language search query or keywords."
    )

class DriveSearchTool(BaseTool):
    name = "google_drive_search"
    description = (
        "Search files in Google Drive. Handles natural language like 'find invoice pdf' "
        "or 'show images'. Returns list of matching files."
    )
    args_schema: Type[BaseModel] = DriveSearchInput

    def _run(self, q_parameter: str) -> str:
        try:
            # We don't scope by default unless the agent chooses to
            results = staged_search(q_parameter)
            if not results:
                return json.dumps({"error": f"No files found for '{q_parameter}'.", "files": []})
            return json.dumps({"files": results})
        except Exception as e:
            return json.dumps({"error": str(e), "files": []})

class FolderSearchInput(BaseModel):
    folder_name: str = Field(
        description="Name of the folder to look inside."
    )

class SearchFolderContentsTool(BaseTool):
    name = "search_folder_contents"
    description = "List files inside a specific folder by name."
    args_schema: Type[BaseModel] = FolderSearchInput

    def _run(self, folder_name: str) -> str:
        try:
            # 1. Find the folder
            folders = staged_search(f"{folder_name} folder")
            if not folders:
                return json.dumps({"error": f"Folder '{folder_name}' not found.", "files": []})

            # 2. List contents of the first matching folder
            folder = folders[0]
            files = search_drive(query=f"'{folder['id']}' in parents", folder_id=None)
            
            if not files:
                return json.dumps({"error": f"Folder '{folder['name']}' is empty.", "files": []})

            return json.dumps({"files": files})
        except Exception as e:
            return json.dumps({"error": str(e), "files": []})


class DiagnosticInput(BaseModel):
    pass

class DiagnosticTool(BaseTool):
    name = "drive_diagnostics"
    description = "Check which files the bot can see. Use this when searches return 0 results."
    args_schema: Type[BaseModel] = DiagnosticInput

    def _run(self, *args, **kwargs) -> str:
        try:
            from backend.google_drive import search_drive
            files = search_drive("trashed = false", page_size=10)
            names = [f["name"] for f in files]
            return json.dumps({
                "bot_can_see_count": len(files),
                "sample_files": names,
                "message": "If count is 0, the service account has NO permissions. Please share your Drive folder with the client_email."
            })
        except Exception as e:
            return json.dumps({"error": str(e)})
