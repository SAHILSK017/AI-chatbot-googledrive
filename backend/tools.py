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
    "please", "related", "search", "show", "that", "the", "with", "within",
    "document", "documents", "image", "images", "called",
}

MIME_INTENTS = {
    "document": "application/vnd.google-apps.document",
    "doc": "application/vnd.google-apps.document",
    "google doc": "application/vnd.google-apps.document",
    "spreadsheet": "application/vnd.google-apps.spreadsheet",
    "sheet": "application/vnd.google-apps.spreadsheet",
    "google sheet": "application/vnd.google-apps.spreadsheet",
    "folder": "application/vnd.google-apps.folder",
    "directory": "application/vnd.google-apps.folder",
    "pdf": "application/pdf",
    "image": "image/",
    "photo": "image/",
    "pic": "image/",
    "video": "video/",
}


def _clean_query(text: str) -> list:
    """Normalize, remove filler words, and return meaningful keywords."""
    words = text.lower().replace("'", "").replace('"', "").split()
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _detect_intent(keywords: list) -> dict:
    """Detect mimeType and date filters from keywords."""
    intent = {"mimeType": None, "modifiedTime": None, "clean_keywords": []}
    
    now = datetime.datetime.utcnow()
    
    # Date mapping
    date_map = {
        "today": now.strftime("%Y-%m-%d"),
        "yesterday": (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
        "recent": (now - datetime.timedelta(days=7)).strftime("%Y-%m-%d"),
        "week": (now - datetime.timedelta(days=7)).strftime("%Y-%m-%d"),
        "month": (now - datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
    }

    temp_keywords = []
    for word in keywords:
        found_date = False
        for k, v in date_map.items():
            if k in word:
                intent["modifiedTime"] = f"{v}T00:00:00Z"
                found_date = True
                break
        
        if found_date:
            continue

        found_mime = False
        for k, v in MIME_INTENTS.items():
            if k in word:
                intent["mimeType"] = v
                found_mime = True
                break
        
        if not found_mime:
            temp_keywords.append(word)

    intent["clean_keywords"] = temp_keywords
    return intent


def _similarity(a: str, b: str) -> int:
    return int(SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100)


def staged_search(raw_query: str, folder_id: str = None) -> list:
    """
    Intelligent 6-stage search pipeline.
    1. Exact Name
    2. MIME Intent only
    3. Partial Token Match
    4. FullText Search
    5. Fuzzy Fallback
    """
    root = folder_id or os.getenv("TARGET_FOLDER_ID")
    keywords = _clean_query(raw_query)
    intent = _detect_intent(keywords)
    clean_kws = intent["clean_keywords"]
    
    # Debug Logging (Step 10)
    print(f"\n[search_debug] Raw: '{raw_query}'", flush=True)
    print(f"[search_debug] Keywords: {clean_kws}", flush=True)
    print(f"[search_debug] Intent: MIME={intent['mimeType']}, Date={intent['modifiedTime']}", flush=True)

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
        return " and ".join(parts) if parts else ""

    # Stage 1: Exact Name Match
    if clean_kws:
        name_q = " and ".join([f"name contains '{k}'" for k in clean_kws])
        q = build_q(name_part=name_q)
        print(f"[search_stage] 1. Precise Name: {q}", flush=True)
        results = search_drive(q, folder_id=root)
        if results: return results

    # Stage 2: Intent Match (e.g. "show images", "find documents")
    # Runs when MIME/date intent was detected but no other keywords remain
    if (not clean_kws) and (intent["mimeType"] or intent["modifiedTime"]):
        q = build_q()
        print(f"[search_stage] 2. Intent Only: {q}", flush=True)
        results = search_drive(q, folder_id=None)  # Always global for intent-only
        if results: return results

    # Stage 3: Partial Token Match (relaxed)
    if clean_kws:
        # Try matching ANY keyword in name if multi-keyword failed
        if len(clean_kws) > 1:
            for kw in clean_kws:
                q = build_q(name_part=f"name contains '{kw}'")
                print(f"[search_stage] 3. Token '{kw}': {q}", flush=True)
                results = search_drive(q, folder_id=root)
                if results: return results

    # Stage 4: FullText Search (Inside content)
    if clean_kws:
        ft_q = " and ".join([f"fullText contains '{k}'" for k in clean_kws])
        q = build_q(full_text_part=ft_q)
        print(f"[search_stage] 4. FullText: {q}", flush=True)
        results = search_drive(q, folder_id=root)
        if results: return results

    # Stage 5: Broad FullText (any token)
    if len(clean_kws) > 1:
        for kw in clean_kws:
            q = build_q(full_text_part=f"fullText contains '{kw}'")
            print(f"[search_stage] 5. Broad FullText '{kw}': {q}", flush=True)
            results = search_drive(q, folder_id=root)
            if results: return results

    # Stage 6: Fuzzy Fallback
    print(f"[search_stage] 6. Fuzzy Fallback", flush=True)
    extra_q = build_q()
    pool = search_drive(extra_q, folder_id=root, page_size=50)
    if pool and clean_kws:
        search_target = " ".join(clean_kws)
        scored = sorted(
            [(_similarity(search_target, f["name"]), f) for f in pool],
            key=lambda x: x[0],
            reverse=True,
        )
        matches = [f for score, f in scored if score >= 45] # Lower threshold for fuzzy
        return matches

    return []


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class DriveSearchInput(BaseModel):
    q_parameter: str = Field(
        description="Natural language search query or Drive API 'q' string."
    )


class DriveSearchTool(BaseTool):
    name = "google_drive_search"
    description = (
        "Search files in Google Drive. You can provide natural language like "
        "'find invoice pdf' or 'show images modified today'."
    )
    args_schema: Type[BaseModel] = DriveSearchInput

    def _run(self, q_parameter: str) -> str:
        root = os.getenv("TARGET_FOLDER_ID")
        try:
            # Check if it's already a valid q-param or a natural sentence
            is_q_param = "=" in q_parameter or "contains" in q_parameter
            
            if is_q_param:
                results = search_drive(query=q_parameter, folder_id=root)
                if results:
                    return json.dumps({"files": results})

            # If empty or natural language, use staged search
            # For intent-only queries (images/sheets/etc), always search globally
            results = staged_search(q_parameter, folder_id=None)
            
            if not results:
                return json.dumps({"error": f"No files found for '{q_parameter}'.", "files": []})
            
            print(f"[search_result] Returned {len(results)} items", flush=True)
            return json.dumps({"files": results})
        except Exception as e:
            return json.dumps({"error": str(e), "files": []})

    def _arun(self, q_parameter: str):
        raise NotImplementedError


class FolderSearchInput(BaseModel):
    folder_name: str = Field(
        description="Name of the folder to look inside. E.g. 'invoices' or 'pics'."
    )


class SearchFolderContentsTool(BaseTool):
    name = "search_folder_contents"
    description = "List files inside a specific folder by name."
    args_schema: Type[BaseModel] = FolderSearchInput

    def _run(self, folder_name: str) -> str:
        try:
            root = os.getenv("TARGET_FOLDER_ID")
            # Force folder mime intent
            folders = staged_search(f"{folder_name} folder", folder_id=root)
            
            if not folders:
                return json.dumps({"error": f"Folder '{folder_name}' not found.", "files": []})

            folder = folders[0]
            files = search_drive(query=f"'{folder['id']}' in parents", folder_id=None)
            
            if not files:
                return json.dumps({"error": f"Folder '{folder['name']}' is empty.", "files": []})

            return json.dumps({"files": files})
        except Exception as e:
            return json.dumps({"error": str(e), "files": []})

    def _arun(self, folder_name: str):
        raise NotImplementedError
