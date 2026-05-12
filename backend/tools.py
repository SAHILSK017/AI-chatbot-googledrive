import datetime
import json
import logging
import os
import re
from difflib import SequenceMatcher
from typing import Any, Type

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool

from backend.google_drive import search_drive

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Helpers
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "all", "any", "containing", "contains", "contents",
    "file", "files", "find", "for", "from", "get", "give",
    "in", "inside", "list", "locate", "me", "my", "named", "of", "only", "open",
    "please", "related", "search", "show", "that", "the", "with", "within",
    "called", "just"
}

MIME_INTENTS = {
    "document": [
        "application/vnd.google-apps.document",
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ],
    "doc": [
        "application/vnd.google-apps.document",
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ],
    "spreadsheet": ["application/vnd.google-apps.spreadsheet"],
    "sheet": ["application/vnd.google-apps.spreadsheet"],
    "folder": ["application/vnd.google-apps.folder"],
    "directory": ["application/vnd.google-apps.folder"],
    "pdf": ["application/pdf"],
    "image": ["image/"],
    "photo": ["image/"],
    "pic": ["image/"],
    "png": ["image/png"],
    "jpg": ["image/jpeg"],
    "jpeg": ["image/jpeg"],
    "video": ["video/"],
}

QUERY_ALIASES = {
    "pics": "pic",
    "pictures": "pic",
    "photos": "photo",
    "images": "image",
    "docs": "doc",
    "documents": "document",
}


def _clean_query(text: str) -> list:
    """Normalize, remove filler words, and return meaningful keywords."""
    words = re.findall(r"[a-zA-Z0-9_-]+", text.lower())
    cleaned = []
    for word in words:
        normalized = QUERY_ALIASES.get(word, word)
        if normalized not in _STOPWORDS and len(normalized) > 1:
            cleaned.append(normalized)
    return cleaned


def _drive_quote(value: str) -> str:
    """Escape a value for use inside a Drive API q string literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


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


def _detect_intent(keywords: list) -> dict:
    """Detect mimeType and date filters from keywords."""
    intent = {"mimeTypes": [], "modifiedTime": None, "clean_keywords": []}
    
    now = datetime.datetime.utcnow()
    
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
        for k, mime_types in MIME_INTENTS.items():
            # Exact match allowing for plurals
            if word == k or word == k + "s":
                for mime_type in mime_types:
                    if mime_type not in intent["mimeTypes"]:
                        intent["mimeTypes"].append(mime_type)
                found_mime = True
                break
        
        # Keep word if it wasn't consumed by intent mapping
        if not found_mime:
            temp_keywords.append(word)

    intent["clean_keywords"] = temp_keywords
    return intent


def _similarity(a: str, b: str) -> int:
    return int(SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100)


def staged_search(raw_query: str, folder_id: str = None) -> list:
    """
    Intelligent 6-stage search pipeline.
    """
    root = folder_id or os.getenv("TARGET_FOLDER_ID")
    keywords = _clean_query(raw_query)
    intent = _detect_intent(keywords)
    clean_kws = intent["clean_keywords"]
    
    logger.debug("Search raw_query=%r keywords=%s intent=%s", raw_query, clean_kws, intent)

    def build_q(name_part=None, full_text_part=None):
        parts = []
        if name_part:
            parts.append(name_part)
        if full_text_part:
            parts.append(full_text_part)
        
        # Robust MIME filtering logic
        if intent["mimeTypes"]:
            mime_parts = []
            for mime_type in intent["mimeTypes"]:
                if mime_type.endswith("/"):
                    mime_parts.append(f"mimeType contains '{mime_type}'")
                else:
                    mime_parts.append(f"mimeType = '{mime_type}'")
            parts.append(f"({' or '.join(mime_parts)})" if len(mime_parts) > 1 else mime_parts[0])
                
        if intent["modifiedTime"]:
            parts.append(f"modifiedTime > '{intent['modifiedTime']}'")
            
        final_q = " and ".join(parts) if parts else ""
        return final_q

    # Stage 1: Exact Name Match
    if clean_kws:
        name_q = " and ".join([f"name contains '{_drive_quote(k)}'" for k in clean_kws])
        q = build_q(name_part=name_q)
        logger.debug("Search stage=precise_name q=%r", q)
        results = search_drive(q, folder_id=root)
        if results: 
            logger.debug("Search stage=precise_name count=%s", len(results))
            return _dedupe_files(results)

    # Stage 2: Intent Match (e.g. "show image files" where clean_kws is [])
    if intent["mimeTypes"] or intent["modifiedTime"]:
        # Execute even if clean_kws is present, as a broader fallback!
        q = build_q()
        logger.debug("Search stage=intent q=%r", q)
        results = search_drive(q, folder_id=root)
        if results: 
            logger.debug("Search stage=intent count=%s", len(results))
            return _dedupe_files(results)

    # Stage 3: Partial Token Match (relaxed)
    if clean_kws and len(clean_kws) > 1:
        for kw in clean_kws:
            q = build_q(name_part=f"name contains '{_drive_quote(kw)}'")
            logger.debug("Search stage=token keyword=%r q=%r", kw, q)
            results = search_drive(q, folder_id=root)
            if results: 
                logger.debug("Search stage=token count=%s", len(results))
                return _dedupe_files(results)

    # Stage 4: FullText Search (Inside content)
    if clean_kws:
        ft_q = " and ".join([f"fullText contains '{_drive_quote(k)}'" for k in clean_kws])
        q = build_q(full_text_part=ft_q)
        logger.debug("Search stage=full_text q=%r", q)
        results = search_drive(q, folder_id=root)
        if results: 
            logger.debug("Search stage=full_text count=%s", len(results))
            return _dedupe_files(results)

    # Stage 5: Broad FullText (any token)
    if clean_kws and len(clean_kws) > 1:
        for kw in clean_kws:
            q = build_q(full_text_part=f"fullText contains '{_drive_quote(kw)}'")
            logger.debug("Search stage=broad_full_text keyword=%r q=%r", kw, q)
            results = search_drive(q, folder_id=root)
            if results: 
                logger.debug("Search stage=broad_full_text count=%s", len(results))
                return _dedupe_files(results)

    # Stage 6: Fuzzy Fallback
    logger.debug("Search stage=fuzzy")
    extra_q = build_q()
    pool = search_drive(extra_q, folder_id=root, page_size=50)
    if pool and clean_kws:
        search_target = " ".join(clean_kws)
        scored = sorted(
            [(_similarity(search_target, f["name"]), f) for f in pool],
            key=lambda x: x[0],
            reverse=True,
        )
        matches = [f for score, f in scored if score >= 45]
        logger.debug("Search stage=fuzzy count=%s", len(matches))
        return _dedupe_files(matches)

    logger.debug("Search returned no results")
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
        "'find invoice pdf' or 'show images'."
    )
    args_schema: Type[BaseModel] = DriveSearchInput

    def _run(self, q_parameter: str) -> str:
        root = os.getenv("TARGET_FOLDER_ID")
        try:
            is_q_param = "=" in q_parameter or "contains" in q_parameter
            
            if is_q_param:
                results = search_drive(query=q_parameter, folder_id=root)
                if results:
                    return json.dumps({"files": results})

            results = staged_search(q_parameter, folder_id=root)
            
            if not results:
                return json.dumps({"error": f"No files found for '{q_parameter}'.", "files": []})
            
            return json.dumps({"files": results})
        except Exception as e:
            logger.exception("google_drive_search tool failed query=%r", q_parameter)
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
            # Directly query for the folder by name and mimeType to avoid intent stripping
            q = f"name contains '{_drive_quote(folder_name)}' and mimeType = 'application/vnd.google-apps.folder'"
            folders = search_drive(q, folder_id=root)
            
            if not folders:
                return json.dumps({"error": f"Folder '{folder_name}' not found.", "files": []})

            folder = folders[0]
            files = search_drive(query=f"'{folder['id']}' in parents", folder_id=None)
            
            if not files:
                return json.dumps({"error": f"Folder '{folder['name']}' is empty.", "files": []})

            return json.dumps({"files": files})
        except Exception as e:
            logger.exception("search_folder_contents tool failed folder=%r", folder_name)
            return json.dumps({"error": str(e), "files": []})

    def _arun(self, folder_name: str):
        raise NotImplementedError
