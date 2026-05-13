import datetime
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Type

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool

from backend.google_drive import (
    FOLDER_MIME,
    collect_descendant_folder_ids,
    dedupe_files,
    drive_quote,
    search_drive,
)

logger = logging.getLogger(__name__)

MAX_RESULT_FILES = int(os.getenv("MAX_RESULT_FILES", "30"))
MAX_SEARCH_FOLDERS = int(os.getenv("MAX_SEARCH_FOLDERS", "60"))
MAX_TOOL_FILES = int(os.getenv("MAX_TOOL_FILES", "8"))
SEARCH_TIMEOUT_SECONDS = float(os.getenv("SEARCH_TIMEOUT_SECONDS", "35"))

STOPWORDS = {
    "a", "an", "all", "any", "can", "could", "drive", "file", "files",
    "find", "for", "from", "get", "give", "google", "in", "inside", "list",
    "locate", "me", "my", "of", "only", "open", "please", "search", "show",
    "the", "to", "with", "within", "you",
}

FOLDER_WORDS = {"folder", "folders", "directory", "directories"}

MIME_ALIASES = {
    "pic": ["image/"],
    "pics": ["image/"],
    "picture": ["image/"],
    "pictures": ["image/"],
    "photo": ["image/"],
    "photos": ["image/"],
    "image": ["image/"],
    "images": ["image/"],
    "png": ["image/png"],
    "jpg": ["image/jpeg"],
    "jpeg": ["image/jpeg"],
    "pdf": ["application/pdf"],
    "pdfs": ["application/pdf"],
    "doc": [
        "application/vnd.google-apps.document",
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ],
    "docs": [
        "application/vnd.google-apps.document",
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ],
    "document": [
        "application/vnd.google-apps.document",
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ],
    "documents": [
        "application/vnd.google-apps.document",
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ],
    "sheet": [
        "application/vnd.google-apps.spreadsheet",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/csv",
    ],
    "sheets": [
        "application/vnd.google-apps.spreadsheet",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/csv",
    ],
    "spreadsheet": [
        "application/vnd.google-apps.spreadsheet",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/csv",
    ],
    "spreadsheets": [
        "application/vnd.google-apps.spreadsheet",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/csv",
    ],
    "video": ["video/"],
    "videos": ["video/"],
}

EXTENSION_ALIASES = {
    "code": ["py", "java", "js", "jsx", "ts", "tsx", "cpp", "c", "h", "html", "css"],
    "source": ["py", "java", "js", "jsx", "ts", "tsx", "cpp", "c", "h", "html", "css"],
    "python": ["py"],
    "py": ["py"],
    "java": ["java"],
    "javascript": ["js", "jsx"],
    "js": ["js"],
    "cpp": ["cpp"],
    "c++": ["cpp"],
    "csv": ["csv"],
    "xlsx": ["xlsx"],
    "xls": ["xls"],
    "docx": ["docx"],
}

RECENT_WORDS = {
    "recent": 14,
    "latest": 14,
    "new": 14,
    "today": 0,
    "yesterday": 1,
    "week": 7,
    "month": 30,
}


@dataclass
class ParsedQuery:
    raw: str
    terms: list[str] = field(default_factory=list)
    folder_terms: list[str] = field(default_factory=list)
    mime_types: list[str] = field(default_factory=list)
    extensions: list[str] = field(default_factory=list)
    modified_after: str | None = None
    wants_folders: bool = False
    folder_scoped: bool = False
    open_intent: bool = False


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_+#.-]+", text.lower())


def _term_variants(term: str) -> list[str]:
    variants = [term]
    if term.endswith("ies") and len(term) > 3:
        variants.append(term[:-3] + "y")
    if term.endswith("es") and len(term) > 3:
        variants.append(term[:-2])
    if term.endswith("s") and len(term) > 2:
        variants.append(term[:-1])
    else:
        variants.append(term + "s")
    return list(dict.fromkeys(v for v in variants if v))


def _add_unique(values: list[str], additions: list[str]) -> None:
    for value in additions:
        if value not in values:
            values.append(value)


def _compact_tool_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for file in files[:MAX_TOOL_FILES]:
        compact.append(
            {
                "id": file.get("id"),
                "name": file.get("name"),
                "mimeType": file.get("mimeType"),
                "webViewLink": file.get("webViewLink") or file.get("link"),
                "modifiedTime": file.get("modifiedTime") or file.get("modified"),
                "size": file.get("size"),
            }
        )
    return compact


def _extract_folder_terms(text: str) -> tuple[list[str], bool]:
    clean = text.lower()
    patterns = [
        r"(?:from|in|inside|within)\s+(.+?)\s+(?:folder|directory)\b",
        r"\b(.+?)\s+(?:folder|directory)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean)
        if not match:
            continue
        raw = match.group(1)
        terms = [
            token for token in _tokens(raw)
            if token not in STOPWORDS and token not in FOLDER_WORDS
        ]
        return terms, True
    return [], False


def parse_query(raw_query: str) -> ParsedQuery:
    parsed = ParsedQuery(raw=raw_query.strip())
    words = _tokens(raw_query)
    parsed.open_intent = "open" in words
    parsed.folder_terms, parsed.folder_scoped = _extract_folder_terms(raw_query)
    parsed.wants_folders = any(word in FOLDER_WORDS for word in words) and not any(
        marker in words for marker in ["from", "in", "inside", "within"]
    )

    now = datetime.datetime.utcnow()
    for word in words:
        if word in FOLDER_WORDS or word in STOPWORDS:
            continue
        if word in parsed.folder_terms:
            continue
        if word in MIME_ALIASES:
            _add_unique(parsed.mime_types, MIME_ALIASES[word])
            continue
        if word in EXTENSION_ALIASES:
            _add_unique(parsed.extensions, EXTENSION_ALIASES[word])
            continue
        if word in RECENT_WORDS:
            days = RECENT_WORDS[word]
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if days:
                start = now - datetime.timedelta(days=days)
            parsed.modified_after = start.strftime("%Y-%m-%dT%H:%M:%SZ")
            continue
        parsed.terms.append(word)

    if parsed.wants_folders and not parsed.folder_terms:
        parsed.folder_terms = parsed.terms
        parsed.terms = []

    parsed.terms = list(dict.fromkeys(parsed.terms))
    parsed.folder_terms = list(dict.fromkeys(parsed.folder_terms))
    logger.debug("Parsed query=%r parsed=%s", raw_query, parsed)
    return parsed


def _mime_clause(mime_types: list[str]) -> str | None:
    if not mime_types:
        return None
    parts = []
    for mime_type in mime_types:
        if mime_type.endswith("/"):
            parts.append(f"mimeType contains '{drive_quote(mime_type)}'")
        else:
            parts.append(f"mimeType = '{drive_quote(mime_type)}'")
    return f"({' or '.join(parts)})" if len(parts) > 1 else parts[0]


def _extension_clause(extensions: list[str]) -> str | None:
    if not extensions:
        return None
    parts = [f"name contains '.{drive_quote(ext.lstrip('.'))}'" for ext in extensions]
    return f"({' or '.join(parts)})" if len(parts) > 1 else parts[0]


def _term_clause(terms: list[str], field_name: str = "name", mode: str = "and") -> str | None:
    if not terms:
        return None

    parts = []
    for term in terms:
        variant_parts = [
            f"{field_name} contains '{drive_quote(variant)}'"
            for variant in _term_variants(term)
        ]
        if len(variant_parts) > 1:
            parts.append(f"({' or '.join(variant_parts)})")
        else:
            parts.append(variant_parts[0])

    joiner = " and " if mode == "and" else " or "
    return f"({joiner.join(parts)})" if len(parts) > 1 else parts[0]


def _build_query(
    parsed: ParsedQuery,
    terms: list[str] | None = None,
    term_field: str = "name",
    term_mode: str = "and",
    folders_only: bool = False,
    include_type_filters: bool = True,
) -> str:
    clauses: list[str] = []
    if folders_only:
        clauses.append(f"mimeType = '{FOLDER_MIME}'")
    elif include_type_filters:
        mime = _mime_clause(parsed.mime_types)
        ext = _extension_clause(parsed.extensions)
        type_parts = [part for part in [mime, ext] if part]
        if len(type_parts) > 1:
            clauses.append(f"({' or '.join(type_parts)})")
        elif type_parts:
            clauses.append(type_parts[0])

    selected_terms = parsed.terms if terms is None else terms
    term_filter = _term_clause(selected_terms, field_name=term_field, mode=term_mode)
    if term_filter:
        clauses.append(term_filter)
    if parsed.modified_after:
        clauses.append(f"modifiedTime >= '{parsed.modified_after}'")
    return " and ".join(clauses)


def _search_many_folders(
    query: str,
    folder_ids: list[str] | None,
    per_folder_limit: int = MAX_RESULT_FILES,
) -> list[dict[str, Any]]:
    logger.info(
        "Vector/search layer executing query=%r scoped_folders=%s per_folder_limit=%s",
        query,
        len(folder_ids or []),
        per_folder_limit,
    )
    if folder_ids:
        files: list[dict[str, Any]] = []
        for folder_id in folder_ids[:MAX_SEARCH_FOLDERS]:
            remaining = max(MAX_RESULT_FILES - len(files), 1)
            files.extend(
                search_drive(
                    query=query,
                    folder_id=folder_id,
                    page_size=min(per_folder_limit, remaining),
                )
            )
            if len(files) >= MAX_RESULT_FILES:
                break
        results = dedupe_files(files)[:MAX_RESULT_FILES]
        logger.info("Scoped search complete count=%s", len(results))
        return results
    results = search_drive(query=query, page_size=MAX_RESULT_FILES)
    logger.info("Global search complete count=%s", len(results))
    return results


def _folder_scope(root_folder_id: str | None) -> list[str] | None:
    if not root_folder_id:
        return None
    return collect_descendant_folder_ids(root_folder_id, max_folders=MAX_SEARCH_FOLDERS)


def _score_terms(terms: list[str], name: str) -> int:
    if not terms:
        return 100
    target = " ".join(terms)
    lower_name = name.lower()
    contains_bonus = sum(25 for term in terms if any(v in lower_name for v in _term_variants(term)))
    ratio = int(SequenceMatcher(None, target, lower_name).ratio() * 100)
    return min(100, ratio + contains_bonus)


def _local_filter(files: list[dict[str, Any]], parsed: ParsedQuery) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for file in files:
        name = file.get("name", "").lower()
        mime = file.get("mimeType", "").lower()
        if parsed.mime_types and not any(
            mime.startswith(m[:-1]) if m.endswith("/") else mime == m
            for m in parsed.mime_types
        ):
            continue
        if parsed.extensions and not any(name.endswith("." + ext.lstrip(".").lower()) for ext in parsed.extensions):
            continue
        if parsed.terms and _score_terms(parsed.terms, name) < 45:
            continue
        filtered.append(file)
    return filtered


def _check_search_timeout(started: float, stage: str) -> None:
    elapsed = time.monotonic() - started
    if elapsed > SEARCH_TIMEOUT_SECONDS:
        logger.error("Search timeout stage=%s elapsed_seconds=%.2f", stage, elapsed)
        raise TimeoutError(f"Drive search timed out during {stage}")


def find_matching_folders(parsed: ParsedQuery, root_folder_id: str | None = None) -> list[dict[str, Any]]:
    logger.info("Folder matching started terms=%s root=%s", parsed.folder_terms or parsed.terms, root_folder_id)
    terms = parsed.folder_terms or parsed.terms
    folder_query = _build_query(parsed, terms=terms, folders_only=True, include_type_filters=False)

    folder_ids = _folder_scope(root_folder_id)
    if folder_ids:
        folders = _search_many_folders(folder_query, folder_ids, per_folder_limit=20)
        if folders:
            return folders
        all_folders = _search_many_folders(
            f"mimeType = '{FOLDER_MIME}'",
            folder_ids,
            per_folder_limit=100,
        )
        return [
            folder for folder in all_folders
            if _score_terms(terms, folder.get("name", "")) >= 45
        ][:MAX_RESULT_FILES]

    folders = search_drive(query=folder_query, page_size=MAX_RESULT_FILES)
    if folders:
        logger.info("Folder matching complete count=%s", len(folders))
        return folders

    # Global fuzzy fallback for plural/singular mismatches.
    all_folders = search_drive(query=f"mimeType = '{FOLDER_MIME}'", page_size=100)
    return [
        folder for folder in all_folders
        if _score_terms(terms, folder.get("name", "")) >= 45
    ][:MAX_RESULT_FILES]


def staged_search(raw_query: str, folder_id: str | None = None) -> list[dict[str, Any]]:
    """Deterministic, recursive Drive search used by both API and tools."""
    started = time.monotonic()
    root = folder_id or os.getenv("TARGET_FOLDER_ID")
    parsed = parse_query(raw_query)
    logger.info("Staged Drive search started query=%r root=%s", raw_query[:500], root)

    if parsed.wants_folders:
        _check_search_timeout(started, "folder_match")
        results = dedupe_files(find_matching_folders(parsed, root))[:MAX_RESULT_FILES]
        logger.info(
            "Staged Drive folder search complete count=%s duration_ms=%s",
            len(results),
            int((time.monotonic() - started) * 1000),
        )
        return results

    scope_folder_ids: list[str] | None = None
    if parsed.folder_scoped and parsed.folder_terms:
        _check_search_timeout(started, "folder_scope")
        folders = find_matching_folders(parsed, root)
        scope_folder_ids = []
        for folder in folders[:5]:
            current_folder_id = folder.get("id")
            if current_folder_id:
                scope_folder_ids.extend(
                    collect_descendant_folder_ids(
                        current_folder_id,
                        max_folders=MAX_SEARCH_FOLDERS,
                    )
                )
        scope_folder_ids = list(dict.fromkeys(scope_folder_ids))
        if not scope_folder_ids:
            logger.debug("No matching folders found for scoped query=%r", raw_query)
            logger.info("Staged Drive search stopped: no matching scope folders")
            return []
    elif root:
        _check_search_timeout(started, "root_scope")
        scope_folder_ids = _folder_scope(root)

    stages = [
        ("name_all", _build_query(parsed, term_field="name", term_mode="and")),
        ("type_or_recent", _build_query(parsed, terms=[], term_field="name")),
        ("name_any", _build_query(parsed, term_field="name", term_mode="or")),
        ("fulltext_all", _build_query(parsed, term_field="fullText", term_mode="and")),
        ("fulltext_any", _build_query(parsed, term_field="fullText", term_mode="or")),
    ]

    for stage_name, query in stages:
        if not query:
            continue
        _check_search_timeout(started, stage_name)
        logger.info("Search stage started stage=%s query=%r folders=%s", stage_name, query, len(scope_folder_ids or []))
        results = _search_many_folders(query, scope_folder_ids)
        if results:
            logger.info(
                "Search stage succeeded stage=%s count=%s duration_ms=%s",
                stage_name,
                len(results),
                int((time.monotonic() - started) * 1000),
            )
            return results
        logger.info("Search stage empty stage=%s", stage_name)

    # Final fallback: list visible scope and fuzzy-filter locally.
    logger.info("Search stage started stage=fuzzy_list folders=%s", len(scope_folder_ids or []))
    _check_search_timeout(started, "fuzzy_list")
    pool = _search_many_folders("", scope_folder_ids, per_folder_limit=100)
    matches = _local_filter(pool, parsed)
    scored = sorted(
        [(_score_terms(parsed.terms, file.get("name", "")), file) for file in matches],
        key=lambda item: item[0],
        reverse=True,
    )
    results = dedupe_files([file for score, file in scored if score >= 45 or not parsed.terms])[:MAX_RESULT_FILES]
    logger.info(
        "Staged Drive search complete stage=fuzzy_list count=%s duration_ms=%s",
        len(results),
        int((time.monotonic() - started) * 1000),
    )
    return results


class DriveSearchInput(BaseModel):
    query: str = Field(description="Drive search query.")


class DriveSearchTool(BaseTool):
    name: str = "google_drive_search"
    description: str = "Search Google Drive files/folders."
    args_schema: Type[BaseModel] = DriveSearchInput

    def _run(self, query: str) -> str:
        try:
            logger.info("Tool google_drive_search started query=%r", query[:500])
            files = staged_search(query)
            logger.info("Tool google_drive_search complete count=%s", len(files))
            return json.dumps({"files": _compact_tool_files(files), "count": len(files)})
        except Exception as exc:
            logger.exception("google_drive_search failed query=%r", query)
            return json.dumps({"files": [], "error": str(exc)})

    async def _arun(self, query: str) -> str:
        return self._run(query)


class FolderContentsInput(BaseModel):
    folder_name: str = Field(description="Folder name.")


class SearchFolderContentsTool(BaseTool):
    name: str = "search_folder_contents"
    description: str = "List files in a Drive folder."
    args_schema: Type[BaseModel] = FolderContentsInput

    def _run(self, folder_name: str) -> str:
        try:
            logger.info("Tool search_folder_contents started folder=%r", folder_name[:500])
            query = folder_name if "folder" in folder_name.lower() else f"{folder_name} folder"
            parsed = parse_query(query)
            folders = find_matching_folders(parsed, os.getenv("TARGET_FOLDER_ID"))
            files: list[dict[str, Any]] = []

            for folder in folders[:5]:
                folder_id = folder.get("id")
                if not folder_id:
                    continue
                folder_ids = collect_descendant_folder_ids(
                    folder_id,
                    max_folders=MAX_SEARCH_FOLDERS,
                )
                for descendant_id in folder_ids:
                    files.extend(search_drive(folder_id=descendant_id, page_size=MAX_RESULT_FILES))

            files = dedupe_files(files)[:MAX_RESULT_FILES]
            logger.info("Tool search_folder_contents complete count=%s", len(files))
            return json.dumps({"files": _compact_tool_files(files), "count": len(files)})
        except Exception as exc:
            logger.exception("search_folder_contents failed folder=%r", folder_name)
            return json.dumps({"files": [], "error": str(exc)})

    async def _arun(self, folder_name: str) -> str:
        return self._run(folder_name)


@lru_cache(maxsize=1)
def get_drive_search_tool() -> DriveSearchTool:
    logger.info("Loading google_drive_search tool")
    return DriveSearchTool()


@lru_cache(maxsize=1)
def get_folder_contents_tool() -> SearchFolderContentsTool:
    logger.info("Loading search_folder_contents tool")
    return SearchFolderContentsTool()


def preload_tools() -> None:
    started = time.monotonic()
    get_drive_search_tool()
    get_folder_contents_tool()
    logger.info("Tools preloaded duration_ms=%s", int((time.monotonic() - started) * 1000))
