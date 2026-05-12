import datetime

import streamlit as st

_BADGE = {
    "folder":      ("badge-folder",      "Folder"),
    "image":       ("badge-image",       "Image"),
    "pdf":         ("badge-pdf",         "PDF"),
    "spreadsheet": ("badge-spreadsheet", "Spreadsheet"),
    "document":    ("badge-document",    "Document"),
    "video":       ("badge-video",       "Video"),
}

_EMOJI = {
    "folder":      "📁",
    "image":       "🖼️",
    "video":       "🎥",
    "spreadsheet": "📊",
    "pdf":         "📕",
    "document":    "📝",
}


def _kind(mime: str) -> str:
    m = mime.lower()
    for key in ("folder", "image", "pdf", "spreadsheet", "document", "video"):
        if key in m:
            return key
    return "other"


def _fmt_size(raw) -> str:
    if not raw:
        return ""
    try:
        size = int(raw)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    except Exception:
        return ""


def _fmt_date(raw: str) -> str:
    try:
        if raw:
            dt = datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return dt.strftime("%b %d, %Y")
    except Exception:
        pass
    return raw[:10] if raw else "Unknown"


def _card_html(file: dict, backend_base: str) -> str:
    name     = file.get("name", "Unknown")
    mime     = file.get("mimeType", "")
    link     = file.get("webViewLink", "#")
    icon_url = file.get("iconLink", "")
    file_id  = file.get("id", "")
    modified = _fmt_date(file.get("modifiedTime", ""))
    owners   = file.get("owners", [])
    owner    = owners[0].get("displayName", "Unknown") if owners else "Unknown"
    size_str = _fmt_size(file.get("size", ""))

    kind = _kind(mime)
    badge_cls, badge_label = _BADGE.get(kind, ("badge-other", "File"))
    emoji = _EMOJI.get(kind, "📄")
    size_part = f" • 💾 {size_str}" if size_str else ""

    if icon_url:
        icon_html = f"<img src='{icon_url}' width='14' style='vertical-align:middle;margin-right:4px;'/>"
    else:
        icon_html = f"<span style='font-size:0.9rem;margin-right:4px;'>{emoji}</span>"

    if kind == "image" and file_id:
        thumb_url = f"{backend_base}/thumbnail/{file_id}"
        thumb = (
            f"<div class='file-thumb'>"
            f"<img class='file-thumb-img' src='{thumb_url}' alt='' loading='lazy'/>"
            f"<div class='file-thumb-icon file-thumb-icon-delayed'>{emoji}</div>"
            f"</div>"
        )
    else:
        thumb = (
            f"<div class='file-thumb'>"
            f"<div class='file-thumb-icon'>{emoji}</div>"
            f"</div>"
        )

    return (
        f"<div class='file-card'>"
        + thumb
        + f"<div class='file-card-body'>"
        + f"<div class='file-card-name' title='{name}'>{icon_html}<span>{name}</span></div>"
        + f"<div class='file-card-meta'>"
        + f"<span class='badge {badge_cls}'>{badge_label}</span>"
        + f"<div class='file-card-meta-row'>📅 {modified}</div>"
        + f"<div class='file-card-meta-row'>👤 {owner}{size_part}</div>"
        + f"</div>"
        + f"<a href='{link}' target='_blank' class='drive-btn'>Open in Drive ↗</a>"
        + f"</div>"
        + f"</div>"
    )


def render_file_results(files: list, backend_base: str = "http://localhost:8000"):
    if not files:
        return

    count = len(files)
    label = "item" if count == 1 else "items"
    st.markdown(
        f"<div class='result-count'>📑 Found {count} {label}</div>",
        unsafe_allow_html=True,
    )

    grid = "".join(_card_html(f, backend_base) for f in files)
    st.markdown(f"<div class='cards-grid'>{grid}</div>", unsafe_allow_html=True)
