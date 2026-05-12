import streamlit as st


def inject_custom_css():
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    header { visibility: hidden; }

    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #d0d7de; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #afb8c1; }

    .cards-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 12px;
    }
    @media (max-width: 900px) { .cards-grid { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 580px) { .cards-grid { grid-template-columns: 1fr; } }

    .file-card {
        background: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 10px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .file-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.10);
        border-color: #0969da;
    }

    .file-thumb {
        width: 100%;
        height: 88px;
        position: relative;
        overflow: hidden;
        background-color: #f6f8fa;
        border-bottom: 1px solid #e1e4e8;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .file-thumb-img {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        z-index: 2;
    }
    .file-thumb-icon {
        position: relative;
        font-size: 2.6rem;
        z-index: 1;
        line-height: 1;
    }

    @keyframes fadeInIcon {
        0%   { opacity: 0; }
        65%  { opacity: 0; }
        100% { opacity: 1; }
    }
    .file-thumb-icon-delayed { animation: fadeInIcon 2.8s ease forwards; }

    .file-card-body {
        padding: 9px 10px 8px;
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        gap: 5px;
    }
    .file-card-name {
        font-size: 0.82rem;
        font-weight: 600;
        color: #24292f;
        display: flex;
        align-items: center;
        gap: 5px;
        line-height: 1.25;
    }
    .file-card-name span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .file-card-meta { font-size: 0.71rem; color: #57606a; display: flex; flex-direction: column; gap: 1px; margin-bottom: 4px; }
    .file-card-meta-row { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

    .badge {
        display: inline-block;
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 0.62rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        width: fit-content;
        margin-bottom: 3px;
    }
    .badge-folder      { background:#fff8c5; color:#9a6700; border:1px solid #d4a72c55; }
    .badge-image       { background:#ddf4ff; color:#0969da; border:1px solid #54aeff55; }
    .badge-pdf         { background:#ffebe9; color:#cf222e; border:1px solid #ff818255; }
    .badge-spreadsheet { background:#dafbe1; color:#1a7f37; border:1px solid #4ac26b55; }
    .badge-document    { background:#eef0ff; color:#3451b2; border:1px solid #8fa6f855; }
    .badge-video       { background:#f5f0ff; color:#8250df; border:1px solid #d2a8ff55; }
    .badge-other       { background:#f6f8fa; color:#57606a; border:1px solid #d0d7de55; }

    .drive-btn {
        display: block;
        width: 100%;
        text-align: center;
        padding: 4px 0;
        background: #f6f8fa;
        color: #24292f;
        border: 1px solid #d0d7de;
        border-radius: 6px;
        text-decoration: none !important;
        font-weight: 600;
        font-size: 0.75rem;
        transition: background 0.15s, color 0.15s, border-color 0.15s;
        margin-top: auto;
    }
    .drive-btn:hover { background: #0969da; color: #ffffff !important; border-color: #0969da; }

    .result-count {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #ddf4ff;
        color: #0969da;
        border: 1px solid #54aeff55;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
