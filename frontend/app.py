import os

import requests
import streamlit as st
from dotenv import load_dotenv

from components.cards import render_file_results
from components.chat import render_chat_message, render_empty_search_state, render_empty_state
from components.sidebar import render_sidebar
from components.styles import inject_custom_css

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000/chat")
BACKEND_BASE = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "50"))
FRIENDLY_SERVER_ERROR = "Server temporarily unavailable. Please try again."


def post_chat_with_retry(prompt: str) -> requests.Response:
    last_error = None
    for attempt in range(2):
        try:
            resp = requests.post(
                API_URL,
                json={"message": prompt},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if attempt == 0 and resp.status_code >= 500:
                continue
            return resp
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            if attempt == 0:
                continue
            raise
    if last_error:
        raise last_error
    raise requests.exceptions.RequestException("Chat request failed")

st.set_page_config(page_title="Drive Agent", page_icon="📂", layout="wide")
inject_custom_css()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "prompt_trigger" not in st.session_state:
    st.session_state.prompt_trigger = None

render_sidebar()

st.markdown("<div style='max-width:800px;margin:0 auto;'>", unsafe_allow_html=True)

if not st.session_state.messages:
    render_empty_state()
else:
    for msg in st.session_state.messages:
        render_chat_message(msg)
        if msg.get("role") == "assistant" and msg.get("files"):
            render_file_results(msg["files"], backend_base=BACKEND_BASE)
        if msg.get("role") == "assistant" and msg.get("tool_used") and not msg.get("files"):
            render_empty_search_state()

prompt = st.chat_input("Ask me to find anything in your Google Drive...")

if st.session_state.prompt_trigger:
    prompt = st.session_state.prompt_trigger
    st.session_state.prompt_trigger = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_chat_message(st.session_state.messages[-1])

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching Drive..."):
            try:
                resp = post_chat_with_retry(prompt)
                try:
                    data = resp.json()
                except ValueError:
                    data = {}

                if resp.status_code >= 500:
                    raise requests.exceptions.HTTPError(FRIENDLY_SERVER_ERROR, response=resp)
                if resp.status_code >= 400:
                    detail = data.get("detail") if isinstance(data, dict) else None
                    raise requests.exceptions.HTTPError(detail or FRIENDLY_SERVER_ERROR, response=resp)
                if not isinstance(data, dict):
                    raise ValueError("Backend returned an invalid response.")

                text = data.get("response", "No response received.")
                files = data.get("files", [])
                tool_used = data.get("tool_used", False)

                st.session_state.messages.append(
                    {"role": "assistant", "content": text, "files": files, "tool_used": tool_used}
                )

                st.markdown(text)
                if files:
                    st.toast(f"Found {len(files)} items!", icon="✅")
                elif tool_used:
                    st.toast("No matching files found.", icon="ℹ️")

            except requests.exceptions.Timeout:
                msg = FRIENDLY_SERVER_ERROR
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.toast("Request timed out")
            except ValueError:
                msg = FRIENDLY_SERVER_ERROR
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.toast("Invalid server response")
            except requests.exceptions.RequestException:
                msg = FRIENDLY_SERVER_ERROR
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.toast("Connection error", icon="❌")

    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
