import streamlit as st


def render_chat_message(message: dict):
    """Renders a chat bubble (text only). File cards are rendered separately."""
    role = message.get("role", "user")
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message.get("content", ""))


def render_empty_state():
    st.markdown(
        """
        <div style="text-align:center;margin-top:50px;margin-bottom:50px;opacity:0.8;">
            <h1 style="font-size:3rem;margin-bottom:10px;color:#24292f;">👋 Welcome!</h1>
            <p style="font-size:1.2rem;color:#57606a;">Your intelligent Google Drive assistant.</p>
            <p style="font-size:1rem;color:#57606a;max-width:500px;margin:0 auto;">
                Ask me to find documents, search for file types, or look inside folders.
                Try a suggested prompt in the sidebar to get started!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_search_state():
    st.info("No files found matching your request.", icon="ℹ️")
    with st.expander("💡 Search tips"):
        st.markdown(
            """
            - **Check spelling** — try a single keyword instead of the full name
            - **Specify file type** — e.g. "budget spreadsheet" or "invoice PDF"
            - **Broaden the date** — e.g. "images from last month" instead of yesterday
            """
        )
