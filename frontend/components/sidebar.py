import streamlit as st

def render_sidebar():
    """Renders the sidebar navigation, history, and suggested prompts."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="margin-bottom: 0; font-size: 1.8rem; background: -webkit-linear-gradient(45deg, #0969da, #8250df); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Drive Agent</h1>
            <p style="color: #57606a; font-size: 0.9rem;">Your intelligent search assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("### ✨ Suggested Prompts")
        
        # We use session state to trigger a prompt from a button click
        if st.button("📁 Find my invoice folders", use_container_width=True):
            st.session_state.prompt_trigger = "Find the invoices folder"
            
        if st.button("🖼️ Show all images", use_container_width=True):
            st.session_state.prompt_trigger = "Show only image files"
            
        if st.button("📊 Show spreadsheets", use_container_width=True):
            st.session_state.prompt_trigger = "Find spreadsheet files"
            
        st.divider()
        
        st.markdown("### 🕒 Recent Searches")
        if "messages" in st.session_state and st.session_state.messages:
            # Get the last 3 user messages
            user_msgs = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
            recent = user_msgs[-3:][::-1] # Reverse to show newest first
            for msg in recent:
                st.caption(f"🔍 {msg[:30]}{'...' if len(msg)>30 else ''}")
        else:
            st.caption("No recent searches.")
            
        st.divider()
        
        if st.button("🗑️ Clear Chat History", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.prompt_trigger = None
            st.rerun()
