import streamlit as st


def init_state():
    """Initialize session_state variables used by the UI."""
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("solver", None)


def reset_state():
    """Reset solver and running flag."""
    st.session_state["solver"] = None
    st.session_state["running"] = False


def toggle_running():
    """Toggle the running flag."""
    st.session_state["running"] = not st.session_state.get("running", False)
