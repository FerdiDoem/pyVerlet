import streamlit as st


def init_state():
    """Initialize session_state variables used by the UI."""
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("solver", None)
    st.session_state.setdefault("canvas_slot", st.empty())
    st.session_state.setdefault("fig_ax_scatter", None)


def reset_state():
    """Reset solver and running flag."""
    st.session_state["solver"] = None
    st.session_state["running"] = False
    if st.session_state.get("fig_ax_scatter"):
        import matplotlib.pyplot as plt

        plt.close(st.session_state["fig_ax_scatter"][0])
    st.session_state["fig_ax_scatter"] = None
    st.session_state["canvas_slot"] = st.empty()


def toggle_running():
    """Toggle the running flag."""
    st.session_state["running"] = not st.session_state.get("running", False)


def start_running():
    """Set running flag to True."""
    st.session_state["running"] = True


def stop_running():
    """Set running flag to False."""
    st.session_state["running"] = False
