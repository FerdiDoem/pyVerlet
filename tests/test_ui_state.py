import streamlit as st

from utils.ui_state import init_state, reset_state, toggle_running


def test_init_state():
    st.session_state.clear()
    init_state()
    assert st.session_state['running'] is False
    assert st.session_state['solver'] is None


def test_toggle_and_reset():
    st.session_state.clear()
    init_state()
    toggle_running()
    assert st.session_state['running'] is True
    toggle_running()
    assert st.session_state['running'] is False
    st.session_state['solver'] = object()
    reset_state()
    assert st.session_state['solver'] is None
    assert st.session_state['running'] is False
