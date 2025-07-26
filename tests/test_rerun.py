import types
import streamlit_app


def test_maybe_rerun_experimental(monkeypatch):
    called = {}

    def exp():
        called['exp'] = True

    stub = types.SimpleNamespace(experimental_rerun=exp)
    monkeypatch.setattr(streamlit_app, "st", stub)
    streamlit_app.maybe_rerun()
    assert called.get('exp', False)


def test_maybe_rerun_new(monkeypatch):
    called = {}

    def new():
        called['rerun'] = True

    stub = types.SimpleNamespace(rerun=new)
    monkeypatch.setattr(streamlit_app, "st", stub)
    streamlit_app.maybe_rerun()
    assert called.get('rerun', False)
