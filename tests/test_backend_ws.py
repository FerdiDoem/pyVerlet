import os
import sys
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.app import app


def test_backend_websocket():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        data = ws.receive_bytes()
        assert len(data) > 0
        data2 = ws.receive_bytes()
        assert len(data2) > 0

