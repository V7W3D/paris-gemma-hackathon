from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.db.mongo import Database
from backend.db.store import MemoryStore
from backend.main import create_app


@pytest.fixture
def client(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    """The app under test, pinned to mocked inference and in-memory storage."""
    monkeypatch.setattr("backend.main.get_settings", lambda: settings)

    async def connect(self):
        self._store = MemoryStore()
        return self._store

    monkeypatch.setattr(Database, "connect", connect)

    with TestClient(create_app()) as test_client:
        yield test_client


def test_status_reports_the_wiring(client: TestClient):
    status = client.get("/api/status").json()
    assert status["llm_mocked"] is True
    assert status["search_mocked"] is True
    assert status["alien_endpoint"] == "mock"


def test_chat_crud_round_trip(client: TestClient):
    created = client.post("/api/chats", json={}).json()
    chat_id = created["id"]

    assert client.get("/api/chats").json()[0]["id"] == chat_id
    assert client.get(f"/api/chats/{chat_id}").json()["messages"] == []

    renamed = client.patch(f"/api/chats/{chat_id}", json={"title": "Water claims"}).json()
    assert renamed["title"] == "Water claims"

    assert client.delete(f"/api/chats/{chat_id}").status_code == 204
    assert client.get(f"/api/chats/{chat_id}").status_code == 404
    assert client.get("/api/chats").json() == []


def test_missing_chat_is_a_404(client: TestClient):
    assert client.get("/api/chats/nope").status_code == 404
    assert client.get("/api/chats/nope/contexts").status_code == 404
    assert client.post("/api/chats/nope/messages", json={"content": "x"}).status_code == 404


def test_empty_message_is_rejected(client: TestClient):
    chat_id = client.post("/api/chats", json={}).json()["id"]
    assert client.post(f"/api/chats/{chat_id}/messages", json={"content": "   "}).status_code == 422


def test_non_streaming_turn_returns_the_message_and_events(client: TestClient):
    chat_id = client.post("/api/chats", json={}).json()["id"]
    payload = client.post(
        f"/api/chats/{chat_id}/messages",
        params={"stream": "false"},
        json={"content": "The Eiffel Tower is 330 metres tall including its antennas."},
    ).json()

    assert payload["message"]["role"] == "assistant"
    assert payload["message"]["verdict"] is not None
    assert {e["type"] for e in payload["events"]} >= {"stage", "retrieval", "message", "done"}
    assert not any(e["type"] == "token" for e in payload["events"])


def test_streaming_turn_emits_sse_frames(client: TestClient):
    chat_id = client.post("/api/chats", json={}).json()["id"]
    with client.stream(
        "POST",
        f"/api/chats/{chat_id}/messages",
        json={"content": "Norway generates most of its electricity from hydropower."},
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line[5:])
            for line in response.iter_lines()
            if line.startswith("data:")
        ]

    assert events[0]["type"] == "turn_started"
    assert events[-1]["type"] == "done"
    assert any(event["type"] == "token" for event in events)

    contexts = client.get(f"/api/chats/{chat_id}/contexts").json()
    assert [context["stage"] for context in contexts][0] == "decompose"
