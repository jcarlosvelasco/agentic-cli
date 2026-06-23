import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.llm.schema.Message import Message, MessageRole
from src.memory.interface.Session import Session
from src.memory.interface.SessionIndex import SessionIndex
from src.memory.preamble import preamble


class TestSession:
    def test_created_at_is_set_on_init(self):
        session = Session()
        assert session.created_at is not None

    def test_id_matches_created_at(self):
        session = Session()
        assert session.id == session.created_at

    def test_append_adds_message(self):
        session = Session()
        msg = Message(role=MessageRole.USER, content="hello")
        session.append(msg)
        assert len(session.messages) == 1
        assert session.messages[0] == msg

    def test_append_multiple_messages(self):
        session = Session()
        msgs = [
            Message(role=MessageRole.USER, content="hi"),
            Message(role=MessageRole.ASSISTANT, content="hello"),
        ]
        for msg in msgs:
            session.append(msg)
        assert len(session.messages) == 2

    def test_serialization_round_trip(self):
        session = Session(
            messages=[
                Message(role=MessageRole.USER, content="test"),
            ]
        )
        data = session.model_dump()
        restored = Session.model_validate(data)
        assert restored.id == session.id
        assert len(restored.messages) == 1
        assert restored.messages[0].content == "test"

    @patch("src.memory.interface.Session.get_session_folder_path")
    @patch("src.memory.interface.Session.Path")
    def test_save_creates_directory_and_file(self, mock_path_class, mock_get_folder):
        mock_path = MagicMock()
        mock_path_class.return_value = mock_path
        mock_path.exists.return_value = False

        tmpdir = tempfile.mkdtemp()
        mock_get_folder.return_value = tmpdir

        session = Session(
            messages=[Message(role=MessageRole.USER, content="hello")]
        )
        session.save()

        session_file = Path(tmpdir) / f"session_{session.id}.json"
        assert session_file.exists()
        with open(session_file) as f:
            data = json.load(f)
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == "hello"


class TestSessionIndex:
    @patch("src.memory.interface.SessionIndex.get_session_index_file_path")
    def test_create_writes_empty_list(self, mock_get_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "session_index.json"
            mock_get_path.return_value = str(index_path)

            SessionIndex.create()
            assert index_path.exists()
            with open(index_path) as f:
                assert json.load(f) == []

    @patch("src.memory.interface.SessionIndex.get_session_index_file_path")
    def test_create_does_not_overwrite_existing(self, mock_get_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "session_index.json"
            index_path.write_text(json.dumps([{"existing": True}]))
            mock_get_path.return_value = str(index_path)

            SessionIndex.create()
            with open(index_path) as f:
                data = json.load(f)
            assert data == [{"existing": True}]

    @patch("src.memory.interface.SessionIndex.get_session_index_file_path")
    @patch("src.memory.interface.SessionIndex.get_session_folder_path")
    def test_update_adds_new_entry(self, mock_get_folder, mock_get_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "session_index.json"
            mock_get_path.return_value = str(index_path)
            mock_get_folder.return_value = str(tmpdir)

            SessionIndex.update("session_123", "Test summary", ["tag1"])

            with open(index_path) as f:
                entries = json.load(f)
            assert len(entries) == 1
            assert entries[0]["summary"] == "Test summary"
            assert entries[0]["tags"] == ["tag1"]

    @patch("src.memory.interface.SessionIndex.get_session_index_file_path")
    @patch("src.memory.interface.SessionIndex.get_session_folder_path")
    def test_update_updates_existing_entry(self, mock_get_folder, mock_get_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "session_index.json"
            index_path.write_text(json.dumps([
                {"summary": "old", "tags": [], "session_path": f"{tmpdir}/session_xyz"}
            ]))
            mock_get_path.return_value = str(index_path)
            mock_get_folder.return_value = str(tmpdir)

            SessionIndex.update("session_xyz", "new summary", ["updated"])

            with open(index_path) as f:
                entries = json.load(f)
            assert len(entries) == 1
            assert entries[0]["summary"] == "new summary"
            assert entries[0]["tags"] == ["updated"]

    @patch("src.memory.interface.SessionIndex.get_session_index_file_path")
    def test_update_creates_index_if_not_exists(self, mock_get_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "session_index.json"
            mock_get_path.return_value = str(index_path)

            with patch("src.memory.interface.SessionIndex.get_session_folder_path") as mock_get_folder:
                mock_get_folder.return_value = str(tmpdir)
                SessionIndex.update("s1", "summary", [])

            assert index_path.exists()


class TestPreamble:
    @patch("src.memory.preamble.get_session_index_file_path")
    async def test_no_index_file_returns_empty(self, mock_get_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_path.return_value = str(Path(tmpdir) / "nonexistent.json")

            config = MagicMock()
            config.memory.max_recent_sessions = 5

            result = await preamble(config)
            assert result == ""

    @patch("src.memory.preamble.get_session_index_file_path")
    async def test_empty_index_returns_empty(self, mock_get_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "session_index.json"
            index_path.write_text("[]")
            mock_get_path.return_value = str(index_path)

            config = MagicMock()
            config.memory.max_recent_sessions = 5

            result = await preamble(config)
            assert result == ""

    @patch("src.memory.preamble.get_session_index_file_path")
    async def test_formats_entries_correctly(self, mock_get_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "session_index.json"
            index_path.write_text(json.dumps([
                {
                    "summary": "Built a feature",
                    "tags": ["python", "testing"],
                    "session_path": f"{tmpdir}/20240101120000",
                }
            ]))
            mock_get_path.return_value = str(index_path)

            config = MagicMock()
            config.memory.max_recent_sessions = 5

            result = await preamble(config)
            assert "## Recent session summaries" in result
            assert "Built a feature" in result
            assert "python, testing" in result

    @patch("src.memory.preamble.get_session_index_file_path")
    async def test_respects_max_recent_sessions(self, mock_get_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "session_index.json"
            index_path.write_text(json.dumps([
                {"summary": f"Session {i}", "tags": [], "session_path": f"{tmpdir}/{i}"}
                for i in range(10)
            ]))
            mock_get_path.return_value = str(index_path)

            config = MagicMock()
            config.memory.max_recent_sessions = 3

            result = await preamble(config)
            assert "Session 7" in result
            assert "Session 9" in result
            assert "Session 0" not in result
