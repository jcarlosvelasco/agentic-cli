from pathlib import Path


def get_session_folder_path() -> str:
    current_dir = Path(__file__).parent
    folder_path = str(current_dir / "data" / "sessions")
    return folder_path


def get_session_index_file_path() -> str:
    current_dir = Path(__file__).parent
    index_path = str(current_dir / "data" / "session_index.json")
    return index_path
