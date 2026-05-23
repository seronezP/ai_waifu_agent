from pathlib import Path

from waifu_agent.memory import MemoryStore


def test_memory_add_recent_and_search(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.add("Пользователь любит локальные AI-модели", kind="user_fact")

    recent = store.recent()
    assert recent[0].content == "Пользователь любит локальные AI-модели"

    found = store.search("локальные")
    assert found
    assert found[0].kind == "user_fact"
