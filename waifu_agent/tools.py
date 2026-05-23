from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from waifu_agent.memory import MemoryStore


@dataclass(frozen=True)
class ToolResult:
    name: str
    content: str


class AgentTools:
    def __init__(self, memory: MemoryStore, workspace: Path) -> None:
        self.memory = memory
        self.workspace = workspace.resolve()

    def remember(self, text: str, kind: str = "fact") -> ToolResult:
        item = self.memory.add(text, kind=kind)
        return ToolResult("remember", f"Сохранено в память #{item.id}: {item.content}")

    def now(self) -> ToolResult:
        return ToolResult("now", datetime.now().isoformat(timespec="seconds"))

    def list_workspace(self) -> ToolResult:
        files = sorted(
            str(path.relative_to(self.workspace))
            for path in self.workspace.rglob("*")
            if path.is_file() and ".git" not in path.parts and "venv" not in path.parts
        )
        if not files:
            return ToolResult("list_workspace", "В рабочей папке нет проектных файлов.")
        return ToolResult("list_workspace", "\n".join(files[:80]))

    def describe(self) -> str:
        return (
            "Доступные безопасные инструменты агента:\n"
            "- remember(text, kind='fact'): сохранить важный факт в долговременную память.\n"
            "- now(): узнать локальное время.\n"
            "- list_workspace(): показать файлы проекта без venv.\n"
            "В этой версии инструменты вызываются приложением явно через CLI-команды, "
            "а модель получает их описание в системном контексте."
        )
