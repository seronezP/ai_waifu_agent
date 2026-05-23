from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Persona:
    name: str
    language: str
    style: list[str]
    boundaries: list[str]
    agent_rules: list[str]
    backstory: str

    @classmethod
    def from_file(cls, path: Path) -> "Persona":
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            name=str(data["name"]),
            language=str(data.get("language", "ru")),
            style=list(data.get("style", [])),
            boundaries=list(data.get("boundaries", [])),
            agent_rules=list(data.get("agent_rules", [])),
            backstory=str(data.get("backstory", "")),
        )

    def system_prompt(self) -> str:
        style = "\n".join(f"- {item}" for item in self.style)
        boundaries = "\n".join(f"- {item}" for item in self.boundaries)
        rules = "\n".join(f"- {item}" for item in self.agent_rules)
        return (
            f"Ты {self.name}, локальная AI-waifu и агент-помощник.\n"
            f"Основной язык: {self.language}.\n\n"
            f"Краткая история:\n{self.backstory}\n\n"
            f"Характер и стиль:\n{style}\n\n"
            f"Границы:\n{boundaries}\n\n"
            f"Агентные правила:\n{rules}\n\n"
            "Используй сохраненную память как факты, но не показывай ее списком без причины. "
            "Если пользователь просит практическую помощь, действуй как агент: уточни цель, предложи шаги, "
            "запомни важные предпочтения и дай конкретный следующий шаг. "
            "Всегда говори о себе от первого лица. Не называй себя 'AI-waifu' в обычных ответах, "
            "если пользователь сам не обсуждает настройку персонажа. Не говори о себе в третьем лице. "
            f"Если пользователь пишет '{self.name}' или 'Рэни', это обращение к тебе, а не имя пользователя."
        )
