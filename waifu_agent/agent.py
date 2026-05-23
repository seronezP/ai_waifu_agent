from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from waifu_agent.memory import MemoryStore
from waifu_agent.model import ChatModel, ModelConfig, load_chat_model
from waifu_agent.persona import Persona
from waifu_agent.tools import AgentTools


@dataclass(frozen=True)
class AgentConfig:
    root: Path = field(default_factory=lambda: Path.cwd())
    persona_path: Optional[Path] = None
    memory_path: Optional[Path] = None
    history_limit: int = 10

    def resolved_persona_path(self) -> Path:
        return self.persona_path or self.root / "data" / "persona.json"

    def resolved_memory_path(self) -> Path:
        return self.memory_path or self.root / "data" / "memory.sqlite3"


class WaifuAgent:
    def __init__(self, config: AgentConfig, model: Optional[ChatModel] = None) -> None:
        self.config = config
        self.persona = Persona.from_file(config.resolved_persona_path())
        self.memory = MemoryStore(config.resolved_memory_path())
        self.tools = AgentTools(self.memory, config.root)
        self.model = model or load_chat_model(ModelConfig())
        self.history: list[dict[str, str]] = []

    def reply(self, user_text: str) -> str:
        memory_context = self.memory.format_context(user_text)
        messages = [
            {
                "role": "system",
                "content": (
                    self.persona.system_prompt()
                    + "\n\nДолговременная память, релевантная текущему сообщению:\n"
                    + memory_context
                    + "\n\n"
                    + self.tools.describe()
                ),
            },
            *self.history[-self.config.history_limit :],
            {
                "role": "system",
                "content": (
                    "Правила текущего ответа: отвечай по-русски, от первого лица. "
                    "Ты уже являешься Рэни; если пользователь пишет 'Рэни', это обращение к тебе. "
                    "Не комментируй язык сообщения пользователя, не проси спрашивать по-русски. "
                    "Обращайся к пользователю на ты."
                ),
            },
            {"role": "user", "content": user_text},
        ]
        answer = self._clean_answer(self.model.generate(messages))
        self.history.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": answer},
            ]
        )
        self._auto_remember(user_text, answer)
        return answer

    def _auto_remember(self, user_text: str, answer: str) -> None:
        lowered = user_text.lower()
        memory_markers = (
            "запомни",
            "помни",
            "меня зовут",
            "я люблю",
            "мне нравится",
            "я не люблю",
            "мой ",
            "моя ",
        )
        if any(marker in lowered for marker in memory_markers):
            self.memory.add(f"Пользователь сказал: {user_text}", kind="user_fact")
        if "запомню" in answer.lower() or "сохраню" in answer.lower():
            self.memory.add(f"Контекст из диалога: {user_text}", kind="dialogue")

    def reset_session(self) -> None:
        self.history.clear()

    @staticmethod
    def _clean_answer(answer: str) -> str:
        banned_fragments = (
            "пожалуйста, спрашивайте меня по-русски",
            "пожалуйста, спрашивай меня по-русски",
            "спрашивайте меня по-русски",
            "спрашивай меня по-русски",
            "задавайте вопросы по-русски",
            "задавай вопросы по-русски",
        )
        cleaned = answer.strip()
        lowered = cleaned.lower()
        for fragment in banned_fragments:
            index = lowered.find(fragment)
            if index != -1:
                cleaned = cleaned[:index].rstrip(" ,.!?;-")
                lowered = cleaned.lower()
        cleaned = re.sub(
            r"^(привет|здравствуй|здарова),?\s+(рэни|rany)[!.]?",
            r"\1!",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = cleaned.replace("вам помочь", "тебе помочь")
        cleaned = cleaned.replace("Вам помочь", "Тебе помочь")
        cleaned = cleaned.replace("помочь вам", "помочь тебе")
        cleaned = cleaned.replace("Помочь вам", "Помочь тебе")
        return cleaned or answer.strip()
