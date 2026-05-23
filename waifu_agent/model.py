from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol


class ChatModel(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = os.getenv("WAIFU_MODEL", "models/qwen2.5-1.5b-instruct")
    lora_path: Optional[str] = os.getenv("WAIFU_LORA_PATH") or None
    max_new_tokens: int = int(os.getenv("WAIFU_MAX_NEW_TOKENS", "260"))
    temperature: float = float(os.getenv("WAIFU_TEMPERATURE", "0.75"))
    top_p: float = float(os.getenv("WAIFU_TOP_P", "0.9"))


class TransformersChatModel:
    def __init__(self, config: ModelConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Не установлены ML-зависимости. Выполни: pip install -r requirements.txt"
            ) from exc

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        if config.lora_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Для LoRA-адаптера нужен пакет peft.") from exc
            self.model = PeftModel.from_pretrained(self.model, config.lora_path)

    def generate(self, messages: list[dict[str, str]]) -> str:
        import torch

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class RuleBasedFallbackModel:
    def generate(self, messages: list[dict[str, str]]) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return (
            "Я пока работаю в fallback-режиме без загруженной LLM. "
            "Поставь зависимости из `requirements.txt`, и я смогу отвечать через локальную модель.\n\n"
            f"Я услышала: {last_user}"
        )


def load_chat_model(config: ModelConfig) -> ChatModel:
    if os.getenv("WAIFU_FALLBACK", "0") == "1":
        return RuleBasedFallbackModel()
    return TransformersChatModel(config)
