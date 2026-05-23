from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class VoiceConfig:
    stt_model: str = os.getenv("WAIFU_STT_MODEL", "openai/whisper-tiny")
    sample_rate: int = int(os.getenv("WAIFU_VOICE_SAMPLE_RATE", "16000"))
    listen_seconds: float = float(os.getenv("WAIFU_LISTEN_SECONDS", "5"))
    wake_words: list[str] = field(default_factory=lambda: ["рэни", "рени", "rany", "rani"])
    speak_enabled: bool = os.getenv("WAIFU_NO_SPEAK", "0") != "1"
    tts_engine: str = os.getenv("WAIFU_TTS", "say")
    say_voice: Optional[str] = os.getenv("WAIFU_SAY_VOICE") or None
    silero_speaker: str = os.getenv("WAIFU_SILERO_SPEAKER", "baya")
    silero_sample_rate: int = int(os.getenv("WAIFU_SILERO_SAMPLE_RATE", "48000"))


class VoiceIO:
    def __init__(self, config: VoiceConfig) -> None:
        self.config = config
        self._processor = None
        self._model = None
        self._silero_model = None

    def listen(self) -> str:
        try:
            import numpy as np
            import sounddevice as sd
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Для голосового режима установи зависимости: pip install -r requirements.txt"
            ) from exc

        if self._processor is None or self._model is None:
            self._processor = WhisperProcessor.from_pretrained(self.config.stt_model)
            self._model = WhisperForConditionalGeneration.from_pretrained(self.config.stt_model)

        frames = int(self.config.listen_seconds * self.config.sample_rate)
        audio = sd.rec(frames, samplerate=self.config.sample_rate, channels=1, dtype="float32")
        sd.wait()
        waveform = np.squeeze(audio)
        if float(np.max(np.abs(waveform))) < 0.01:
            return ""

        inputs = self._processor(
            waveform,
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
        )
        predicted_ids = self._model.generate(
            inputs.input_features,
            max_new_tokens=96,
            language="russian",
            task="transcribe",
        )
        text = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return " ".join(text.strip().split())

    def say(self, text: str) -> None:
        if not self.config.speak_enabled or not text.strip():
            return
        if self.config.tts_engine == "silero":
            self._say_silero(text)
            return
        command = ["say"]
        if self.config.say_voice:
            command.extend(["-v", self.config.say_voice])
        command.append(text)
        try:
            subprocess.run(command, check=False)
        except FileNotFoundError:
            print("(TTS недоступен: команда say не найдена)")

    def _say_silero(self, text: str) -> None:
        try:
            import sounddevice as sd
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Для Silero TTS нужны torch и sounddevice. Выполни: pip install -r requirements.txt"
            ) from exc

        if self._silero_model is None:
            self._silero_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="ru",
                speaker="v4_ru",
                trust_repo=True,
            )
        audio = self._silero_model.apply_tts(
            text=text,
            speaker=self.config.silero_speaker,
            sample_rate=self.config.silero_sample_rate,
        )
        sd.play(audio, self.config.silero_sample_rate)
        sd.wait()

    def has_wake_word(self, text: str) -> bool:
        normalized = self._normalize(text)
        return any(self._normalize(word) in normalized for word in self.config.wake_words)

    def strip_wake_word(self, text: str) -> str:
        cleaned = text
        for word in self.config.wake_words:
            cleaned = re.sub(re.escape(word), "", cleaned, flags=re.IGNORECASE)
        return " ".join(cleaned.strip(" ,.!?-").split())

    @staticmethod
    def _normalize(text: str) -> str:
        lowered = text.lower().replace("ё", "е")
        return re.sub(r"[^a-zа-я0-9]+", " ", lowered).strip()
