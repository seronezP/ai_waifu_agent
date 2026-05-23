# AI Agent Waifu Rany

Локальный CLI-агент с характером, долговременной памятью и базой для LoRA-дообучения готовой instruct-модели.

По умолчанию проект использует локально скачанную `Qwen/Qwen2.5-1.5B-Instruct` из `models/qwen2.5-1.5b-instruct`: она заметно стабильнее держит русский язык и характер.

## Быстрый старт

```bash
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Если модель еще не скачана, `transformers` скачает ее из Hugging Face при первом запуске.

## Команды в чате

- `/exit` - выйти.
- `/memory` - показать последние факты из памяти.
- `/profile` - показать текущий профиль персонажа.
- `/remember текст` - вручную сохранить факт.
- `/reset-session` - очистить краткосрочный контекст текущего запуска.

## Голосовой режим

Push-to-talk режим:

```bash
python main.py --voice
```

В этом режиме нажми `e`, скажи фразу и дождись ответа. Для выхода нажми `q`.

Hands-free режим с wake-word `рэни/rany`:

```bash
python main.py --hands-free
```

Полезные настройки:

```bash
python main.py --voice --listen-seconds 7
python main.py --hands-free --wake-word рэни --wake-word rany
python main.py --voice --no-speak
```

Распознавание речи использует локальную `openai/whisper-tiny`, а озвучка на macOS идет через системную команду `say`.

### Подбор голоса Silero

Прослушать кандидатов:

```bash
python scripts/preview_silero_voices.py
```

Прослушать конкретный голос:

```bash
python scripts/preview_silero_voices.py --speaker baya
python scripts/preview_silero_voices.py --speaker kseniya
python scripts/preview_silero_voices.py --speaker xenia
```

Запустить Рэни с Silero:

```bash
WAIFU_TTS=silero WAIFU_SILERO_SPEAKER=baya python main.py --voice
```

## Дообучение

1. Подготовь датасет в JSONL:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

2. Запусти LoRA fine-tuning:

```bash
python scripts/train_lora.py --dataset data/train_waifu.jsonl --output-dir models/waifu-lora
```

3. Запусти агента с адаптером:

```bash
WAIFU_LORA_PATH=models/waifu-lora python main.py
```

## Структура

- `waifu_agent/agent.py` - сборка промпта, память, цикл ответа.
- `waifu_agent/model.py` - загрузка готовой модели и LoRA-адаптера.
- `waifu_agent/memory.py` - SQLite-память с поиском.
- `waifu_agent/persona.py` - характер и системный промпт.
- `scripts/train_lora.py` - дообучение LoRA.
- `data/persona.json` - редактируемая персона.
