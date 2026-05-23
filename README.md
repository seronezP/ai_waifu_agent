# AI Locale Waifu-Agent - "Rany"

A local CLI agent with personality, long-term memory, and a framework for LoRA fine-tuning a pre-trained instruct model.

By default, the project uses a locally downloaded `Qwen/Qwen2.5-1.5B-Instruct` model located in `models/qwen2.5-1.5b-instruct`.

### Requirements

Before running the agent, you need to:

* Install Python 3.9
* Create a virtual environment (`venv`)
* Install dependencies
* Download the `Qwen/Qwen2.5-1.5B-Instruct` model

Recommended system requirements:

* **CPU**: Ryzen 3 2200g (or equivalent)
* **GPU**: GTX 1050ti 4GB (or equivalent)
* **RAM**: 8GB
* **SSD**: ~4GB of free space

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/seronezP/ai_waifu_agent.git
cd ai_waifu_agent

```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### Model Installation

The project relies on a local model: `Qwen/Qwen2.5-1.5B-Instruct`.

It can be downloaded automatically during the first launch:

```bash
python main.py

```

Alternatively, you can download it manually via Hugging Face:

[https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

After downloading, place the model files into the following directory:

```
models/qwen2.5-1.5b-instruct

```

## Chat Commands

* `/exit` - Exit the application.
* `/memory` - Show recent facts stored in memory.
* `/profile` - Display the character's current profile.
* `/remember <text>` - Manually save a fact to memory.
* `/reset-session` - Clear the short-term context of the current session.

## Voice Mode

### Push-to-Talk Mode:

```bash
python main.py --voice

```

In this mode, press `e`, speak your phrase, and wait for the response. Press `q` to exit.

### Hands-Free Mode (with wake-word `рэни/rany`):

```bash
python main.py --hands-free

```

### Useful Settings:

```bash
python main.py --voice --listen-seconds 7
python main.py --hands-free --wake-word рэни --wake-word rany
python main.py --voice --no-speak

```

Speech recognition is powered locally by `openai/whisper-tiny`. On macOS, text-to-speech (TTS) voice output utilizes the system's native `say` command.

### Choosing a Silero Voice

Preview available voice candidates:

```bash
python scripts/preview_silero_voices.py

```

Preview a specific voice:

```bash
python scripts/preview_silero_voices.py --speaker baya
python scripts/preview_silero_voices.py --speaker kseniya
python scripts/preview_silero_voices.py --speaker xenia

```

Launch Rany using Silero TTS:

```bash
WAIFU_TTS=silero WAIFU_SILERO_SPEAKER=baya python main.py --voice

```

## Fine-Tuning

1. Prepare your dataset in JSONL format:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

```

2. Start the LoRA fine-tuning process:

```bash
python scripts/train_lora.py --dataset data/train_waifu.jsonl --output-dir models/waifu-lora

```

3. Launch the agent with the trained adapter:

```bash
WAIFU_LORA_PATH=models/waifu-lora python main.py

```

## Project Structure

* `waifu_agent/agent.py` - Prompt assembly, memory handling, and response loop.
* `waifu_agent/model.py` - Core model and LoRA adapter loading logic.
* `waifu_agent/memory.py` - SQLite-based memory storage with search capabilities.
* `waifu_agent/persona.py` - Character traits and system prompt configurations.
* `scripts/train_lora.py` - Script for LoRA fine-tuning.
* `data/persona.json` - Editable profile file for character customization.
