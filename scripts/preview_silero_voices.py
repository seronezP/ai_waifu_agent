from __future__ import annotations

import argparse


DEFAULT_TEXT = "Привет, я Рэни. Давай подберем мне голос: мягкий, женственный и живой."
DEFAULT_SPEAKERS = ["baya", "kseniya", "xenia", "aidar", "eugene"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview Russian Silero TTS voices.")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--speaker", action="append", default=None)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--list", action="store_true", help="Print known candidate speakers and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    speakers = args.speaker or DEFAULT_SPEAKERS
    if args.list:
        print("\n".join(speakers))
        return

    try:
        import sounddevice as sd
        import torch
    except ImportError as exc:
        raise SystemExit("Установи зависимости: pip install -r requirements.txt") from exc

    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v4_ru",
        trust_repo=True,
    )

    for speaker in speakers:
        print(f"speaker: {speaker}")
        audio = model.apply_tts(
            text=args.text,
            speaker=speaker,
            sample_rate=args.sample_rate,
        )
        sd.play(audio, args.sample_rate)
        sd.wait()


if __name__ == "__main__":
    main()
