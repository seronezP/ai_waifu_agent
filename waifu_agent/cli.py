from __future__ import annotations

import argparse
import sys
import termios
import tty
from pathlib import Path

from waifu_agent.agent import AgentConfig, WaifuAgent
from waifu_agent.voice import VoiceConfig, VoiceIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local AI waifu agent.")
    parser.add_argument("--voice", action="store_true", help="Start push-to-talk voice mode.")
    parser.add_argument("--hands-free", action="store_true", help="Continuously listen for wake words.")
    parser.add_argument("--wake-word", action="append", default=None, help="Wake word for hands-free mode.")
    parser.add_argument("--listen-seconds", type=float, default=5.0, help="Seconds to record per voice turn.")
    parser.add_argument("--no-speak", action="store_true", help="Disable text-to-speech output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    try:
        agent = WaifuAgent(AgentConfig(root=root))
    except RuntimeError as exc:
        print(str(exc))
        print("Можно временно запустить fallback: WAIFU_FALLBACK=1 python main.py")
        raise SystemExit(1) from exc

    if args.voice or args.hands_free:
        voice = VoiceIO(
            VoiceConfig(
                listen_seconds=args.listen_seconds,
                wake_words=args.wake_word or ["рэни", "рени", "rany", "rani"],
                speak_enabled=not args.no_speak,
            )
        )
        run_voice_chat(agent, voice, hands_free=args.hands_free)
        return

    run_text_chat(agent)


def run_text_chat(agent: WaifuAgent) -> None:
    print(f"{agent.persona.name}: я готова. Напиши /exit для выхода.")
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            return
        if user_text == "/memory":
            for item in agent.memory.recent(20):
                print(f"#{item.id} [{item.kind}] {item.content}")
            continue
        if user_text == "/profile":
            print(agent.persona.system_prompt())
            continue
        if user_text == "/reset-session":
            agent.reset_session()
            print("session cleared")
            continue
        if user_text.startswith("/remember "):
            result = agent.tools.remember(user_text.removeprefix("/remember "))
            print(result.content)
            continue

        answer = agent.reply(user_text)
        print(f"{agent.persona.name}> {answer}")
        sys.stdout.flush()


def run_voice_chat(agent: WaifuAgent, voice: VoiceIO, hands_free: bool) -> None:
    wake_words = ", ".join(voice.config.wake_words)
    if hands_free:
        print(f"{agent.persona.name}: hands-free режим. Wake words: {wake_words}. Ctrl+C для выхода.")
    else:
        print(f"{agent.persona.name}: голосовой режим. Нажимай e, говори, потом жди ответ. q - выход.")

    while True:
        try:
            if hands_free:
                print("listening...")
                spoken = voice.listen()
                if not spoken:
                    continue
                print(f"heard> {spoken}")
                if not voice.has_wake_word(spoken):
                    continue
                user_text = voice.strip_wake_word(spoken).strip()
                if not user_text:
                    voice.say("Я слушаю.")
                    user_text = voice.listen()
            else:
                print("press e to talk, q to exit> ", end="", flush=True)
                command = read_single_key().lower()
                print(command)
                if command == "q":
                    return
                if command != "e":
                    continue
                spoken = voice.listen()
                if not spoken:
                    print("Не расслышала.")
                    continue
                print(f"you> {spoken}")
                user_text = spoken

            if user_text.lower() in {"/exit", "/quit", "выход", "стоп"}:
                return
            answer = agent.reply(user_text)
            print(f"{agent.persona.name}> {answer}")
            voice.say(answer)
        except (EOFError, KeyboardInterrupt):
            print()
            return


def read_single_key() -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
