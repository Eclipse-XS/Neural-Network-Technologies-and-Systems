from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

from openai import APIConnectionError, APIError

from .client import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    create_client,
    ensure_model_available,
    normalize_base_url,
    request_chat_completion,
    validate_generation_settings,
)

LAB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIALOGS_DIR = LAB_ROOT / "outputs" / "dialogs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive client for a local LM Studio model."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser


def write_transcript(
    history: Sequence[dict[str, str]],
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    output_dir: Path = DEFAULT_DIALOGS_DIR,
    timestamp: datetime | None = None,
) -> Path | None:
    turns = [message for message in history if message["role"] in {"user", "assistant"}]
    if not turns:
        return None

    recorded_at = timestamp or datetime.now().astimezone()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"dialog_{recorded_at:%Y%m%d_%H%M%S_%f}.md"
    lines = [
        "# Dialogue",
        "",
        "## Configuration",
        f"- Timestamp: {recorded_at.isoformat(timespec='seconds')}",
        f"- Base URL: {base_url}",
        f"- Model: {model}",
        f"- Temperature: {temperature}",
        f"- Max tokens: {max_tokens}",
        f"- System prompt: {system_prompt or '(none)'}",
    ]
    for message in turns:
        heading = "User" if message["role"] == "user" else "Assistant"
        lines.extend(["", f"## {heading}", "", message["content"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_chat(args: argparse.Namespace) -> int:
    validate_generation_settings(args.model, args.temperature, args.max_tokens)
    base_url = normalize_base_url(args.base_url)
    client = create_client(base_url)

    try:
        ensure_model_available(client, args.model)
    except APIConnectionError:
        print(
            f"Не вдалося підключитися до LM Studio API за {base_url}. "
            "Переконайтеся, що LM Studio запущений і Local Model API має статус Running."
        )
        return 1
    except APIError as error:
        print(f"LM Studio API відхилив перевірку моделей: {error}")
        return 1
    except ValueError as error:
        print(error)
        return 1

    print(f"LM Studio API: {base_url}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print("Type 'exit' to finish.")

    history: list[dict[str, str]] = []
    if args.system_prompt.strip():
        history.append({"role": "system", "content": args.system_prompt.strip()})

    try:
        while True:
            try:
                prompt = input("\nYou: ").strip()
            except EOFError:
                print()
                break
            if prompt.lower() in {"exit", "quit"}:
                break
            if not prompt:
                continue

            pending_history = [*history, {"role": "user", "content": prompt}]
            try:
                answer = request_chat_completion(
                    client,
                    model=args.model,
                    messages=pending_history,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except APIConnectionError:
                print(
                    f"Не вдалося підключитися до LM Studio API за {base_url}. "
                    "Переконайтеся, що Local Model API має статус Running."
                )
                continue
            except APIError as error:
                print(f"LM Studio API не виконав запит: {error}")
                continue
            except RuntimeError as error:
                print(error)
                continue

            history.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            )
            print(f"\nAssistant: {answer}")
    except KeyboardInterrupt:
        print("\nСесію завершено.")
    finally:
        transcript = write_transcript(
            history,
            base_url=base_url,
            model=args.model,
            system_prompt=args.system_prompt.strip(),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        if transcript is not None:
            print(f"Transcript saved to: {transcript}")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_chat(args)
    except ValueError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
