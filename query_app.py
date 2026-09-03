import argparse
import json
import sys

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_PROMPT = "Расскажи сказку на 500 токенов"


def read_health(base_url: str, timeout: float) -> None:
    response = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
    response.raise_for_status()

    try:
        payload = json.dumps(response.json(), ensure_ascii=False)
    except requests.JSONDecodeError:
        payload = response.text

    print(f"health: {payload}")


def stream_prompt(base_url: str, prompt: str, timeout: float) -> None:
    url = f"{base_url.rstrip('/')}/stream"
    with requests.get(
        url,
        params={"prompt": prompt},
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()

        for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
            if chunk:
                print(chunk, end="", flush=True)
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the FastAPI streaming app.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_PROMPT,
        help="Prompt to send to /stream.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"FastAPI base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Do not call /health before /stream.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if not args.skip_health:
            read_health(args.base_url, args.timeout)
        stream_prompt(args.base_url, args.prompt, args.timeout)
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response is not None else ""
        print(f"HTTP error: {exc} {body}", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
