from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from dots_boxes_mcts.game import apply_move, new_game, state_snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
STATIC_DIR = Path(__file__).with_name("viewer_static")


def list_game_files(runs_dir: Path = RUNS_DIR) -> list[str]:
    if not runs_dir.exists():
        return []

    return sorted(
        path.relative_to(runs_dir).as_posix()
        for path in runs_dir.rglob("*.jsonl")
        if path.is_file()
    )


def resolve_game_file(file_name: str, runs_dir: Path = RUNS_DIR) -> Path:
    if not file_name:
        raise ValueError("Choose a JSONL file.")

    runs_root = runs_dir.resolve()
    requested_path = (runs_root / file_name).resolve()

    if runs_root != requested_path and runs_root not in requested_path.parents:
        raise ValueError("File must be inside runs/.")
    if requested_path.suffix != ".jsonl":
        raise ValueError("File must be a .jsonl file.")
    if not requested_path.is_file():
        raise FileNotFoundError(file_name)

    return requested_path


def load_game_record(file_name: str, line_number: int, runs_dir: Path = RUNS_DIR) -> dict:
    if line_number < 1:
        raise ValueError("Line number must be at least 1.")

    game_path = resolve_game_file(file_name, runs_dir)

    with game_path.open("r", encoding="utf8") as game_file:
        for current_line_number, line in enumerate(game_file, start=1):
            if current_line_number == line_number:
                try:
                    return json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Line {line_number} is not valid JSON.") from error

    raise IndexError(f"{file_name} has fewer than {line_number} lines.")


def replay_record(record: dict) -> list[dict]:
    rows = int(record["rows"])
    cols = int(record["cols"])
    state = new_game(rows=rows, cols=cols)
    frames = [
        {
            "moveNumber": 0,
            "move": None,
            "historyEntry": None,
            "state": state_snapshot(state),
        }
    ]

    for move_number, move in enumerate(record["moves"], start=1):
        state = apply_move(state, move)
        frames.append(
            {
                "moveNumber": move_number,
                "move": move,
                "historyEntry": state.history[-1],
                "state": state_snapshot(state),
            }
        )

    return frames


def game_payload(file_name: str, line_number: int, runs_dir: Path = RUNS_DIR) -> dict:
    record = load_game_record(file_name, line_number, runs_dir)
    return {
        "file": file_name,
        "line": line_number,
        "record": record,
        "frames": replay_record(record),
    }


class ViewerHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)

        if parsed_url.path == "/api/files":
            self._send_json({"files": list_game_files()})
            return

        if parsed_url.path == "/api/game":
            self._handle_game_request(parsed_url.query)
            return

        self._handle_static_request(parsed_url.path)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _handle_game_request(self, query: str) -> None:
        params = parse_qs(query)
        file_name = params.get("file", [""])[0]
        raw_line = params.get("line", ["1"])[0]

        try:
            line_number = int(raw_line)
            payload = game_payload(file_name, line_number)
        except FileNotFoundError:
            self._send_error(HTTPStatus.NOT_FOUND, "That JSONL file was not found.")
            return
        except (IndexError, KeyError, TypeError, ValueError) as error:
            self._send_error(HTTPStatus.BAD_REQUEST, str(error))
            return

        self._send_json(payload)

    def _handle_static_request(self, request_path: str) -> None:
        if request_path in {"", "/"}:
            relative_path = Path("index.html")
        else:
            relative_path = Path(unquote(request_path.lstrip("/")))

        static_root = STATIC_DIR.resolve()
        static_path = (static_root / relative_path).resolve()

        if static_root != static_path and static_root not in static_path.parents:
            self._send_error(HTTPStatus.FORBIDDEN, "Path is outside static files.")
            return

        if not static_path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "Static file was not found.")
            return

        content_type = mimetypes.guess_type(static_path.name)[0] or "application/octet-stream"
        content = static_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), ViewerHandler)
    url = f"http://{host}:{port}"
    print(f"Dots and Boxes replay viewer running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped replay viewer.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Dots and Boxes replay viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

