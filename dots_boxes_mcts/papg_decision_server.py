from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from dots_boxes_mcts.az_mcts import NetworkEvaluator, NetworkGuidedMCTS
from dots_boxes_mcts.external_games import append_jsonl, external_game_record
from dots_boxes_mcts.fast_mcts import FastUCTMCTS
from dots_boxes_mcts.game import apply_move, new_game, state_snapshot
from dots_boxes_mcts.mcts import UCTMCTS, result_payload
from dots_boxes_mcts.papg_common import checkpoint_bot_name, infer_papg_reply

_EVALUATORS: dict[tuple[str, str], NetworkEvaluator] = {}


def decision_response(payload: dict[str, Any]) -> dict:
    rows = int(payload["rows"])
    cols = int(payload["cols"])
    simulations = int(payload["simulations"])
    our_player = int(payload.get("our_player", 0))
    if our_player not in {0, 1}:
        raise ValueError("our_player must be 0 or 1")
    checkpoint = payload.get("checkpoint")
    moves = list(payload["moves"])
    state = new_game(rows=rows, cols=cols)
    for move in moves:
        state = apply_move(state, move)

    missing = [edge for edge in payload["drawn_edges"] if edge not in state.edges]
    if missing:
        for move in infer_papg_reply(state, missing, papg_player=1 - our_player):
            moves.append(move)
            state = apply_move(state, move)

    if payload.get("write_record"):
        return write_record_response(
            payload=payload,
            moves=moves,
            checkpoint=checkpoint,
            simulations=simulations,
            rows=rows,
            cols=cols,
        )

    if state.terminal:
        return {
            "moves": moves,
            "terminal": True,
            "scores": list(state.scores),
            "winner": state.winner,
        }

    if state.current_player != our_player:
        raise ValueError(
            f"Synced state is player {state.current_player}'s turn, "
            f"but this worker controls player {our_player}."
        )

    searcher = searcher_from_payload(
        payload=payload,
        checkpoint=checkpoint,
        simulations=simulations,
        seed=int(payload["seed"]) + len(moves),
    )
    result = searcher.search(state)
    return {
        "moves": moves,
        "terminal": False,
        "move": result.move,
        "search": result_payload(result),
        "decision": {
            "turn": len(moves),
            "player": state.current_player,
            "ourPlayer": our_player,
            "state": state_snapshot(state),
            "search": result_payload(result),
        },
    }


def write_record_response(
    *,
    payload: dict[str, Any],
    moves: list[str],
    checkpoint: str | None,
    simulations: int,
    rows: int,
    cols: int,
) -> dict:
    bot = bot_name(checkpoint=checkpoint, simulations=simulations)
    our_player = int(payload.get("our_player", 0))
    record = external_game_record(
        source="papg",
        opponent="papg",
        bot=bot,
        rows=rows,
        cols=cols,
        moves=moves,
        our_player=our_player,
        notes=(
            "Browser-backed Papg game with network-guided MCTS."
            if checkpoint
            else "Browser-backed Stage 2.5 Papg game."
        ),
    )
    record["seed"] = payload["seed"]
    record["simulations"] = simulations
    record["decisions"] = payload.get("decisions", [])
    if checkpoint:
        record["checkpoint"] = checkpoint
        record["cPuct"] = payload["cPuct"]
        record["mlxDevice"] = payload["mlxDevice"]
    append_jsonl(record, Path(payload["out"]))
    return {
        "moves": moves,
        "terminal": record["terminal"],
        "scores": record["finalScores"],
        "winner": record["winner"],
        "finalScores": record["finalScores"],
        "record": record,
    }


def searcher_from_payload(
    *,
    payload: dict[str, Any],
    checkpoint: str | None,
    simulations: int,
    seed: int,
):
    if checkpoint:
        device = payload["mlxDevice"]
        evaluator = cached_evaluator(checkpoint=checkpoint, device=device)
        return NetworkGuidedMCTS(
            evaluator=evaluator,
            simulations=simulations,
            c_puct=float(payload["cPuct"]),
            seed=seed,
        )
    backend = payload.get("backend", "numba")
    if backend == "numba":
        return FastUCTMCTS(simulations=simulations, seed=seed)
    if backend == "python":
        return UCTMCTS(simulations=simulations, seed=seed)
    raise ValueError(f"Unknown MCTS backend: {backend}")


def cached_evaluator(*, checkpoint: str, device: str) -> NetworkEvaluator:
    key = (checkpoint, device)
    if key not in _EVALUATORS:
        _EVALUATORS[key] = NetworkEvaluator(checkpoint=Path(checkpoint), device=device)
    return _EVALUATORS[key]


def bot_name(*, checkpoint: str | None, simulations: int) -> str:
    if checkpoint:
        return checkpoint_bot_name(checkpoint=Path(checkpoint), simulations=simulations)
    return f"uct_mcts_{simulations}"


class PapgDecisionHandler(BaseHTTPRequestHandler):
    server_version = "PapgDecisionServer/0.1"

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        self.write_json({"ok": True})

    def do_POST(self) -> None:
        if self.path != "/decision":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf8"))
            self.write_json(decision_response(payload))
        except Exception as error:
            self.send_response(500)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(error)}).encode("utf8"))

    def log_message(self, format: str, *args) -> None:
        return

    def write_json(self, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve PAPG browser move decisions over localhost.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--stdio", action="store_true", help="Read JSON payloads from stdin and write JSON lines.")
    parser.add_argument("--file-bridge", type=Path, help="Directory for request-*.json / response-*.json files.")
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    args = parser.parse_args()

    if args.stdio:
        serve_stdio()
        return
    if args.file_bridge is not None:
        serve_file_bridge(args.file_bridge, poll_seconds=args.poll_seconds)
        return

    server = ThreadingHTTPServer((args.host, args.port), PapgDecisionHandler)
    print(f"Papg decision server listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def serve_stdio() -> None:
    print("Papg decision worker ready", flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            response = decision_response(json.loads(line))
        except Exception as error:
            response = {"error": str(error)}
        print(json.dumps(response, separators=(",", ":")), flush=True)


def serve_file_bridge(directory: Path, *, poll_seconds: float = 0.05) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Papg decision file bridge ready at {directory}", flush=True)
    while True:
        for request_path in sorted(directory.glob("request-*.json")):
            request_id = request_path.stem.removeprefix("request-")
            response_path = directory / f"response-{request_id}.json"
            if response_path.exists():
                continue
            try:
                payload = json.loads(request_path.read_text(encoding="utf8"))
                response = decision_response(payload)
            except Exception as error:
                response = {"error": str(error)}
            tmp_path = directory / f".response-{request_id}.tmp"
            tmp_path.write_text(json.dumps(response, separators=(",", ":")), encoding="utf8")
            tmp_path.replace(response_path)
            try:
                request_path.unlink()
            except FileNotFoundError:
                pass
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
