from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dots_boxes_mcts.game import GameState, apply_move, legal_moves, new_game, state_snapshot


def play_random_game(rows: int = 5, cols: int = 5, seed: int = 1) -> dict:
    rng = random.Random(seed)
    state = new_game(rows=rows, cols=cols)
    moves: list[str] = []

    while not state.terminal:
        move = rng.choice(legal_moves(state))
        moves.append(move)
        state = apply_move(state, move)

    return game_record(state=state, moves=moves, seed=seed)


def generate_random_games(games: int, rows: int = 5, cols: int = 5, seed: int = 1) -> list[dict]:
    records: list[dict] = []
    for game_index in range(games):
        game_seed = seed + game_index
        record = play_random_game(rows=rows, cols=cols, seed=game_seed)
        record["gameIndex"] = game_index
        records.append(record)
    return records


def write_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf8") as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            output.write("\n")


def game_record(state: GameState, moves: list[str], seed: int) -> dict:
    return {
        "seed": seed,
        "rows": state.rows,
        "cols": state.cols,
        "moves": moves,
        "finalScores": [state.scores[0], state.scores[1]],
        "winner": state.winner,
        "terminal": state.terminal,
        "state": state_snapshot(state),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random Dots and Boxes self-play games.")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("runs/random-self-play.jsonl"))
    args = parser.parse_args()

    if args.games < 1:
        raise SystemExit("--games must be at least 1")

    records = generate_random_games(
        games=args.games,
        rows=args.rows,
        cols=args.cols,
        seed=args.seed,
    )
    write_jsonl(records, args.out)
    print(f"Wrote {len(records)} games to {args.out}")


if __name__ == "__main__":
    main()
