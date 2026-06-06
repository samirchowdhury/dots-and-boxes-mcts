from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_boxes_mcts.game import GameState, apply_move, new_game, state_snapshot


def papg_index_to_edge(index: int, rows: int, cols: int) -> str:
    if rows < 2 or cols < 2:
        raise ValueError("Papg index conversion needs at least a 2x2 dot grid.")
    if index < 0:
        raise ValueError(f"Unknown Papg move index: {index}")

    table_width = cols * 2 - 1
    table_height = rows * 2 - 1
    table_row, table_col = divmod(index, table_width)

    if table_row >= table_height:
        raise ValueError(f"Unknown Papg {rows}x{cols} move index: {index}")

    if table_row % 2 == 0 and table_col % 2 == 1:
        return f"h:{table_row // 2}:{(table_col - 1) // 2}"

    if table_row % 2 == 1 and table_col % 2 == 0:
        return f"v:{(table_row - 1) // 2}:{table_col // 2}"

    raise ValueError(f"Papg index {index} is a dot or box cell, not an edge.")


def papg_indexes_to_edges(indexes: list[int], rows: int = 3, cols: int = 3) -> list[str]:
    return [papg_index_to_edge(index, rows=rows, cols=cols) for index in indexes]


def edge_to_papg_index(edge_id: str, rows: int, cols: int) -> int:
    kind, raw_row, raw_col = edge_id.split(":")
    row = int(raw_row)
    col = int(raw_col)
    table_width = cols * 2 - 1

    if kind == "h":
        return (row * 2) * table_width + col * 2 + 1
    if kind == "v":
        return (row * 2 + 1) * table_width + col * 2

    raise ValueError(f"Invalid edge id: {edge_id}")


def external_game_record(
    *,
    source: str,
    opponent: str,
    bot: str,
    rows: int,
    cols: int,
    moves: list[str],
    our_player: int = 0,
    notes: str | None = None,
) -> dict:
    if our_player not in {0, 1}:
        raise ValueError("our_player must be 0 or 1.")

    state = replay_moves(rows=rows, cols=cols, moves=moves)
    opponent_player = 1 if our_player == 0 else 0
    record = {
        "source": source,
        "opponent": opponent,
        "bot": bot,
        "rows": state.rows,
        "cols": state.cols,
        "players": {
            str(our_player): bot,
            str(opponent_player): opponent,
        },
        "ourPlayer": our_player,
        "moves": moves,
        "finalScores": [state.scores[0], state.scores[1]],
        "winner": state.winner,
        "terminal": state.terminal,
        "state": state_snapshot(state),
    }
    if notes:
        record["notes"] = notes
    return record


def replay_moves(*, rows: int, cols: int, moves: list[str]) -> GameState:
    state = new_game(rows=rows, cols=cols)
    for move in moves:
        state = apply_move(state, move)
    return state


def append_jsonl(record: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf8") as output:
        output.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
        output.write("\n")


def parse_moves(raw_moves: list[str], papg_indexes: bool, rows: int, cols: int) -> list[str]:
    if not raw_moves:
        raise ValueError("Provide at least one move.")

    if papg_indexes:
        return papg_indexes_to_edges([int(move) for move in raw_moves], rows=rows, cols=cols)

    return raw_moves


def main() -> None:
    parser = argparse.ArgumentParser(description="Record an externally played Dots and Boxes game.")
    parser.add_argument("--source", default="papg")
    parser.add_argument("--opponent", default="papg")
    parser.add_argument("--bot", required=True, help="Name of the local strategy that played the game.")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--our-player", type=int, choices=[0, 1], default=0)
    parser.add_argument("--papg-indexes", action="store_true", help="Interpret moves as Papg URL move indexes.")
    parser.add_argument("--notes")
    parser.add_argument("--out", type=Path, default=Path("runs/papg/stage-2/papg-4x4-games.jsonl"))
    parser.add_argument("moves", nargs="+", help="Edge ids, or Papg URL move indexes with --papg-indexes.")
    args = parser.parse_args()

    try:
        moves = parse_moves(
            raw_moves=args.moves,
            papg_indexes=args.papg_indexes,
            rows=args.rows,
            cols=args.cols,
        )
        record = external_game_record(
            source=args.source,
            opponent=args.opponent,
            bot=args.bot,
            rows=args.rows,
            cols=args.cols,
            moves=moves,
            our_player=args.our_player,
            notes=args.notes,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    append_jsonl(record, args.out)
    print(f"Wrote 1 game to {args.out}")


if __name__ == "__main__":
    main()
