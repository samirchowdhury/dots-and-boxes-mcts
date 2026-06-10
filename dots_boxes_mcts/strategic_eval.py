from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from dots_boxes_mcts.game import (
    GameState,
    apply_move,
    box_edge_ids,
    legal_moves,
    new_game,
    state_snapshot,
)

PerspectiveSelector = int | Callable[[dict], int]


def summarize_strategic_records(
    records: list[dict],
    *,
    perspective_player: PerspectiveSelector,
) -> dict:
    total = empty_strategic_totals()
    for record in records:
        player = (
            perspective_player(record)
            if callable(perspective_player)
            else perspective_player
        )
        total = add_totals(
            total,
            analyze_record_strategy(record, perspective_player=int(player)),
        )
    return strategic_rates(total, games=len(records))


def analyze_record_strategy(record: dict, *, perspective_player: int) -> dict:
    if perspective_player not in {0, 1}:
        raise ValueError("perspective_player must be 0 or 1")
    if "rows" not in record or "cols" not in record or "moves" not in record:
        return empty_strategic_totals()

    state = new_game(rows=int(record["rows"]), cols=int(record["cols"]))
    totals = empty_strategic_totals()
    for move in record["moves"]:
        player = state.current_player
        profile = legal_move_profile(state)
        next_state = apply_move(state, move)
        scored_boxes = len(next_state.history[-1]["boxes"])
        new_three_sided_boxes = new_three_sided_box_count(state, next_state)

        if player == perspective_player:
            totals["moves"] += 1
            if scored_boxes:
                totals["scoringMoves"] += 1
                totals["capturedBoxes"] += scored_boxes
            else:
                totals["nonScoringMoves"] += 1
                if profile["safeMoves"]:
                    totals["safeMoveAvailableTurns"] += 1
                if new_three_sided_boxes:
                    totals["openerMoves"] += 1
                    totals["openedThreeSidedBoxes"] += new_three_sided_boxes
                    if profile["safeMoves"]:
                        totals["unsafeOpenerMoves"] += 1
                        totals["unsafeOpenedThreeSidedBoxes"] += new_three_sided_boxes
                    else:
                        totals["forcedOpenerMoves"] += 1

        state = next_state
    return totals


def extract_unsafe_opener_positions(
    records: Iterable[dict],
    *,
    perspective_player: PerspectiveSelector | None = None,
) -> list[dict]:
    selector = perspective_player or infer_perspective_player
    positions: list[dict] = []
    for record_index, record in enumerate(records):
        player = (
            selector(record)
            if callable(selector)
            else selector
        )
        positions.extend(
            record_unsafe_opener_positions(
                record,
                perspective_player=int(player),
                record_index=record_index,
            )
        )
    return positions


def record_unsafe_opener_positions(
    record: dict,
    *,
    perspective_player: int,
    record_index: int | None = None,
) -> list[dict]:
    state = new_game(rows=int(record["rows"]), cols=int(record["cols"]))
    positions: list[dict] = []
    for ply, move in enumerate(record["moves"]):
        player = state.current_player
        profile = legal_move_profile(state)
        next_state = apply_move(state, move)
        scored_boxes = len(next_state.history[-1]["boxes"])
        new_three_sided_boxes = new_three_sided_box_count(state, next_state)

        if (
            player == perspective_player
            and not scored_boxes
            and new_three_sided_boxes
            and profile["safeMoves"]
        ):
            position = {
                "recordIndex": record_index,
                "ply": ply,
                "player": player,
                "move": move,
                "newThreeSidedBoxes": new_three_sided_boxes,
                "safeMoves": profile["safeMoves"],
                "scoringMoves": profile["scoringMoves"],
                "openerMoves": profile["openerMoves"],
                "state": state_snapshot(state),
                "finalScores": record.get("finalScores"),
                "winner": record.get("winner"),
            }
            for key in (
                "bot",
                "checkpoint",
                "candidateCheckpoint",
                "baselineCheckpoint",
                "source",
                "opponent",
                "ourPlayer",
                "candidatePlayer",
                "_path",
                "_line",
            ):
                if key in record:
                    position[key] = record[key]
            positions.append(position)

        state = next_state
    return positions


def legal_move_profile(state: GameState) -> dict:
    profile = {"safeMoves": 0, "scoringMoves": 0, "openerMoves": 0}
    for move in legal_moves(state):
        next_state = apply_move(state, move)
        scored_boxes = len(next_state.history[-1]["boxes"])
        if scored_boxes:
            profile["scoringMoves"] += 1
        elif new_three_sided_box_count(state, next_state) == 0:
            profile["safeMoves"] += 1
        else:
            profile["openerMoves"] += 1
    return profile


def new_three_sided_box_count(before: GameState, after: GameState) -> int:
    count = 0
    for row in range(before.rows - 1):
        for col in range(before.cols - 1):
            if before.boxes[row][col] is not None or after.boxes[row][col] is not None:
                continue
            if box_edge_count(before, row=row, col=col) < 3 and box_edge_count(after, row=row, col=col) == 3:
                count += 1
    return count


def box_edge_count(state: GameState, *, row: int, col: int) -> int:
    return sum(1 for edge in box_edge_ids(row, col) if edge in state.edges)


def empty_strategic_totals() -> dict[str, int]:
    return {
        "moves": 0,
        "scoringMoves": 0,
        "capturedBoxes": 0,
        "nonScoringMoves": 0,
        "safeMoveAvailableTurns": 0,
        "openerMoves": 0,
        "openedThreeSidedBoxes": 0,
        "unsafeOpenerMoves": 0,
        "unsafeOpenedThreeSidedBoxes": 0,
        "forcedOpenerMoves": 0,
    }


def add_totals(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: left.get(key, 0) + right.get(key, 0) for key in empty_strategic_totals()}


def strategic_rates(totals: dict[str, int], *, games: int) -> dict:
    non_scoring = totals["nonScoringMoves"]
    moves = totals["moves"]
    safe_turns = totals["safeMoveAvailableTurns"]
    summary: dict[str, Any] = dict(totals)
    summary["unsafeOpenerRate"] = (
        totals["unsafeOpenerMoves"] / non_scoring
        if non_scoring
        else 0.0
    )
    summary["unsafeOpenerPerGame"] = (
        totals["unsafeOpenerMoves"] / games
        if games
        else 0.0
    )
    summary["unsafeOpenedThreeSidedBoxesPerGame"] = (
        totals["unsafeOpenedThreeSidedBoxes"] / games
        if games
        else 0.0
    )
    summary["safeMoveAvailableTurnRate"] = (
        safe_turns / moves
        if moves
        else 0.0
    )
    return summary


def infer_perspective_player(record: dict) -> int:
    for key in ("ourPlayer", "candidatePlayer", "guidedPlayer", "mctsPlayer"):
        if key in record:
            return int(record[key])
    return 0


def load_jsonl_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record["_path"] = str(path)
            record["_line"] = line_number
            records.append(record)
    return records


def grouped_strategic_summaries(records: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(strategic_group_key(record), []).append(record)
    return {
        key: {
            "records": len(group_records),
            "strategic": summarize_strategic_records(
                group_records,
                perspective_player=infer_perspective_player,
            ),
        }
        for key, group_records in sorted(groups.items())
    }


def strategic_group_key(record: dict) -> str:
    for key in ("checkpoint", "candidateCheckpoint", "bot"):
        if key in record:
            return str(record[key])
    if "_path" in record:
        return str(record["_path"])
    return "unknown"


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            output.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Dots and Boxes strategic safety metrics from replay JSONL files."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--suite-out", type=Path)
    args = parser.parse_args()

    records = load_jsonl_records(args.inputs)
    summary = summarize_strategic_records(records, perspective_player=infer_perspective_player)
    positions = extract_unsafe_opener_positions(records)
    result = {
        "records": len(records),
        "strategic": summary,
        "groups": grouped_strategic_summaries(records),
        "unsafeOpenerPositions": len(positions),
    }

    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf8",
        )
    if args.suite_out is not None:
        write_jsonl(positions, args.suite_out)

    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
