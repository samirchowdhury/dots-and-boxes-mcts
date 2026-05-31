from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

DEFAULT_ROWS = 5
DEFAULT_COLS = 5
Winner = int | Literal["draw"] | None


@dataclass(frozen=True)
class GameState:
    rows: int
    cols: int
    current_player: int = 0
    edges: frozenset[str] = field(default_factory=frozenset)
    edge_owners: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    boxes: tuple[tuple[int | None, ...], ...] = field(default_factory=tuple)
    scores: tuple[int, int] = (0, 0)
    history: tuple[dict, ...] = field(default_factory=tuple)
    terminal: bool = False
    winner: Winner = None


def new_game(rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> GameState:
    if not isinstance(rows, int) or not isinstance(cols, int) or rows < 2 or cols < 2:
        raise ValueError("Dots and Boxes needs at least a 2x2 dot grid.")

    boxes = tuple(tuple(None for _ in range(cols - 1)) for _ in range(rows - 1))
    return GameState(rows=rows, cols=cols, boxes=boxes)


def legal_moves(state: GameState) -> list[str]:
    if state.terminal:
        return []
    return [edge for edge in all_edge_ids(state.rows, state.cols) if edge not in state.edges]


def apply_move(state: GameState, edge_id_value: str) -> GameState:
    if state.terminal:
        raise ValueError("Cannot move after the game is over.")

    kind, row, col = parse_edge_id(edge_id_value)
    if not _is_edge_in_bounds(state, kind, row, col):
        raise ValueError(f"Edge is out of bounds: {edge_id_value}")

    if edge_id_value in state.edges:
        raise ValueError(f"Edge is already drawn: {edge_id_value}")

    player = state.current_player
    edges = frozenset((*state.edges, edge_id_value))
    edge_owners = _sorted_edge_owners((*state.edge_owners, (edge_id_value, player)))
    boxes_array = np.array(state.boxes, dtype=object)
    scores = [state.scores[0], state.scores[1]]

    completed_boxes = [
        box
        for box in boxes_for_edge(state.rows, state.cols, edge_id_value)
        if _is_box_complete(edges, box["row"], box["col"])
    ]

    scored_boxes: list[dict[str, int]] = []
    for box in completed_boxes:
        row_index = box["row"]
        col_index = box["col"]
        if boxes_array[row_index, col_index] is None:
            boxes_array[row_index, col_index] = player
            scores[player] += 1
            scored_boxes.append({"row": row_index, "col": col_index})

    history_entry = {
        "player": player,
        "edgeId": edge_id_value,
        "scored": bool(scored_boxes),
        "boxes": scored_boxes,
    }

    next_state = replace(
        state,
        edges=edges,
        edge_owners=edge_owners,
        boxes=_array_to_boxes(boxes_array),
        scores=(scores[0], scores[1]),
        history=(*state.history, history_entry),
    )

    if len(legal_moves(next_state)) == 0:
        return replace(next_state, terminal=True, winner=winner_for(next_state))

    if scored_boxes:
        return next_state

    return replace(next_state, current_player=_other_player(player))


def all_edge_ids(rows: int, cols: int) -> list[str]:
    edges: list[str] = []
    for row in range(rows):
        for col in range(cols - 1):
            edges.append(edge_id("h", row, col))
    for row in range(rows - 1):
        for col in range(cols):
            edges.append(edge_id("v", row, col))
    return edges


def edge_id(kind: str, row: int, col: int) -> str:
    return f"{kind}:{row}:{col}"


def parse_edge_id(edge_id_value: str) -> tuple[str, int, int]:
    parts = edge_id_value.split(":")
    if len(parts) != 3 or parts[0] not in {"h", "v"}:
        raise ValueError(f"Invalid edge id: {edge_id_value}")
    try:
        return parts[0], int(parts[1]), int(parts[2])
    except ValueError as error:
        raise ValueError(f"Invalid edge id: {edge_id_value}") from error


def box_edge_ids(row: int, col: int) -> list[str]:
    return [
        edge_id("h", row, col),
        edge_id("h", row + 1, col),
        edge_id("v", row, col),
        edge_id("v", row, col + 1),
    ]


def boxes_for_edge(rows: int, cols: int, edge_id_value: str) -> list[dict[str, int]]:
    kind, row, col = parse_edge_id(edge_id_value)
    boxes: list[dict[str, int]] = []

    if kind == "h":
        if row > 0:
            boxes.append({"row": row - 1, "col": col})
        if row < rows - 1:
            boxes.append({"row": row, "col": col})

    if kind == "v":
        if col > 0:
            boxes.append({"row": row, "col": col - 1})
        if col < cols - 1:
            boxes.append({"row": row, "col": col})

    return [
        box
        for box in boxes
        if box["row"] >= 0 and box["row"] < rows - 1 and box["col"] >= 0 and box["col"] < cols - 1
    ]


def winner_for(state: GameState) -> Winner:
    if state.scores[0] > state.scores[1]:
        return 0
    if state.scores[1] > state.scores[0]:
        return 1
    return "draw"


def state_snapshot(state: GameState) -> dict:
    return {
        "rows": state.rows,
        "cols": state.cols,
        "currentPlayer": state.current_player,
        "edges": sorted(state.edges),
        "edgeOwners": [[edge, owner] for edge, owner in state.edge_owners],
        "boxes": [[cell for cell in row] for row in state.boxes],
        "scores": [state.scores[0], state.scores[1]],
        "terminal": state.terminal,
        "winner": state.winner,
    }


def serialize_state(state: GameState) -> str:
    return json.dumps(state_snapshot(state), separators=(",", ":"), sort_keys=True)


def _is_box_complete(edges: frozenset[str], row: int, col: int) -> bool:
    return all(edge in edges for edge in box_edge_ids(row, col))


def _is_edge_in_bounds(state: GameState, kind: str, row: int, col: int) -> bool:
    if kind == "h":
        return row >= 0 and row < state.rows and col >= 0 and col < state.cols - 1
    return row >= 0 and row < state.rows - 1 and col >= 0 and col < state.cols


def _array_to_boxes(boxes_array: np.ndarray) -> tuple[tuple[int | None, ...], ...]:
    return tuple(tuple(None if cell is None else int(cell) for cell in row) for row in boxes_array)


def _sorted_edge_owners(edge_owners: tuple[tuple[str, int], ...]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(edge_owners, key=lambda entry: entry[0]))


def _other_player(player: int) -> int:
    return 1 if player == 0 else 0
