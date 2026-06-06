from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dots_boxes_mcts.game import all_edge_ids, box_edge_ids, new_game, parse_edge_id, state_snapshot

CHANNEL_NAMES = (
    "drawn_edge",
    "current_player_edge",
    "opponent_edge",
    "current_player_box",
    "opponent_box",
    "legal_move",
    "current_player",
    "score_margin",
)


@dataclass(frozen=True)
class EncodedPosition:
    tensor: np.ndarray
    legal_mask: np.ndarray
    action_ids: tuple[str, ...]


def board_shape(rows: int, cols: int) -> tuple[int, int]:
    return 2 * rows - 1, 2 * cols - 1


def action_ids(rows: int, cols: int) -> tuple[str, ...]:
    return tuple(all_edge_ids(rows, cols))


def action_index(edge_id_value: str, rows: int, cols: int) -> int:
    try:
        return action_ids(rows, cols).index(edge_id_value)
    except ValueError as error:
        raise ValueError(f"Unknown edge id for {rows}x{cols} board: {edge_id_value}") from error


def edge_coordinate(edge_id_value: str) -> tuple[int, int]:
    kind, row, col = parse_edge_id(edge_id_value)
    if kind == "h":
        return 2 * row, 2 * col + 1
    return 2 * row + 1, 2 * col


def box_coordinate(row: int, col: int) -> tuple[int, int]:
    return 2 * row + 1, 2 * col + 1


def encode_snapshot(snapshot: dict) -> EncodedPosition:
    rows = int(snapshot["rows"])
    cols = int(snapshot["cols"])
    player = int(snapshot["currentPlayer"])
    opponent = 1 if player == 0 else 0
    height, width = board_shape(rows, cols)
    tensor = np.zeros((len(CHANNEL_NAMES), height, width), dtype=np.float32)

    edge_owners = {edge: owner for edge, owner in snapshot.get("edgeOwners", [])}
    for edge in snapshot.get("edges", []):
        row, col = edge_coordinate(edge)
        owner = edge_owners.get(edge)
        tensor[0, row, col] = 1.0
        if owner == player:
            tensor[1, row, col] = 1.0
        elif owner == opponent:
            tensor[2, row, col] = 1.0

    for row_index, row in enumerate(snapshot.get("boxes", [])):
        for col_index, owner in enumerate(row):
            if owner is None:
                continue
            row_coord, col_coord = box_coordinate(row_index, col_index)
            if owner == player:
                tensor[3, row_coord, col_coord] = 1.0
            elif owner == opponent:
                tensor[4, row_coord, col_coord] = 1.0

    edge_set = set(snapshot.get("edges", []))
    ids = action_ids(rows, cols)
    legal_mask = np.array([edge not in edge_set for edge in ids], dtype=np.float32)
    for edge, is_legal in zip(ids, legal_mask, strict=True):
        if is_legal:
            row, col = edge_coordinate(edge)
            tensor[5, row, col] = 1.0

    tensor[6, :, :] = 1.0 if player == 1 else 0.0
    scores = snapshot.get("scores", [0, 0])
    total_boxes = max((rows - 1) * (cols - 1), 1)
    tensor[7, :, :] = (float(scores[player]) - float(scores[opponent])) / total_boxes

    return EncodedPosition(tensor=tensor, legal_mask=legal_mask, action_ids=ids)


def legal_mask_from_snapshot(snapshot: dict) -> np.ndarray:
    return encode_snapshot(snapshot).legal_mask


def empty_board_snapshot(rows: int, cols: int) -> dict:
    return state_snapshot(new_game(rows=rows, cols=cols))


def box_edges_as_coordinates(row: int, col: int) -> list[tuple[int, int]]:
    return [edge_coordinate(edge) for edge in box_edge_ids(row, col)]
