import numpy as np

from dots_boxes_mcts.encoding import (
    CHANNEL_NAMES,
    action_ids,
    action_index,
    box_coordinate,
    edge_coordinate,
    encode_snapshot,
)
from dots_boxes_mcts.game import apply_move, new_game, state_snapshot


def test_edge_and_box_coordinates_preserve_board_geometry() -> None:
    assert edge_coordinate("h:0:1") == (0, 3)
    assert edge_coordinate("v:2:0") == (5, 0)
    assert box_coordinate(1, 2) == (3, 5)


def test_encode_snapshot_marks_edges_boxes_legal_moves_and_metadata() -> None:
    state = new_game(rows=3, cols=3)
    for move in ["h:0:0", "h:1:0", "v:0:0"]:
        state = apply_move(state, move)
    state = apply_move(state, "v:0:1")

    encoded = encode_snapshot(state_snapshot(state))

    assert encoded.tensor.shape == (len(CHANNEL_NAMES), 5, 5)
    assert encoded.action_ids == action_ids(3, 3)
    assert encoded.legal_mask[action_index("h:0:0", 3, 3)] == 0
    assert encoded.legal_mask[action_index("h:0:1", 3, 3)] == 1
    assert encoded.tensor[0, edge_coordinate("h:0:0")[0], edge_coordinate("h:0:0")[1]] == 1
    assert encoded.tensor[3, box_coordinate(0, 0)[0], box_coordinate(0, 0)[1]] == 1
    assert np.all(encoded.tensor[7] == 0.25)
