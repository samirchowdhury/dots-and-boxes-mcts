import json

import pytest

from dots_boxes_mcts.game import (
    all_edge_ids,
    apply_move,
    box_edge_ids,
    legal_moves,
    new_game,
    serialize_state,
)


def test_default_5x5_dot_grid_starts_with_40_legal_moves() -> None:
    state = new_game()

    assert len(legal_moves(state)) == 40
    assert state.scores == (0, 0)
    assert state.current_player == 0


def test_edge_order_matches_js_action_space() -> None:
    assert all_edge_ids(3, 3) == [
        "h:0:0",
        "h:0:1",
        "h:1:0",
        "h:1:1",
        "h:2:0",
        "h:2:1",
        "v:0:0",
        "v:0:1",
        "v:0:2",
        "v:1:0",
        "v:1:1",
        "v:1:2",
    ]


def test_non_scoring_move_switches_turn() -> None:
    state = apply_move(new_game(rows=5, cols=5), "h:0:0")

    assert "h:0:0" not in legal_moves(state)
    assert state.current_player == 1


def test_single_box_capture_scores_and_keeps_turn() -> None:
    state = new_game(rows=3, cols=3)
    for move in ["h:0:0", "h:1:0", "v:0:0"]:
        state = apply_move(state, move)

    current_player = state.current_player
    state = apply_move(state, "v:0:1")

    assert state.scores[current_player] == 1
    assert state.current_player == current_player
    assert state.boxes[0][0] == current_player


def test_one_edge_can_complete_two_boxes() -> None:
    state = new_game(rows=3, cols=3)
    for move in ["h:0:0", "v:0:0", "h:1:0", "h:0:1", "v:0:2", "h:1:1"]:
        state = apply_move(state, move)

    current_player = state.current_player
    state = apply_move(state, "v:0:1")

    assert state.scores[current_player] == 2
    assert state.boxes[0][0] == current_player
    assert state.boxes[0][1] == current_player


def test_terminal_after_every_edge_is_drawn() -> None:
    state = new_game(rows=2, cols=2)
    for move in list(legal_moves(state)):
        state = apply_move(state, move)

    assert state.terminal is True
    assert sum(state.scores) == 1
    assert state.winner == 1


def test_box_edge_ids_match_canonical_geometry() -> None:
    assert box_edge_ids(1, 2) == ["h:1:2", "h:2:2", "v:1:2", "v:1:3"]


def test_illegal_moves_raise_errors() -> None:
    state = new_game(rows=2, cols=2)
    state = apply_move(state, "h:0:0")

    with pytest.raises(ValueError, match="already drawn"):
        apply_move(state, "h:0:0")

    with pytest.raises(ValueError, match="out of bounds"):
        apply_move(state, "h:0:1")


def test_serialize_state_is_json() -> None:
    state = apply_move(new_game(rows=2, cols=2), "h:0:0")
    snapshot = json.loads(serialize_state(state))

    assert snapshot["edges"] == ["h:0:0"]
    assert snapshot["edgeOwners"] == [["h:0:0", 0]]
