import pytest

from dots_boxes_mcts.fast_mcts import (
    FastUCTMCTS,
    NUMBA_IMPORT_ERROR,
    fast_apply_move,
    fast_state_from_game,
)
from dots_boxes_mcts.game import apply_move, legal_moves, new_game


pytestmark = pytest.mark.skipif(NUMBA_IMPORT_ERROR is not None, reason="Numba is not installed")


def assert_fast_matches_python(python_state, fast_state) -> None:
    assert fast_state.rows == python_state.rows
    assert fast_state.cols == python_state.cols
    assert fast_state.current_player == python_state.current_player
    assert fast_state.edge_count == len(python_state.edges)
    assert tuple(int(value) for value in fast_state.scores) == python_state.scores
    flat_boxes = tuple(
        -1 if owner is None else owner
        for row in python_state.boxes
        for owner in row
    )
    assert tuple(int(value) for value in fast_state.boxes) == flat_boxes


def test_fast_apply_matches_python_for_every_opening_move() -> None:
    state = new_game(rows=4, cols=4)
    fast_state = fast_state_from_game(state)

    for move in legal_moves(state):
        assert_fast_matches_python(
            apply_move(state, move),
            fast_apply_move(fast_state, move),
        )


def test_fast_apply_matches_extra_turn_and_double_capture() -> None:
    state = new_game(rows=2, cols=3)
    for move in ["h:0:0", "h:1:0", "v:0:0", "h:0:1", "h:1:1", "v:0:2"]:
        state = apply_move(state, move)
    fast_state = fast_state_from_game(state)

    python_next = apply_move(state, "v:0:1")
    fast_next = fast_apply_move(fast_state, "v:0:1")

    assert python_next.scores[python_next.current_player] == 2
    assert_fast_matches_python(python_next, fast_next)


def test_fast_search_returns_legal_deterministic_move_and_stats() -> None:
    state = new_game(rows=3, cols=3)

    first = FastUCTMCTS(simulations=20, seed=7).search(state)
    second = FastUCTMCTS(simulations=20, seed=7).search(state)

    assert first.move in legal_moves(state)
    assert first.move == second.move
    assert first.root_player == state.current_player
    assert sum(stat.visits for stat in first.stats) == 20
