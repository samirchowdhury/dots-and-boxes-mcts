from dots_boxes_mcts.game import apply_move, legal_moves, new_game
from dots_boxes_mcts.mcts import UCTMCTS, terminal_value


def test_terminal_value_is_normalized_score_margin() -> None:
    state = new_game(rows=2, cols=2)
    for move in legal_moves(state):
        state = apply_move(state, move)

    assert terminal_value(state, 0) == -1.0
    assert terminal_value(state, 1) == 1.0


def test_mcts_returns_a_legal_move_from_initial_state() -> None:
    state = new_game(rows=3, cols=3)
    searcher = UCTMCTS(simulations=20, seed=1)

    result = searcher.search(state)

    assert result.move in legal_moves(state)
    assert result.simulations == 20
    assert result.root_player == 0
    assert sum(stat.visits for stat in result.stats) == 20


def test_mcts_takes_available_box_and_keeps_turn() -> None:
    state = new_game(rows=3, cols=3)
    for move in ["h:0:0", "h:1:0", "v:0:0"]:
        state = apply_move(state, move)

    searcher = UCTMCTS(simulations=60, exploration_constant=0.5, seed=2)

    assert searcher.choose_move(state) == "v:0:1"
