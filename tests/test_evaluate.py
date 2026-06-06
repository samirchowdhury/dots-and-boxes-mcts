from dots_boxes_mcts.evaluate import (
    generate_mcts_vs_random_games,
    play_mcts_vs_random_game,
    summarize_records,
)
from dots_boxes_mcts.game import apply_move, legal_moves, new_game


def test_mcts_vs_random_game_terminates_and_records_search_decisions() -> None:
    record = play_mcts_vs_random_game(rows=3, cols=3, simulations=10, seed=5)

    assert record["terminal"] is True
    assert sum(record["finalScores"]) == 4
    assert len(record["moves"]) == 12
    assert record["players"] == {"0": "uct_mcts", "1": "random"}
    assert record["decisions"]
    assert record["decisions"][0]["search"]["simulations"] == 10


def test_mcts_vs_random_records_only_legal_moves() -> None:
    record = play_mcts_vs_random_game(rows=3, cols=3, simulations=5, seed=7)
    state = new_game(rows=record["rows"], cols=record["cols"])

    for move in record["moves"]:
        assert move in legal_moves(state)
        state = apply_move(state, move)

    assert [state.scores[0], state.scores[1]] == record["finalScores"]


def test_generate_and_summarize_mcts_records() -> None:
    records = generate_mcts_vs_random_games(games=3, rows=2, cols=2, simulations=5, seed=10)
    summary = summarize_records(records, mcts_player=0)

    assert [record["seed"] for record in records] == [10, 11, 12]
    assert [record["gameIndex"] for record in records] == [0, 1, 2]
    assert summary["games"] == 3
    assert summary["wins"] + summary["draws"] + summary["losses"] == 3
