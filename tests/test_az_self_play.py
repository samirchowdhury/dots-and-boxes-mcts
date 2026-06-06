from dots_boxes_mcts.az_self_play import (
    generate_mcts_self_play_games,
    play_mcts_self_play_game,
    summarize_records,
)
from dots_boxes_mcts.game import apply_move, legal_moves, new_game
from dots_boxes_mcts.train import examples_from_records


def test_mcts_self_play_game_records_every_decision() -> None:
    record = play_mcts_self_play_game(rows=3, cols=3, simulations=5, seed=1)

    assert record["terminal"] is True
    assert record["players"] == {"0": "uct_mcts", "1": "uct_mcts"}
    assert record["dataSource"] == "mcts_self_play"
    assert len(record["moves"]) == 12
    assert len(record["decisions"]) == len(record["moves"])
    assert {decision["player"] for decision in record["decisions"]} == {0, 1}
    assert record["decisions"][0]["search"]["simulations"] == 5


def test_mcts_self_play_records_only_legal_moves() -> None:
    record = play_mcts_self_play_game(rows=3, cols=3, simulations=5, seed=2)
    state = new_game(rows=record["rows"], cols=record["cols"])

    for move in record["moves"]:
        assert move in legal_moves(state)
        state = apply_move(state, move)

    assert [state.scores[0], state.scores[1]] == record["finalScores"]


def test_mcts_self_play_records_feed_training_examples() -> None:
    record = play_mcts_self_play_game(rows=3, cols=3, simulations=5, seed=3)

    examples = examples_from_records([record])

    assert len(examples) == len(record["decisions"])
    assert {example.player for example in examples} == {0, 1}


def test_generate_and_summarize_mcts_self_play_records() -> None:
    records = generate_mcts_self_play_games(games=3, rows=2, cols=2, simulations=3, seed=10)
    summary = summarize_records(records)

    assert [record["seed"] for record in records] == [10, 11, 12]
    assert [record["gameIndex"] for record in records] == [0, 1, 2]
    assert summary["games"] == 3
    assert summary["player0Wins"] + summary["player1Wins"] + summary["draws"] == 3
    assert summary["averageDecisionsPerGame"] == 4
