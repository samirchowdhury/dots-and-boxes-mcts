import json

from dots_boxes_mcts.game import apply_move, legal_moves, new_game
from dots_boxes_mcts.self_play import generate_random_games, write_jsonl


def test_random_self_play_games_terminate_and_score_every_box() -> None:
    records = generate_random_games(games=3, rows=3, cols=3, seed=10)

    assert [record["seed"] for record in records] == [10, 11, 12]
    assert [record["gameIndex"] for record in records] == [0, 1, 2]

    for record in records:
        assert record["terminal"] is True
        assert sum(record["finalScores"]) == 4
        assert len(record["moves"]) == 12
        assert record["state"]["terminal"] is True


def test_random_self_play_records_only_legal_moves() -> None:
    record = generate_random_games(games=1, rows=3, cols=3, seed=20)[0]
    state = new_game(rows=record["rows"], cols=record["cols"])

    for move in record["moves"]:
        assert move in legal_moves(state)
        state = apply_move(state, move)

    assert state.terminal is True
    assert [state.scores[0], state.scores[1]] == record["finalScores"]


def test_write_jsonl_round_trips_records(tmp_path) -> None:
    out_path = tmp_path / "games.jsonl"
    records = generate_random_games(games=2, rows=2, cols=2, seed=30)

    write_jsonl(records, out_path)

    lines = out_path.read_text(encoding="utf8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == records
