import json

import pytest

from dots_boxes_mcts.self_play import generate_random_games, write_jsonl
from dots_boxes_mcts.viewer import game_payload, list_game_files, load_game_record, resolve_game_file


def test_list_game_files_finds_nested_jsonl(tmp_path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "games.jsonl").write_text("{}", encoding="utf8")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf8")

    assert list_game_files(tmp_path) == ["nested/games.jsonl"]


def test_load_game_record_uses_one_based_line_numbers(tmp_path) -> None:
    out_path = tmp_path / "games.jsonl"
    records = generate_random_games(games=2, rows=2, cols=2, seed=40)
    write_jsonl(records, out_path)

    assert load_game_record("games.jsonl", 2, tmp_path) == records[1]


def test_game_payload_replays_every_move(tmp_path) -> None:
    out_path = tmp_path / "games.jsonl"
    records = generate_random_games(games=1, rows=3, cols=3, seed=50)
    write_jsonl(records, out_path)

    payload = game_payload("games.jsonl", 1, tmp_path)

    assert payload["record"] == records[0]
    assert len(payload["frames"]) == len(records[0]["moves"]) + 1
    assert payload["frames"][0]["move"] is None
    assert payload["frames"][-1]["state"] == records[0]["state"]


def test_load_game_record_rejects_invalid_line(tmp_path) -> None:
    (tmp_path / "games.jsonl").write_text(json.dumps({"rows": 2}), encoding="utf8")

    with pytest.raises(ValueError, match="at least 1"):
        load_game_record("games.jsonl", 0, tmp_path)

    with pytest.raises(IndexError, match="fewer than 2"):
        load_game_record("games.jsonl", 2, tmp_path)


def test_resolve_game_file_stays_inside_runs_dir(tmp_path) -> None:
    outside = tmp_path.parent / "outside.jsonl"
    outside.write_text("{}", encoding="utf8")

    with pytest.raises(ValueError, match="inside runs"):
        resolve_game_file("../outside.jsonl", tmp_path)

