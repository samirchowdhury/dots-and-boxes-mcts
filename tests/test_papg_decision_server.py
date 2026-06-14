import json

import pytest

from dots_boxes_mcts.fast_mcts import FastUCTMCTS, NUMBA_IMPORT_ERROR
from dots_boxes_mcts.game import legal_moves, new_game
from dots_boxes_mcts.mcts import UCTMCTS
from dots_boxes_mcts.papg_decision_server import bot_name, decision_response, searcher_from_payload


def test_decision_response_returns_uct_move_for_nonterminal_state() -> None:
    if NUMBA_IMPORT_ERROR is not None:
        pytest.skip("Numba is not installed")

    payload = {
        "rows": 2,
        "cols": 2,
        "simulations": 1,
        "seed": 1,
        "moves": [],
        "drawn_edges": [],
    }

    response = decision_response(payload)

    assert response["terminal"] is False
    assert response["move"] in legal_moves(new_game(rows=2, cols=2))
    assert response["decision"]["turn"] == 0


def test_decision_response_syncs_papg_first_when_model_is_second() -> None:
    if NUMBA_IMPORT_ERROR is not None:
        pytest.skip("Numba is not installed")

    payload = {
        "rows": 2,
        "cols": 2,
        "simulations": 1,
        "seed": 1,
        "our_player": 1,
        "moves": [],
        "drawn_edges": ["h:0:0"],
    }

    response = decision_response(payload)

    assert response["moves"] == ["h:0:0"]
    assert response["terminal"] is False
    assert response["decision"]["player"] == 1
    assert response["decision"]["ourPlayer"] == 1


def test_decision_response_writes_terminal_record(tmp_path) -> None:
    out_path = tmp_path / "papg.jsonl"
    payload = {
        "rows": 2,
        "cols": 2,
        "simulations": 1,
        "seed": 1,
        "moves": ["h:0:0", "h:1:0", "v:0:0", "v:0:1"],
        "drawn_edges": [],
        "decisions": [],
        "write_record": True,
        "out": str(out_path),
    }

    response = decision_response(payload)

    assert response["terminal"] is True
    assert response["winner"] == 1
    records = [json.loads(line) for line in out_path.read_text(encoding="utf8").splitlines()]
    assert records[0]["bot"] == "uct_mcts_1"
    assert records[0]["finalScores"] == [0, 1]


def test_decision_response_writes_second_player_record(tmp_path) -> None:
    out_path = tmp_path / "papg-second.jsonl"
    payload = {
        "rows": 2,
        "cols": 2,
        "simulations": 1,
        "seed": 1,
        "our_player": 1,
        "moves": ["h:0:0", "h:1:0", "v:0:0", "v:0:1"],
        "drawn_edges": [],
        "decisions": [],
        "write_record": True,
        "out": str(out_path),
    }

    response = decision_response(payload)

    assert response["terminal"] is True
    records = [json.loads(line) for line in out_path.read_text(encoding="utf8").splitlines()]
    assert records[0]["ourPlayer"] == 1
    assert records[0]["players"] == {"1": "uct_mcts_1", "0": "papg"}


def test_bot_name_for_checkpoint_uses_checkpoint_stem() -> None:
    assert bot_name(checkpoint="/tmp/candidate.npz", simulations=250) == "network_guided_mcts_250_candidate"


def test_searcher_from_payload_defaults_to_fast_mcts() -> None:
    if NUMBA_IMPORT_ERROR is not None:
        pytest.skip("Numba is not installed")

    searcher = searcher_from_payload(
        payload={},
        checkpoint=None,
        simulations=1,
        seed=1,
    )

    assert isinstance(searcher, FastUCTMCTS)


def test_searcher_from_payload_can_use_python_mcts() -> None:
    searcher = searcher_from_payload(
        payload={"backend": "python"},
        checkpoint=None,
        simulations=1,
        seed=1,
    )

    assert isinstance(searcher, UCTMCTS)
