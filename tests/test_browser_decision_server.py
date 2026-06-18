import json

import pytest

from dots_boxes_mcts.fast_mcts import FastUCTMCTS, NUMBA_IMPORT_ERROR
from dots_boxes_mcts.game import legal_moves, new_game
from dots_boxes_mcts.mcts import SearchResult, SearchStats, UCTMCTS
from dots_boxes_mcts.browser_decision_server import (
    bot_name,
    decision_response,
    searcher_from_payload,
    select_eval_move,
)


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
    assert response["decision"]["moveSelection"] == "max_visit"


def test_decision_response_syncs_opponent_first_when_model_is_second() -> None:
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
    assert response["decision"]["controlledDecisionTurn"] == 1


def test_decision_response_uses_controlled_decision_turn_for_opening_sweep() -> None:
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
        "decisions_played": 0,
        "opening_top_k": 2,
    }

    response = decision_response(payload)

    assert response["decision"]["controlledDecisionTurn"] == 0
    assert response["decision"]["openingTopK"] == 2


def test_decision_response_writes_terminal_record(tmp_path) -> None:
    out_path = tmp_path / "external.jsonl"
    payload = {
        "rows": 2,
        "cols": 2,
        "simulations": 1,
        "seed": 1,
        "opponent": "dotsandboxes.org",
        "source": "dotsandboxes.org",
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
    assert records[0]["opponent"] == "dotsandboxes.org"
    assert records[0]["source"] == "dotsandboxes.org"
    assert records[0]["finalScores"] == [0, 1]


def test_decision_response_writes_second_player_record(tmp_path) -> None:
    out_path = tmp_path / "external-second.jsonl"
    payload = {
        "rows": 2,
        "cols": 2,
        "simulations": 1,
        "seed": 1,
        "our_player": 1,
        "opponent": "dotsandboxes.org",
        "source": "dotsandboxes.org",
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
    assert records[0]["players"] == {"1": "uct_mcts_1", "0": "dotsandboxes.org"}


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


def test_searcher_from_payload_checkpoint_defaults_to_python_network_mcts(monkeypatch) -> None:
    class FakeNetworkGuidedMCTS:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    evaluator = object()
    monkeypatch.setattr(
        "dots_boxes_mcts.browser_decision_server.cached_evaluator",
        lambda checkpoint, device: evaluator,
    )
    monkeypatch.setattr(
        "dots_boxes_mcts.browser_decision_server.NetworkGuidedMCTS",
        FakeNetworkGuidedMCTS,
    )

    searcher = searcher_from_payload(
        payload={"mlxDevice": "gpu", "cPuct": 1.25},
        checkpoint="candidate.npz",
        simulations=7,
        seed=3,
    )

    assert isinstance(searcher, FakeNetworkGuidedMCTS)
    assert searcher.kwargs == {
        "evaluator": evaluator,
        "simulations": 7,
        "c_puct": 1.25,
        "seed": 3,
    }


def test_searcher_from_payload_checkpoint_can_use_cpp_network_mcts(monkeypatch) -> None:
    class FakeFastNetworkGuidedMCTS:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    evaluator = object()
    monkeypatch.setattr(
        "dots_boxes_mcts.browser_decision_server.cached_evaluator",
        lambda checkpoint, device: evaluator,
    )
    monkeypatch.setattr(
        "dots_boxes_mcts.browser_decision_server.FastNetworkGuidedMCTS",
        FakeFastNetworkGuidedMCTS,
    )

    searcher = searcher_from_payload(
        payload={
            "mlxDevice": "gpu",
            "cPuct": 1.25,
            "mctsBackend": "cpp",
            "mctsBatchSize": 8,
            "virtualLoss": 0.5,
        },
        checkpoint="candidate.npz",
        simulations=7,
        seed=3,
    )

    assert isinstance(searcher, FakeFastNetworkGuidedMCTS)
    assert searcher.kwargs == {
        "evaluator": evaluator,
        "simulations": 7,
        "c_puct": 1.25,
        "seed": 3,
        "batch_size": 8,
        "virtual_loss": 0.5,
    }


def test_select_eval_move_sweeps_top_k_only_on_first_controlled_move() -> None:
    result = SearchResult(
        move="best",
        simulations=10,
        root_player=0,
        stats=[
            SearchStats(move="best", visits=9, mean_value=0.5),
            SearchStats(move="explore", visits=1, mean_value=0.0),
        ],
    )

    assert select_eval_move(
        result=result,
        turn=0,
        opening_top_k=2,
        opening_index=1,
    ) == ("explore", "opening_top_k")
    assert select_eval_move(
        result=result,
        turn=1,
        opening_top_k=2,
        opening_index=1,
    ) == ("best", "max_visit")


def test_select_eval_move_wraps_opening_index_through_top_k() -> None:
    result = SearchResult(
        move="a",
        simulations=10,
        root_player=0,
        stats=[
            SearchStats(move="a", visits=5, mean_value=0.5),
            SearchStats(move="b", visits=4, mean_value=0.4),
            SearchStats(move="c", visits=1, mean_value=0.0),
        ],
    )

    assert select_eval_move(
        result=result,
        turn=0,
        opening_top_k=2,
        opening_index=3,
    ) == ("b", "opening_top_k")
