from pathlib import Path

import pytest

from dots_boxes_mcts.external_bot_common import (
    checkpoint_bot_name,
    external_game_record,
    infer_opponent_reply,
    summarize_external_records,
)
from dots_boxes_mcts.game import apply_move, new_game


def test_external_game_record_replays_complete_game() -> None:
    moves = ["h:0:0", "h:1:0", "v:0:0", "v:0:1"]

    record = external_game_record(
        opponent="dotsandboxes.org",
        bot="manual_test_bot",
        rows=2,
        cols=2,
        moves=moves,
        our_player=0,
        source="dotsandboxes.org",
        notes="tiny capture",
    )

    assert record["players"] == {"0": "manual_test_bot", "1": "dotsandboxes.org"}
    assert record["moves"] == moves
    assert record["terminal"] is True
    assert record["finalScores"] == [0, 1]
    assert record["winner"] == 1
    assert record["notes"] == "tiny capture"
    assert record["state"]["boxes"] == [[1]]


def test_infer_opponent_reply_orders_extra_turn_chain() -> None:
    state = new_game(rows=3, cols=3)
    for move in ["h:1:0", "h:2:0", "h:1:1", "h:0:1", "h:0:0", "v:1:2", "v:1:1"]:
        state = apply_move(state, move)

    reply = infer_opponent_reply(state, ["v:1:0", "h:2:1", "v:0:0"])

    assert set(reply) == {"v:1:0", "h:2:1", "v:0:0"}
    assert reply[-1] == "v:0:0"


def test_infer_opponent_reply_can_treat_opponent_as_first_player() -> None:
    reply = infer_opponent_reply(new_game(rows=2, cols=2), ["h:0:0"], opponent_player=0)

    assert reply == ["h:0:0"]


def test_infer_opponent_reply_rejects_impossible_order() -> None:
    state = apply_move(new_game(rows=3, cols=3), "h:0:0")

    with pytest.raises(ValueError, match="legal"):
        infer_opponent_reply(state, ["h:2:0", "h:2:1", "v:0:0"])


def test_checkpoint_bot_name_uses_checkpoint_stem() -> None:
    assert checkpoint_bot_name(
        checkpoint=Path("runs/stage-3.6/candidate.npz"),
        simulations=250,
    ) == "network_guided_mcts_250_candidate"


def test_summarize_external_records_uses_each_record_our_player() -> None:
    records = [
        {
            "rows": 2,
            "cols": 2,
            "moves": ["h:0:0", "h:1:0", "v:0:0", "v:0:1"],
            "ourPlayer": 0,
            "finalScores": [0, 1],
            "winner": 1,
        },
        {
            "rows": 2,
            "cols": 2,
            "moves": ["h:0:0", "h:1:0", "v:0:0", "v:0:1"],
            "ourPlayer": 1,
            "finalScores": [0, 1],
            "winner": 1,
        },
    ]

    summary = summarize_external_records(records)

    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["averageScoreMargin"] == 0
    assert summary["byOurPlayer"]["0"]["losses"] == 1
    assert summary["byOurPlayer"]["1"]["wins"] == 1
