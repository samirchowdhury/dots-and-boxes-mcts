from pathlib import Path

import pytest

from dots_boxes_mcts.game import apply_move, new_game
from dots_boxes_mcts.papg_common import (
    checkpoint_bot_name,
    edge_owner_from_cell,
    infer_papg_reply,
    initial_papg_state_string,
    parse_drawn_edges,
    parse_edge_owners,
    parse_move_links,
    summarize_papg_records,
)


def test_parse_move_links_reads_papg_indexes() -> None:
    html = '<a href="/dab?.+1+1+0+0+17+STATE">H</a><a href=/dab?.+1+1+0+0+31+STATE>H</a>'

    assert parse_move_links(html) == {
        17: "/dab?.+1+1+0+0+17+STATE",
        31: "/dab?.+1+1+0+0+31+STATE",
    }


def test_parse_move_links_reads_tokenized_papg_indexes() -> None:
    html = (
        '<a href="/dab?Ik1DVFMi+2+2+0+0+1+606060">H</a>'
        '<a href=/dab?Ik1DVFMi+2+2+0+0+17+606060>V</a>'
    )

    assert parse_move_links(html) == {
        1: "/dab?Ik1DVFMi+2+2+0+0+1+606060",
        17: "/dab?Ik1DVFMi+2+2+0+0+17+606060",
    }


def test_initial_papg_state_string_matches_empty_4x4_board() -> None:
    assert initial_papg_state_string(rows=4, cols=4) == "6060606030303060606060303030606060603030306060606"


def test_edge_owner_from_cell_reads_blue_and_red_edges() -> None:
    assert edge_owner_from_cell('<img src="/assets/dab_HB.gif">') == 0
    assert edge_owner_from_cell('<img src="/assets/dab_VR.gif">') == 1
    assert edge_owner_from_cell('<img src="/assets/dab_H.gif">') is None


def test_parse_edge_owners_reads_board_table() -> None:
    html = """
    <table>
      <tr><td><img src="/assets/dab_D.gif"></td><td><img src="/assets/dab_HB.gif"></td><td><img src="/assets/dab_D.gif"></td></tr>
      <tr><td><img src="/assets/dab_VR.gif"></td><td></td><td><img src="/assets/dab_V.gif"></td></tr>
      <tr><td><img src="/assets/dab_D.gif"></td><td><img src="/assets/dab_H.gif"></td><td><img src="/assets/dab_D.gif"></td></tr>
    </table>
    """

    assert parse_edge_owners(html, rows=2, cols=2) == {
        "h:0:0": 0,
        "v:0:0": 1,
    }


def test_parse_drawn_edges_uses_missing_move_links_when_available() -> None:
    move_links = {
        1: "/dab?.+1+1+0+0+1+STATE",
        7: "/dab?.+1+1+0+0+7+STATE",
    }

    assert parse_drawn_edges(move_links=move_links, edge_owners={}, rows=2, cols=2) == {
        "v:0:0",
        "v:0:1",
    }


def test_parse_drawn_edges_prefers_colored_board_edges() -> None:
    move_links = {
        1: "/dab?.+1+1+0+0+1+STATE",
        7: "/dab?.+1+1+0+0+7+STATE",
    }

    assert parse_drawn_edges(
        move_links=move_links,
        edge_owners={"h:0:0": 0},
        rows=2,
        cols=2,
    ) == {"h:0:0"}


def test_infer_papg_reply_orders_extra_turn_chain() -> None:
    state = new_game(rows=3, cols=3)
    for move in ["h:1:0", "h:2:0", "h:1:1", "h:0:1", "h:0:0", "v:1:2", "v:1:1"]:
        state = apply_move(state, move)

    reply = infer_papg_reply(state, ["v:1:0", "h:2:1", "v:0:0"])

    assert set(reply) == {"v:1:0", "h:2:1", "v:0:0"}
    assert reply[-1] == "v:0:0"


def test_infer_papg_reply_can_treat_papg_as_first_player() -> None:
    reply = infer_papg_reply(new_game(rows=2, cols=2), ["h:0:0"], papg_player=0)

    assert reply == ["h:0:0"]


def test_infer_papg_reply_rejects_impossible_order() -> None:
    state = apply_move(new_game(rows=3, cols=3), "h:0:0")

    with pytest.raises(ValueError, match="legal"):
        infer_papg_reply(state, ["h:2:0", "h:2:1", "v:0:0"])


def test_checkpoint_bot_name_uses_checkpoint_stem() -> None:
    assert checkpoint_bot_name(
        checkpoint=Path("runs/stage-3.6/candidate.npz"),
        simulations=250,
    ) == "network_guided_mcts_250_candidate"


def test_summarize_papg_records_uses_each_record_our_player() -> None:
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

    summary = summarize_papg_records(records)

    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["averageScoreMargin"] == 0
    assert summary["byOurPlayer"]["0"]["losses"] == 1
    assert summary["byOurPlayer"]["1"]["wins"] == 1
