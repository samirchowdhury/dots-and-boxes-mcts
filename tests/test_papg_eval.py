import pytest

from dots_boxes_mcts.game import apply_move, new_game
from dots_boxes_mcts.papg_eval import (
    edge_owner_from_cell,
    infer_papg_reply,
    initial_papg_state_string,
    is_thinking_page,
    papg_thinking_url,
    parse_drawn_edges,
    parse_edge_owners,
    parse_move_links,
)


def test_parse_move_links_reads_papg_indexes() -> None:
    html = '<a href="/dab?.+1+1+0+0+17+STATE">H</a><a href=/dab?.+1+1+0+0+31+STATE>H</a>'

    assert parse_move_links(html) == {
        17: "/dab?.+1+1+0+0+17+STATE",
        31: "/dab?.+1+1+0+0+31+STATE",
    }


def test_initial_papg_state_string_matches_empty_4x4_board() -> None:
    assert initial_papg_state_string(rows=4, cols=4) == "6060606030303060606060303030606060603030306060606"


def test_is_thinking_page_detects_intermediate_papg_response() -> None:
    assert is_thinking_page("<p>Thinking...</p>") is True
    assert is_thinking_page("<p>Your move.</p>") is False


def test_papg_thinking_url_switches_human_move_to_compute_poll() -> None:
    assert papg_thinking_url("http://www.papg.com/dab?.+1+1+0+0+15+STATE") == (
        "http://www.papg.com/dab?.+2+1+0+0+15+STATE"
    )


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


def test_infer_papg_reply_orders_extra_turn_chain() -> None:
    state = new_game(rows=3, cols=3)
    for move in ["h:1:0", "h:2:0", "h:1:1", "h:0:1", "h:0:0", "v:1:2", "v:1:1"]:
        state = apply_move(state, move)

    reply = infer_papg_reply(state, ["v:1:0", "h:2:1", "v:0:0"])

    assert set(reply) == {"v:1:0", "h:2:1", "v:0:0"}
    assert reply[-1] == "v:0:0"


def test_infer_papg_reply_rejects_impossible_order() -> None:
    state = apply_move(new_game(rows=3, cols=3), "h:0:0")

    with pytest.raises(ValueError, match="legal"):
        infer_papg_reply(state, ["h:2:0", "h:2:1", "v:0:0"])
