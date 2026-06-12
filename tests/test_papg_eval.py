from pathlib import Path

import pytest

from dots_boxes_mcts.game import apply_move, new_game
from dots_boxes_mcts.papg_eval import (
    PapgPage,
    checkpoint_bot_name,
    generate_mcts_vs_papg_games,
    edge_owner_from_cell,
    generate_network_guided_mcts_vs_papg_games,
    infer_papg_reply,
    initial_papg_state_string,
    is_thinking_page,
    papg_thinking_url,
    play_searcher_vs_papg_game,
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


def test_is_thinking_page_detects_intermediate_papg_response() -> None:
    assert is_thinking_page("<p>Thinking...</p>") is True
    assert is_thinking_page("<p>Your move.</p>") is False


def test_papg_thinking_url_switches_human_move_to_compute_poll() -> None:
    assert papg_thinking_url("http://www.papg.com/dab?.+1+1+0+0+15+STATE") == (
        "http://www.papg.com/dab?.+2+1+0+0+15+STATE"
    )


def test_papg_thinking_url_switches_tokenized_move_to_compute_poll() -> None:
    assert papg_thinking_url("http://www.papg.com/dab?Ik1DVFMi+1+2+0+0+15+STATE") == (
        "http://www.papg.com/dab?Ik1DVFMi+2+2+0+0+15+STATE"
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


def test_network_guided_papg_generation_passes_checkpoint_metadata(monkeypatch) -> None:
    created_searchers = []

    def fake_evaluator(*, checkpoint, device):
        return {"checkpoint": checkpoint, "device": device}

    class FakeSearcher:
        def __init__(self, *, evaluator, simulations, c_puct, seed) -> None:
            self.evaluator = evaluator
            self.simulations = simulations
            self.c_puct = c_puct
            self.seed = seed
            created_searchers.append(self)

    def fake_play_searcher_vs_papg_game(**kwargs):
        record = {
            "seed": kwargs["seed"],
            "bot": kwargs["bot"],
            "simulations": kwargs["simulations"],
            "finalScores": [1, 0],
            "winner": 0,
            "terminal": True,
        }
        record.update(kwargs["record_fields"])
        return record

    monkeypatch.setattr("dots_boxes_mcts.papg_eval.NetworkEvaluator", fake_evaluator)
    monkeypatch.setattr("dots_boxes_mcts.papg_eval.NetworkGuidedMCTS", FakeSearcher)
    monkeypatch.setattr(
        "dots_boxes_mcts.papg_eval.play_searcher_vs_papg_game",
        fake_play_searcher_vs_papg_game,
    )

    records = generate_network_guided_mcts_vs_papg_games(
        checkpoint=Path("runs/stage-3.6/candidate.npz"),
        games=2,
        simulations=250,
        seed=10,
        c_puct=1.25,
        device="cpu",
    )

    assert [record["seed"] for record in records] == [10, 11]
    assert [record["gameIndex"] for record in records] == [0, 1]
    assert all(record["checkpoint"] == "runs/stage-3.6/candidate.npz" for record in records)
    assert all(record["cPuct"] == 1.25 for record in records)
    assert all(record["mlxDevice"] == "cpu" for record in records)
    assert [searcher.seed for searcher in created_searchers] == [10, 11]


def test_network_guided_papg_generation_can_alternate_players(monkeypatch) -> None:
    def fake_evaluator(*, checkpoint, device):
        return {"checkpoint": checkpoint, "device": device}

    class FakeSearcher:
        def __init__(self, **kwargs) -> None:
            pass

    def fake_play_searcher_vs_papg_game(**kwargs):
        return {
            "seed": kwargs["seed"],
            "ourPlayer": kwargs["our_player"],
            "finalScores": [1, 0] if kwargs["our_player"] == 0 else [0, 1],
            "winner": kwargs["our_player"],
            "terminal": True,
        }

    monkeypatch.setattr("dots_boxes_mcts.papg_eval.NetworkEvaluator", fake_evaluator)
    monkeypatch.setattr("dots_boxes_mcts.papg_eval.NetworkGuidedMCTS", FakeSearcher)
    monkeypatch.setattr(
        "dots_boxes_mcts.papg_eval.play_searcher_vs_papg_game",
        fake_play_searcher_vs_papg_game,
    )

    records = generate_network_guided_mcts_vs_papg_games(
        checkpoint=Path("runs/stage-4/candidate.npz"),
        games=4,
        simulations=250,
        alternate_players=True,
    )

    assert [record["ourPlayer"] for record in records] == [0, 1, 0, 1]


def test_mcts_papg_generation_can_alternate_players(monkeypatch) -> None:
    def fake_play_mcts_vs_papg_game(**kwargs):
        return {
            "seed": kwargs["seed"],
            "ourPlayer": kwargs["our_player"],
            "finalScores": [1, 0] if kwargs["our_player"] == 0 else [0, 1],
            "winner": kwargs["our_player"],
            "terminal": True,
        }

    monkeypatch.setattr(
        "dots_boxes_mcts.papg_eval.play_mcts_vs_papg_game",
        fake_play_mcts_vs_papg_game,
    )

    records = generate_mcts_vs_papg_games(games=4, alternate_players=True)

    assert [record["ourPlayer"] for record in records] == [0, 1, 0, 1]


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


def test_play_searcher_waits_for_papg_reply_before_next_local_search(monkeypatch) -> None:
    class FakeSearcher:
        def __init__(self) -> None:
            self.moves = iter(["h:0:0", "h:1:0"])
            self.players = []

        def search(self, state):
            self.players.append(state.current_player)
            return type(
                "Result",
                (),
                {
                    "move": next(self.moves),
                    "simulations": 1,
                    "root_player": state.current_player,
                    "stats": [],
                },
            )()

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.polls = 0

        def new_game(self, *, rows, cols, our_player=0):
            return PapgPage(rows=rows, cols=cols, move_links={1: "h"}, edge_owners={}, drawn_edges=set())

        def play_move(self, page, papg_index):
            return PapgPage(
                rows=2,
                cols=2,
                move_links={},
                edge_owners={},
                drawn_edges=set(page.drawn_edges),
                poll_url="/poll",
            )

        def poll_reply(self, page):
            self.polls += 1
            replies = [
                {"h:0:0", "v:0:0"},
                {"h:0:0", "v:0:0", "h:1:0", "v:0:1"},
            ]
            return PapgPage(
                rows=2,
                cols=2,
                move_links={},
                edge_owners={},
                drawn_edges=replies[self.polls - 1],
                poll_url="/poll",
            )

    monkeypatch.setattr("dots_boxes_mcts.papg_eval.PapgClient", FakeClient)
    searcher = FakeSearcher()

    record = play_searcher_vs_papg_game(
        searcher=searcher,
        bot="fake",
        rows=2,
        cols=2,
        simulations=1,
        seed=1,
        request_delay=1.0,
        timeout=1.0,
        debug_dir=None,
        notes="test",
        reuse_tree=False,
    )

    assert searcher.players == [0, 0]
    assert [decision["player"] for decision in record["decisions"]] == [0, 0]
    assert record["moves"] == ["h:0:0", "v:0:0", "h:1:0", "v:0:1"]


def test_play_searcher_syncs_papg_opening_when_local_bot_is_player_one(monkeypatch) -> None:
    class FakeSearcher:
        def __init__(self) -> None:
            self.moves = iter(["h:1:0", "v:0:1"])
            self.players = []

        def search(self, state):
            self.players.append(state.current_player)
            return type(
                "Result",
                (),
                {
                    "move": next(self.moves),
                    "simulations": 1,
                    "root_player": state.current_player,
                    "stats": [],
                },
            )()

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.polls = 0

        def new_game(self, *, rows, cols, our_player=0):
            assert our_player == 1
            return PapgPage(
                rows=rows,
                cols=cols,
                move_links={7: "h"},
                edge_owners={},
                drawn_edges={"h:0:0"},
            )

        def play_move(self, page, papg_index):
            return PapgPage(
                rows=2,
                cols=2,
                move_links={},
                edge_owners={},
                drawn_edges={*page.drawn_edges, "h:1:0"},
                poll_url="/poll",
            )

        def poll_reply(self, page):
            self.polls += 1
            return PapgPage(
                rows=2,
                cols=2,
                move_links={},
                edge_owners={},
                drawn_edges={*page.drawn_edges, "v:0:0"},
                poll_url="/poll",
            )

    monkeypatch.setattr("dots_boxes_mcts.papg_eval.PapgClient", FakeClient)
    searcher = FakeSearcher()

    record = play_searcher_vs_papg_game(
        searcher=searcher,
        bot="fake",
        rows=2,
        cols=2,
        simulations=1,
        seed=1,
        request_delay=1.0,
        timeout=1.0,
        debug_dir=None,
        notes="test",
        reuse_tree=False,
        our_player=1,
    )

    assert searcher.players == [1, 1]
    assert [decision["player"] for decision in record["decisions"]] == [1, 1]
    assert record["ourPlayer"] == 1
    assert record["moves"] == ["h:0:0", "h:1:0", "v:0:0", "v:0:1"]


def test_play_searcher_refuses_to_play_for_papg_when_reply_never_arrives(monkeypatch) -> None:
    class FakeSearcher:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, state):
            self.calls += 1
            return type(
                "Result",
                (),
                {
                    "move": "h:0:0",
                    "simulations": 1,
                    "root_player": state.current_player,
                    "stats": [],
                },
            )()

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def new_game(self, *, rows, cols, our_player=0):
            return PapgPage(rows=rows, cols=cols, move_links={1: "h"}, edge_owners={}, drawn_edges=set())

        def play_move(self, page, papg_index):
            return PapgPage(
                rows=2,
                cols=2,
                move_links={},
                edge_owners={},
                drawn_edges={"h:0:0"},
                poll_url="/poll",
            )

        def poll_reply(self, page):
            return PapgPage(
                rows=2,
                cols=2,
                move_links={},
                edge_owners={},
                drawn_edges={"h:0:0"},
                poll_url="/poll",
            )

    monkeypatch.setattr("dots_boxes_mcts.papg_eval.PapgClient", FakeClient)
    searcher = FakeSearcher()

    with pytest.raises(RuntimeError, match="Refusing to let the local searcher play for Papg"):
        play_searcher_vs_papg_game(
            searcher=searcher,
            bot="fake",
            rows=2,
            cols=2,
            simulations=1,
            seed=1,
            request_delay=1.0,
            timeout=1.0,
            debug_dir=None,
            notes="test",
            reuse_tree=False,
        )

    assert searcher.calls == 1
