import pytest

from dots_boxes_mcts.dotsandboxes_org_browser_eval import (
    DEFAULT_BOT_DISPLAY_NAME,
    dotsandboxes_org_edge_number_to_edge,
    drawn_edges_from_dotsandboxes_org_lines,
    edge_to_dotsandboxes_org_coords,
    edge_to_dotsandboxes_org_edge_number,
    move_scores_box,
    recording_move_detail,
)


def test_default_recording_bot_name_is_epsilonzero() -> None:
    assert DEFAULT_BOT_DISPLAY_NAME == "EpsilonZero"


def test_move_scores_box_detects_completed_box() -> None:
    moves = ["h:0:0", "h:1:0", "v:0:0"]

    assert move_scores_box(rows=2, cols=2, moves=moves, move="v:0:1")


def test_move_scores_box_rejects_non_scoring_move() -> None:
    assert not move_scores_box(rows=2, cols=2, moves=[], move="h:0:0")


def test_recording_move_detail_includes_search_summary() -> None:
    detail = recording_move_detail(
        {
            "search": {
                "stats": [
                    {
                        "move": "h:0:0",
                        "visits": 123,
                        "meanValue": 0.25,
                    }
                ]
            }
        }
    )

    assert detail == "Top visit count: 123 | value +0.250"


def test_dotsandboxes_org_edge_numbers_map_to_repo_edges() -> None:
    assert [
        dotsandboxes_org_edge_number_to_edge(edge_number, rows=4, cols=4)
        for edge_number in range(24)
    ] == [
        "h:0:0",
        "h:0:1",
        "h:0:2",
        "h:1:0",
        "h:1:1",
        "h:1:2",
        "h:2:0",
        "h:2:1",
        "h:2:2",
        "h:3:0",
        "h:3:1",
        "h:3:2",
        "v:0:0",
        "v:1:0",
        "v:2:0",
        "v:0:1",
        "v:1:1",
        "v:2:1",
        "v:0:2",
        "v:1:2",
        "v:2:2",
        "v:0:3",
        "v:1:3",
        "v:2:3",
    ]


def test_repo_edges_round_trip_to_dotsandboxes_org_edge_numbers() -> None:
    edges = ["h:0:0", "h:3:2", "v:0:0", "v:2:3"]

    assert [
        dotsandboxes_org_edge_number_to_edge(
            edge_to_dotsandboxes_org_edge_number(edge, rows=4, cols=4),
            rows=4,
            cols=4,
        )
        for edge in edges
    ] == edges


def test_dotsandboxes_org_edge_number_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        dotsandboxes_org_edge_number_to_edge(24, rows=4, cols=4)


def test_edge_to_dotsandboxes_org_coords_uses_site_xy_order() -> None:
    assert edge_to_dotsandboxes_org_coords("h:2:1") == (1, 2, 2, 2)
    assert edge_to_dotsandboxes_org_coords("v:2:1") == (1, 2, 1, 3)


def test_drawn_edges_from_dotsandboxes_org_lines_reads_h_and_v_arrays() -> None:
    drawn = drawn_edges_from_dotsandboxes_org_lines(
        h_lines=[
            [1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        v_lines=[
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 0],
        ],
        rows=4,
        cols=4,
    )

    assert drawn == {"h:0:0", "h:2:1", "v:2:1", "v:0:3"}
