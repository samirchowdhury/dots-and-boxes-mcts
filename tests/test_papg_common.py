import pytest

from dots_boxes_mcts.papg_common import (
    papg_game_record,
    papg_indexes_to_edges,
)


def test_papg_3x3_indexes_map_to_repo_edges() -> None:
    assert papg_indexes_to_edges([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]) == [
        "h:0:0",
        "h:0:1",
        "v:0:0",
        "v:0:1",
        "v:0:2",
        "h:1:0",
        "h:1:1",
        "v:1:0",
        "v:1:1",
        "v:1:2",
        "h:2:0",
        "h:2:1",
    ]

def test_papg_4x4_indexes_map_to_repo_edges() -> None:
    assert papg_indexes_to_edges(
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47],
        rows=4,
        cols=4,
    ) == [
        "h:0:0",
        "h:0:1",
        "h:0:2",
        "v:0:0",
        "v:0:1",
        "v:0:2",
        "v:0:3",
        "h:1:0",
        "h:1:1",
        "h:1:2",
        "v:1:0",
        "v:1:1",
        "v:1:2",
        "v:1:3",
        "h:2:0",
        "h:2:1",
        "h:2:2",
        "v:2:0",
        "v:2:1",
        "v:2:2",
        "v:2:3",
        "h:3:0",
        "h:3:1",
        "h:3:2",
    ]


def test_papg_index_mapping_rejects_non_edge_cells() -> None:
    with pytest.raises(ValueError, match="dot or box"):
        papg_indexes_to_edges([0], rows=4, cols=4)


def test_papg_game_record_replays_complete_game() -> None:
    moves = ["h:0:0", "h:1:0", "v:0:0", "v:0:1"]

    record = papg_game_record(
        opponent="papg",
        bot="manual_test_bot",
        rows=2,
        cols=2,
        moves=moves,
        our_player=0,
        notes="tiny capture",
    )

    assert record["players"] == {"0": "manual_test_bot", "1": "papg"}
    assert record["moves"] == moves
    assert record["terminal"] is True
    assert record["finalScores"] == [0, 1]
    assert record["winner"] == 1
    assert record["notes"] == "tiny capture"
    assert record["state"]["boxes"] == [[1]]
