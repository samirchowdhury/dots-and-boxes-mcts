from dots_boxes_mcts.papg_browser_eval import (
    DEFAULT_BOT_DISPLAY_NAME,
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
