from dots_boxes_mcts.game import apply_move, new_game, state_snapshot
from dots_boxes_mcts.mcts_simulation_probe import (
    classify_move,
    state_from_snapshot,
    summarize_probe,
)


def test_state_from_snapshot_round_trips_position() -> None:
    state = new_game(rows=3, cols=3)
    for move in ["h:0:0", "v:0:0", "h:2:1"]:
        state = apply_move(state, move)

    restored = state_from_snapshot(state_snapshot(state))

    assert restored.rows == state.rows
    assert restored.cols == state.cols
    assert restored.current_player == state.current_player
    assert restored.edges == state.edges
    assert restored.edge_owners == state.edge_owners
    assert restored.boxes == state.boxes
    assert restored.scores == state.scores


def test_classify_move_marks_avoidable_three_sided_box_as_unsafe_opener() -> None:
    state = new_game(rows=3, cols=3)
    for move in ["h:0:0", "h:2:1", "h:2:0", "v:0:0"]:
        state = apply_move(state, move)

    assert classify_move(state, "v:0:1") == "unsafe_opener"


def test_summarize_probe_counts_selection_rates_by_budget() -> None:
    results = [
        {
            "simulations": 10,
            "trials": [
                {
                    "isSafeOrScoring": True,
                    "isUnsafeOpener": False,
                    "matchesOriginalUnsafeMove": False,
                    "safeOrScoringVisitShare": 0.7,
                    "unsafeOpenerVisitShare": 0.3,
                },
                {
                    "isSafeOrScoring": False,
                    "isUnsafeOpener": True,
                    "matchesOriginalUnsafeMove": True,
                    "safeOrScoringVisitShare": 0.4,
                    "unsafeOpenerVisitShare": 0.6,
                },
            ],
        }
    ]

    summary = summarize_probe(results, positions=1, seeds=[1, 2])

    assert summary[0]["safeOrScoringSelectionRate"] == 0.5
    assert summary[0]["unsafeOpenerSelectionRate"] == 0.5
    assert summary[0]["originalUnsafeMoveSelectionRate"] == 0.5
    assert summary[0]["averageSafeOrScoringVisitShare"] == 0.55
