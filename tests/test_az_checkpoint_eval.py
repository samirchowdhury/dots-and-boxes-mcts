from pathlib import Path

from dots_boxes_mcts.az_checkpoint_eval import (
    checkpoint_match_record,
    summarize_checkpoint_match_records,
)
from dots_boxes_mcts.game import apply_move, new_game


def test_checkpoint_match_summary_uses_candidate_perspective_with_alternating_colors() -> None:
    records = [
        {"candidatePlayer": 0, "finalScores": [3, 1], "winner": 0},
        {"candidatePlayer": 1, "finalScores": [3, 1], "winner": 0},
        {"candidatePlayer": 1, "finalScores": [2, 2], "winner": "draw"},
    ]

    summary = summarize_checkpoint_match_records(records)

    assert summary == {
        "games": 3,
        "wins": 1,
        "draws": 1,
        "losses": 1,
        "winRate": 1 / 3,
        "averageScoreMargin": 0.0,
    }


def test_checkpoint_match_record_names_candidate_and_baseline_roles() -> None:
    state = apply_move(new_game(rows=2, cols=2), "h:0:0")

    record = checkpoint_match_record(
        state=state,
        moves=["h:0:0"],
        seed=7,
        candidate_checkpoint=Path("candidate.npz"),
        baseline_checkpoint=Path("baseline.npz"),
        candidate_player=1,
        simulations=100,
        c_puct=1.5,
        decisions=[],
    )

    assert record["dataSource"] == "checkpoint_match"
    assert record["candidatePlayer"] == 1
    assert record["players"] == {
        "1": "candidate_network_guided_mcts",
        "0": "baseline_network_guided_mcts",
    }
    assert record["candidateCheckpoint"] == "candidate.npz"
    assert record["baselineCheckpoint"] == "baseline.npz"
