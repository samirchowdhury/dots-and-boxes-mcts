from dots_boxes_mcts.game import apply_move, new_game
from dots_boxes_mcts.strategic_eval import (
    extract_unsafe_opener_positions,
    legal_move_profile,
    summarize_strategic_records,
)


def test_legal_move_profile_counts_safe_scoring_and_opener_moves() -> None:
    state = new_game(rows=3, cols=3)
    for move in ["h:0:0", "v:0:0"]:
        state = apply_move(state, move)

    profile = legal_move_profile(state)

    assert profile["safeMoves"] > 0
    assert profile["openerMoves"] > 0
    assert profile["scoringMoves"] == 0


def test_strategic_summary_detects_unsafe_opener_with_safe_moves_available() -> None:
    record = {
        "rows": 3,
        "cols": 3,
        "moves": ["h:0:0", "h:2:1", "h:2:0", "v:0:0", "v:0:1"],
        "finalScores": [0, 0],
        "winner": None,
    }

    summary = summarize_strategic_records([record], perspective_player=0)
    positions = extract_unsafe_opener_positions([record], perspective_player=0)

    assert summary["unsafeOpenerMoves"] == 1
    assert summary["unsafeOpenedThreeSidedBoxes"] == 1
    assert summary["unsafeOpenerPerGame"] == 1.0
    assert positions[0]["ply"] == 4
    assert positions[0]["move"] == "v:0:1"


def test_strategic_summary_keeps_forced_openers_separate() -> None:
    state = new_game(rows=2, cols=2)
    moves = ["h:0:0", "h:1:0", "v:0:0", "v:0:1"]
    for move in moves:
        if not state.terminal:
            state = apply_move(state, move)
    record = {
        "rows": 2,
        "cols": 2,
        "moves": moves,
        "finalScores": list(state.scores),
        "winner": state.winner,
    }

    summary = summarize_strategic_records([record], perspective_player=0)

    assert summary["unsafeOpenerMoves"] == 0
