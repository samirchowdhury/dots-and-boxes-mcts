from pathlib import Path

from dots_boxes_mcts.az_checkpoint_eval import (
    checkpoint_match_record,
    generate_checkpoint_match_games,
    summarize_checkpoint_match_records,
)
from dots_boxes_mcts.game import apply_move, legal_moves, new_game
from dots_boxes_mcts.mcts import SearchResult


def test_checkpoint_match_summary_uses_candidate_perspective_with_alternating_colors() -> None:
    records = [
        {"candidatePlayer": 0, "finalScores": [3, 1], "winner": 0},
        {"candidatePlayer": 1, "finalScores": [3, 1], "winner": 0},
        {"candidatePlayer": 1, "finalScores": [2, 2], "winner": "draw"},
    ]

    summary = summarize_checkpoint_match_records(records)

    assert summary["games"] == 3
    assert summary["wins"] == 1
    assert summary["draws"] == 1
    assert summary["losses"] == 1
    assert summary["winRate"] == 1 / 3
    assert summary["averageScoreMargin"] == 0.0
    assert summary["strategic"]["unsafeOpenerMoves"] == 0


def test_checkpoint_match_record_names_candidate_and_baseline_roles() -> None:
    state = apply_move(new_game(rows=2, cols=2), "h:0:0")

    record = checkpoint_match_record(
        state=state,
        moves=["h:0:0"],
        seed=7,
        candidate_checkpoint=Path("candidate.npz"),
        baseline_checkpoint=Path("baseline.npz"),
        candidate_player=1,
        simulations=2000,
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
    assert record["reuseTree"] is True


def test_checkpoint_match_uses_seeded_random_openings_for_game_diversity(monkeypatch) -> None:
    class FakeNetworkGuidedMCTS:
        def __init__(self, **kwargs) -> None:
            pass

        def search(self, state):
            return SearchResult(
                move=legal_moves(state)[0],
                simulations=1,
                root_player=state.current_player,
                stats=[],
            )

        def search_reusing_tree(self, state):
            return self.search(state)

        def advance_tree(self, move, next_state):
            return True

    monkeypatch.setattr(
        "dots_boxes_mcts.az_checkpoint_eval.NetworkEvaluator",
        lambda checkpoint, device: object(),
    )
    monkeypatch.setattr(
        "dots_boxes_mcts.az_checkpoint_eval.NetworkGuidedMCTS",
        FakeNetworkGuidedMCTS,
    )

    records = generate_checkpoint_match_games(
        candidate_checkpoint=Path("candidate.npz"),
        baseline_checkpoint=Path("baseline.npz"),
        games=3,
        rows=3,
        cols=3,
        simulations=1,
        seed=7001,
        opening_random_plies=2,
    )

    assert [record["openingRandomPlies"] for record in records] == [2, 2, 2]
    assert all(len(record["openingMoves"]) == 2 for record in records)
    assert len({tuple(record["openingMoves"]) for record in records}) > 1
    assert [record["decisions"][0]["turn"] for record in records] == [2, 2, 2]
