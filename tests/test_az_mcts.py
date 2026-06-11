import os

import pytest

from dots_boxes_mcts.az_guided_self_play import play_guided_self_play_game
from dots_boxes_mcts.az_mcts import NetworkEvaluator, NetworkGuidedMCTS
from dots_boxes_mcts.evaluate import play_mcts_vs_random_game
from dots_boxes_mcts.game import GameState, apply_move, legal_moves, new_game
from dots_boxes_mcts.train import examples_from_records, overfit_examples


class UniformEvaluator:
    def evaluate(self, state: GameState) -> tuple[dict[str, float], float]:
        moves = legal_moves(state)
        return {move: 1.0 / len(moves) for move in moves}, 0.0


def stats_signature(result) -> list[tuple[str, int, float]]:
    return [
        (stat.move, stat.visits, round(stat.mean_value, 8))
        for stat in result.stats
    ]


def test_network_guided_mcts_can_reuse_tree_for_budget_ladder() -> None:
    state = new_game(rows=3, cols=3)
    searcher = NetworkGuidedMCTS(evaluator=UniformEvaluator(), simulations=5, seed=1)

    partial, full = searcher.search_many(state, [2, 5])
    fresh = NetworkGuidedMCTS(evaluator=UniformEvaluator(), simulations=5, seed=1).search(state)

    assert partial.simulations == 2
    assert full.simulations == 5
    assert full.move == fresh.move
    assert stats_signature(full) == stats_signature(fresh)


def test_network_guided_mcts_advances_reused_tree_to_existing_child() -> None:
    state = new_game(rows=3, cols=3)
    searcher = NetworkGuidedMCTS(evaluator=UniformEvaluator(), simulations=3, seed=1)
    result = searcher.search_reusing_tree(state)
    child = searcher._reuse_root.children[result.move]  # noqa: SLF001
    next_state = apply_move(state, result.move)

    assert searcher.advance_tree(result.move, next_state) is True
    assert searcher._reuse_root is child  # noqa: SLF001
    assert searcher._reuse_root.state == next_state  # noqa: SLF001


def test_network_guided_mcts_reused_tree_tops_up_to_budget() -> None:
    state = new_game(rows=3, cols=3)
    searcher = NetworkGuidedMCTS(evaluator=UniformEvaluator(), simulations=5, seed=1)
    result = searcher.search_reusing_tree(state)
    next_state = apply_move(state, result.move)
    assert searcher.advance_tree(result.move, next_state) is True
    reused_visits = searcher._reuse_root.visits  # noqa: SLF001

    searcher.search_reusing_tree(next_state)

    assert reused_visits > 0
    assert searcher._reuse_root.visits == 5  # noqa: SLF001


def test_network_guided_mcts_resets_reused_tree_on_state_mismatch() -> None:
    state = new_game(rows=3, cols=3)
    searcher = NetworkGuidedMCTS(evaluator=UniformEvaluator(), simulations=3, seed=1)
    result = searcher.search_reusing_tree(state)
    mismatched_move = next(move for move in legal_moves(state) if move != result.move)
    mismatched_state = apply_move(state, mismatched_move)

    assert searcher.advance_tree(result.move, mismatched_state) is False
    assert searcher._reuse_root is None  # noqa: SLF001


def test_network_guided_mcts_fresh_search_does_not_mutate_reused_tree() -> None:
    state = new_game(rows=3, cols=3)
    searcher = NetworkGuidedMCTS(evaluator=UniformEvaluator(), simulations=3, seed=1)

    fresh = searcher.search(state)

    assert fresh.move in legal_moves(state)
    assert searcher._reuse_root is None  # noqa: SLF001


def test_network_guided_mcts_returns_legal_move_from_checkpoint(tmp_path) -> None:
    if os.environ.get("RUN_MLX_TESTS") != "1":
        pytest.skip("Set RUN_MLX_TESTS=1 to run MLX runtime tests.")

    record = play_mcts_vs_random_game(rows=3, cols=3, simulations=5, seed=11)
    examples = examples_from_records([record])[:8]
    model, _ = overfit_examples(
        examples,
        epochs=2,
        learning_rate=0.001,
        hidden_size=16,
        residual_blocks=1,
        diagnostics_every=1,
    )
    checkpoint = tmp_path / "tiny-resconv.npz"
    model.save(checkpoint)

    state = new_game(rows=3, cols=3)
    evaluator = NetworkEvaluator(checkpoint=checkpoint, device="cpu")
    searcher = NetworkGuidedMCTS(evaluator=evaluator, simulations=3)
    result = searcher.search(state)

    assert result.move in legal_moves(state)
    assert result.simulations == 3
    assert result.stats


def test_guided_self_play_records_decisions_from_checkpoint(tmp_path) -> None:
    if os.environ.get("RUN_MLX_TESTS") != "1":
        pytest.skip("Set RUN_MLX_TESTS=1 to run MLX runtime tests.")

    record = play_mcts_vs_random_game(rows=3, cols=3, simulations=5, seed=12)
    examples = examples_from_records([record])[:8]
    model, _ = overfit_examples(
        examples,
        epochs=2,
        learning_rate=0.001,
        hidden_size=16,
        residual_blocks=1,
        diagnostics_every=1,
    )
    checkpoint = tmp_path / "tiny-resconv.npz"
    model.save(checkpoint)

    guided_record = play_guided_self_play_game(
        checkpoint=checkpoint,
        rows=3,
        cols=3,
        simulations=2,
        device="cpu",
    )

    assert guided_record["terminal"] is True
    assert guided_record["dataSource"] == "network_guided_self_play"
    assert len(guided_record["moves"]) == 12
    assert len(guided_record["decisions"]) == 12
