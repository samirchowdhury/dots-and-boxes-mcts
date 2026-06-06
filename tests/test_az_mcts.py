import os

import pytest

from dots_boxes_mcts.az_guided_self_play import play_guided_self_play_game
from dots_boxes_mcts.az_mcts import NetworkEvaluator, NetworkGuidedMCTS
from dots_boxes_mcts.evaluate import play_mcts_vs_random_game
from dots_boxes_mcts.game import legal_moves, new_game
from dots_boxes_mcts.train import examples_from_records, overfit_examples


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
