import os

import numpy as np
import pytest

from dots_boxes_mcts.evaluate import play_mcts_vs_random_game
from dots_boxes_mcts.train import (
    examples_from_records,
    examples_from_payloads,
    overfit_examples,
    policy_from_search,
    policy_vector,
    serializable_example,
    tensors_from_examples,
    train_checkpoint,
    value_from_record,
)


def test_policy_from_search_normalizes_visit_counts_over_moves() -> None:
    state = {"rows": 2, "cols": 2}
    search = {
        "move": "h:0:0",
        "stats": [
            {"move": "h:0:0", "visits": 3},
            {"move": "v:0:1", "visits": 1},
        ],
    }

    assert policy_from_search(state, search) == {"h:0:0": 0.75, "v:0:1": 0.25}


def test_value_from_record_uses_decision_player_perspective() -> None:
    record = {"rows": 3, "cols": 3, "finalScores": [3, 1]}

    assert value_from_record(record, player=0) == 0.5
    assert value_from_record(record, player=1) == -0.5


def test_examples_from_mcts_record_include_encoding_ready_targets() -> None:
    record = play_mcts_vs_random_game(rows=3, cols=3, simulations=5, seed=1)

    examples = examples_from_records([record])
    first = examples[0]
    vector = policy_vector(first)
    payload = serializable_example(first, include_encoding_summary=True)

    assert examples
    assert first.state == record["decisions"][0]["state"]
    assert first.selected_move == record["decisions"][0]["search"]["move"]
    assert np.isclose(vector.sum(), 1.0)
    assert payload["encoding"]["tensorShape"] == [8, 5, 5]
    assert payload["encoding"]["policyVectorNonZero"] > 0


def test_examples_from_payloads_reads_serialized_examples() -> None:
    record = play_mcts_vs_random_game(rows=3, cols=3, simulations=5, seed=4)
    original = examples_from_records([record])[0]

    [loaded] = examples_from_payloads([serializable_example(original)])

    assert loaded.state == original.state
    assert loaded.player == original.player
    assert loaded.policy == original.policy
    assert loaded.value == original.value


def test_tensors_from_examples_match_training_shapes() -> None:
    record = play_mcts_vs_random_game(rows=3, cols=3, simulations=5, seed=2)
    examples = examples_from_records([record])[:4]

    x, policy_target, value_target, legal_mask = tensors_from_examples(examples)

    assert x.shape == (4, 5, 5, 8)
    assert policy_target.shape == (4, 12)
    assert value_target.shape == (4,)
    assert legal_mask.shape == (4, 12)
    assert np.allclose(policy_target.sum(axis=1), 1.0)


def test_tiny_overfit_network_reduces_loss_on_small_batch() -> None:
    if os.environ.get("RUN_MLX_TESTS") != "1":
        pytest.skip("Set RUN_MLX_TESTS=1 to run MLX runtime tests.")

    record = play_mcts_vs_random_game(rows=3, cols=3, simulations=8, seed=3)
    examples = examples_from_records([record])[:6]

    _, diagnostics = overfit_examples(
        examples,
        epochs=20,
        learning_rate=0.001,
        hidden_size=32,
        residual_blocks=1,
        seed=1,
        diagnostics_every=10,
    )

    assert diagnostics[-1].loss < diagnostics[0].loss
    assert diagnostics[-1].policy_loss < diagnostics[0].policy_loss
    assert diagnostics[-1].policy_kl < diagnostics[0].policy_kl


def test_train_checkpoint_reports_train_and_validation_diagnostics() -> None:
    if os.environ.get("RUN_MLX_TESTS") != "1":
        pytest.skip("Set RUN_MLX_TESTS=1 to run MLX runtime tests.")

    records = [
        play_mcts_vs_random_game(rows=3, cols=3, simulations=5, seed=seed)
        for seed in range(10, 13)
    ]
    examples = examples_from_records(records)

    _, diagnostics = train_checkpoint(
        examples,
        epochs=2,
        batch_size=8,
        learning_rate=0.001,
        hidden_size=16,
        residual_blocks=1,
        seed=1,
        diagnostics_every=1,
    )

    assert {item["split"] for item in diagnostics} == {"train", "validation"}
    assert diagnostics[-1]["epoch"] == 2
