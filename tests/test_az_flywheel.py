from pathlib import Path

from dots_boxes_mcts.az_flywheel import (
    FlywheelConfig,
    command_plan,
    default_eval_champion_seed,
    default_init_checkpoint,
    default_self_play_seed,
    example_inputs,
    flywheel_paths,
    init_checkpoint,
)


def test_flywheel_paths_derive_iteration_outputs() -> None:
    config = FlywheelConfig(
        iteration=3,
    )

    paths = flywheel_paths(config)

    assert paths.games == Path(
        "runs/stage-3.6/guided-self-play-4x4-iter003-games200-sims250.jsonl"
    )
    assert paths.examples == Path(
        "runs/stage-3.6/guided-examples-4x4-iter003-games200-sims250.jsonl"
    )
    assert paths.checkpoint == Path(
        "runs/stage-3.6/mlx-resconv-policy-value-4x4-iter003-guided-sims250.npz"
    )
    assert paths.eval_champion == Path("runs/stage-3.6/iter003-vs-champion-sims100.jsonl")


def test_flywheel_seed_defaults_advance_by_iteration() -> None:
    assert default_self_play_seed(1) == 6001
    assert default_self_play_seed(2) == 8001
    assert default_self_play_seed(3) == 10001
    assert default_eval_champion_seed(1) == 7001
    assert default_eval_champion_seed(2) == 9001
    assert default_eval_champion_seed(3) == 11001


def test_example_inputs_use_only_current_iteration() -> None:
    config = FlywheelConfig(
        iteration=3,
    )

    assert example_inputs(config) == [
        Path("runs/stage-3.6/guided-examples-4x4-iter003-games200-sims250.jsonl"),
    ]


def test_default_init_checkpoint_uses_champion_for_first_iteration() -> None:
    champion = Path("runs/stage-3.3/champion.npz")

    assert (
        default_init_checkpoint(
            iteration=1,
            champion_checkpoint=champion,
            stage_dir=Path("runs/stage-3.6"),
            simulations=250,
        )
        == champion
    )


def test_default_init_checkpoint_uses_previous_candidate_after_first_iteration() -> None:
    champion = Path("runs/stage-3.6/iter001.npz")

    assert default_init_checkpoint(
        iteration=3,
        champion_checkpoint=champion,
        stage_dir=Path("runs/stage-3.6"),
        simulations=250,
    ) == Path("runs/stage-3.6/mlx-resconv-policy-value-4x4-iter002-guided-sims250.npz")


def test_explicit_init_checkpoint_overrides_default() -> None:
    explicit = Path("runs/custom-init.npz")
    config = FlywheelConfig(iteration=3, init_checkpoint=explicit)

    assert init_checkpoint(config) == explicit


def test_command_plan_uses_champion_for_self_play_and_eval_but_init_for_training() -> None:
    champion = Path("runs/stage-3.3/champion.npz")
    training_init = Path("runs/stage-3.6/training-init.npz")
    config = FlywheelConfig(
        iteration=3,
        champion_checkpoint=champion,
        init_checkpoint=training_init,
    )

    commands = command_plan(config)

    assert len(commands) == 4
    assert commands[0][commands[0].index("--checkpoint") + 1] == str(champion)
    assert commands[0][commands[0].index("--seed") + 1] == "10001"
    assert commands[2][commands[2].index("--init-checkpoint") + 1] == str(training_init)
    assert commands[2][3] == "runs/stage-3.6/guided-examples-4x4-iter003-games200-sims250.jsonl"
    assert commands[3][commands[3].index("--baseline") + 1] == str(champion)
    assert commands[3][commands[3].index("--seed") + 1] == "11001"
