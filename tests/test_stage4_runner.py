from pathlib import Path

from dots_boxes_mcts.stage4_runner import (
    DEFAULT_TACTICAL_SUITE,
    STAGE4_DIR,
    Stage4Config,
    Stage4State,
    command_plan,
    default_seed,
    load_state,
    save_state,
    stage4_history_path,
    stage4_paths,
    stage4_state_path,
    training_init_checkpoint,
)


def test_stage4_paths_are_independent_from_stage3_flywheel() -> None:
    config = Stage4Config(iteration=2, games=3, simulations=50_000)

    paths = stage4_paths(config)

    assert STAGE4_DIR == Path("runs/stage-4")
    assert paths.games == Path(
        "runs/stage-4/stage4-self-play-4x4-iter002-games3-sims50000.jsonl"
    )
    assert paths.checkpoint == Path(
        "runs/stage-4/mlx-resconv-policy-value-4x4-iter002-pure-restart-sims50000.npz"
    )
    assert stage4_state_path() == Path("runs/stage-4/stage4-state.json")
    assert stage4_history_path() == Path("runs/stage-4/stage4-history.jsonl")


def test_stage4_first_iteration_trains_from_random_weights() -> None:
    config = Stage4Config(iteration=1, tactical_suite=None)
    commands = command_plan(config, state=Stage4State())

    train_command = commands[2]

    assert "--init-checkpoint" not in train_command
    assert train_command[train_command.index("--checkpoint-out") + 1] == str(
        stage4_paths(config).checkpoint
    )


def test_stage4_later_iteration_uses_stage4_champion_as_training_init() -> None:
    champion = Path("runs/stage-4/champion.npz")
    state = Stage4State(champion_checkpoint=champion)
    config = Stage4Config(iteration=2, tactical_suite=None)

    assert training_init_checkpoint(config, state) == champion

    commands = command_plan(config, state=state)

    train_command = commands[2]
    assert train_command[train_command.index("--init-checkpoint") + 1] == str(champion)


def test_stage4_command_plan_includes_numba_tactical_probe_by_default() -> None:
    config = Stage4Config(iteration=1)

    commands = command_plan(config, state=Stage4State())

    probe = commands[4]
    assert probe[2] == "dots_boxes_mcts.mcts_simulation_probe"
    assert str(DEFAULT_TACTICAL_SUITE) in probe
    assert probe[probe.index("--backend") + 1] == "numba"
    assert probe[probe.index("--simulations") + 1] == "50000"


def test_stage4_state_round_trips_without_az_flywheel(tmp_path: Path) -> None:
    state = Stage4State(
        next_iteration=3,
        champion_checkpoint=Path("runs/stage-4/champion.npz"),
        latest_candidate_checkpoint=Path("runs/stage-4/candidate.npz"),
        last_evaluation={"decision": "pending"},
    )

    save_state(state, tmp_path)

    assert stage4_state_path(tmp_path).exists()
    assert load_state(tmp_path) == state


def test_stage4_default_seed_advances_by_iteration() -> None:
    assert default_seed(1) == 42_001
    assert default_seed(2) == 44_001
