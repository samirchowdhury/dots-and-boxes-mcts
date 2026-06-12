import argparse
from pathlib import Path

from dots_boxes_mcts.stage4_runner import (
    DEFAULT_TACTICAL_SUITE,
    DEFAULT_STAGE4_SIMULATIONS,
    STAGE4_DIR,
    Stage4Config,
    Stage4State,
    command_plan,
    default_seed,
    load_state,
    random_checkpoint_path,
    run_loop,
    save_state,
    should_promote,
    stage4_history_path,
    stage4_paths,
    stage4_state_path,
    training_init_checkpoint,
    unsafe_selection_rate,
)


CHAMPION = Path("runs/stage-4/champion.npz")


def test_stage4_paths_are_independent_from_stage3_flywheel() -> None:
    config = Stage4Config(iteration=2, games=3)

    paths = stage4_paths(config)

    assert STAGE4_DIR == Path("runs/stage-4")
    assert paths.games == Path(
        "runs/stage-4/stage4-self-play-4x4-iter002-games3-sims2000.jsonl"
    )
    assert paths.checkpoint == Path(
        "runs/stage-4/mlx-resconv-policy-value-4x4-iter002-pure-restart-sims2000.npz"
    )
    assert stage4_state_path() == Path("runs/stage-4/stage4-state.json")
    assert stage4_history_path() == Path("runs/stage-4/stage4-history.jsonl")
    assert DEFAULT_STAGE4_SIMULATIONS == 2_000


def test_random_checkpoint_path_stays_under_stage4_dir() -> None:
    assert random_checkpoint_path(rows=4, cols=4, seed=7) == Path(
        "runs/stage-4/stage4-random-policy-value-4x4-seed7.npz"
    )


def test_stage4_first_iteration_uses_current_checkpoint_for_guided_self_play_and_training() -> None:
    config = Stage4Config(iteration=1, tactical_suite=None)
    commands = command_plan(config, state=Stage4State(champion_checkpoint=CHAMPION))

    self_play_command = commands[0]
    train_command = commands[2]

    assert self_play_command[2] == "dots_boxes_mcts.az_guided_self_play"
    assert self_play_command[self_play_command.index("--checkpoint") + 1] == str(CHAMPION)
    assert self_play_command[self_play_command.index("--simulations") + 1] == "2000"
    assert "--evaluator-cache-entries" in self_play_command
    assert "--overwrite" in self_play_command
    assert train_command[train_command.index("--init-checkpoint") + 1] == str(CHAMPION)
    assert train_command[train_command.index("--checkpoint-out") + 1] == str(
        stage4_paths(config).checkpoint
    )


def test_stage4_later_iteration_uses_stage4_champion_as_training_init() -> None:
    state = Stage4State(champion_checkpoint=CHAMPION)
    config = Stage4Config(iteration=2, tactical_suite=None)

    assert training_init_checkpoint(config, state) == CHAMPION

    commands = command_plan(config, state=state)

    train_command = commands[2]
    assert train_command[train_command.index("--init-checkpoint") + 1] == str(CHAMPION)


def test_stage4_command_plan_includes_network_guided_tactical_probe_by_default() -> None:
    config = Stage4Config(iteration=1)

    commands = command_plan(config, state=Stage4State(champion_checkpoint=CHAMPION))

    probe = commands[4]
    assert probe[2] == "dots_boxes_mcts.az_mcts_simulation_probe"
    assert str(DEFAULT_TACTICAL_SUITE) in probe
    assert probe[probe.index("--checkpoint") + 1] == str(stage4_paths(config).checkpoint)
    assert probe[probe.index("--simulations") + 1] == "2000"
    assert probe[probe.index("--cache-entries") + 1] == "500000"


def test_stage4_command_plan_passes_cache_entries_to_champion_eval() -> None:
    config = Stage4Config(iteration=1, tactical_suite=None, evaluator_cache_entries=1234)

    commands = command_plan(config, state=Stage4State(champion_checkpoint=CHAMPION))

    eval_command = commands[4]
    assert eval_command[2] == "dots_boxes_mcts.az_checkpoint_eval"
    assert eval_command[eval_command.index("--evaluator-cache-entries") + 1] == "1234"


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


def test_stage4_champion_gate_uses_win_rate_and_margin() -> None:
    assert should_promote(
        {"winRate": 0.55, "averageScoreMargin": 0.0},
        min_win_rate=0.55,
        min_average_score_margin=0.0,
    )
    assert not should_promote(
        {"winRate": 0.54, "averageScoreMargin": 1.0},
        min_win_rate=0.55,
        min_average_score_margin=0.0,
    )
    assert not should_promote(
        {"winRate": 0.60, "averageScoreMargin": -0.1},
        min_win_rate=0.55,
        min_average_score_margin=0.0,
    )


def test_unsafe_selection_rate_reads_matching_simulation_budget() -> None:
    evaluation = {
        "tacticalProbe": [
            {"simulations": 100, "unsafeOpenerSelectionRate": 0.4},
            {"simulations": 2_000, "unsafeOpenerSelectionRate": 0.25},
        ]
    }

    assert unsafe_selection_rate(evaluation, 2_000) == 0.25


def test_stage4_loop_rejects_then_promotes_by_champion_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seed_checkpoint = tmp_path / "random.npz"
    seed_checkpoint.write_text("", encoding="utf8")
    save_state(Stage4State(champion_checkpoint=seed_checkpoint), tmp_path)
    created_checkpoints: list[Path] = []
    self_play_checkpoints: list[Path] = []
    training_init_checkpoints: list[Path] = []
    summaries = [
        {"winRate": 0.50, "averageScoreMargin": 1.0},
        {"winRate": 0.60, "averageScoreMargin": 0.5},
    ]

    def fake_run_commands(commands: list[list[str]], dry_run: bool) -> None:
        assert not dry_run
        for command in commands:
            if command[2] == "dots_boxes_mcts.az_guided_self_play":
                self_play_checkpoints.append(Path(command[command.index("--checkpoint") + 1]))
            if command[2] == "dots_boxes_mcts.train" and "--checkpoint-out" in command:
                training_init_checkpoints.append(
                    Path(command[command.index("--init-checkpoint") + 1])
                )
            if "--out" in command:
                out_path = Path(command[command.index("--out") + 1])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text("", encoding="utf8")
            if "--checkpoint-out" in command:
                checkpoint = Path(command[command.index("--checkpoint-out") + 1])
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text("", encoding="utf8")
                created_checkpoints.append(checkpoint)
            if "--diagnostics-out" in command:
                diagnostics = Path(command[command.index("--diagnostics-out") + 1])
                diagnostics.parent.mkdir(parents=True, exist_ok=True)
                diagnostics.write_text("", encoding="utf8")
            if "--summary-out" in command:
                summary = Path(command[command.index("--summary-out") + 1])
                summary.parent.mkdir(parents=True, exist_ok=True)
                summary.write_text("{}\n", encoding="utf8")
            if "--suite-out" in command:
                suite = Path(command[command.index("--suite-out") + 1])
                suite.parent.mkdir(parents=True, exist_ok=True)
                suite.write_text("", encoding="utf8")

    def fake_read_evaluation_summary(path: Path):
        return summaries.pop(0)

    def fake_read_json(path: Path):
        if path.name == "summary.json":
            return [{"simulations": 2_000, "unsafeOpenerSelectionRate": 0.25}]
        return {}

    monkeypatch.setattr("dots_boxes_mcts.stage4_runner.run_commands", fake_run_commands)
    monkeypatch.setattr(
        "dots_boxes_mcts.stage4_runner.read_evaluation_summary",
        fake_read_evaluation_summary,
    )
    monkeypatch.setattr("dots_boxes_mcts.stage4_runner.read_json", fake_read_json)
    args = argparse.Namespace(
        iterations=2,
        stage_dir=tmp_path,
        games=1,
        rows=4,
        cols=4,
        simulations=2_000,
        seed=None,
        train_epochs=1,
        batch_size=256,
        learning_rate=0.0005,
        validation_fraction=0.1,
        diagnostics_every=5,
        hidden_size=64,
        residual_blocks=4,
        mlx_device="cpu",
        init_checkpoint=None,
        no_tactical_probe=True,
        tactical_suite=DEFAULT_TACTICAL_SUITE,
        tactical_probe_seed=1,
        eval_previous_games=1,
        eval_previous_simulations=1,
        eval_previous_seed=None,
        c_puct=1.5,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        temperature_moves=8,
        sampling_temperature=1.0,
        evaluator_cache_entries=500_000,
        quiet=True,
        min_win_rate=0.55,
        min_average_score_margin=0.0,
        dry_run=False,
        overwrite=False,
    )

    run_loop(args)

    state = load_state(tmp_path)
    assert self_play_checkpoints == [seed_checkpoint, created_checkpoints[0]]
    assert training_init_checkpoints == [seed_checkpoint, created_checkpoints[0]]
    assert state.next_iteration == 3
    assert state.champion_checkpoint == created_checkpoints[-1]
    assert state.latest_candidate_checkpoint == created_checkpoints[-1]
    assert state.last_evaluation is not None
    assert state.last_evaluation["decision"] == "promoted"
