from pathlib import Path
from types import SimpleNamespace

from dots_boxes_mcts.az_flywheel import (
    FlywheelConfig,
    FlywheelState,
    LEDGER_DIR,
    auto_decision_reason,
    command_plan,
    default_eval_champion_seed,
    default_init_checkpoint,
    default_self_play_seed,
    example_inputs,
    flywheel_history_path,
    flywheel_paths,
    flywheel_state_path,
    init_checkpoint,
    load_state,
    promote_iteration,
    record_completed_iteration,
    run_loop,
    save_state,
    should_promote,
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


def test_state_defaults_to_stage_33_champion_when_missing(tmp_path: Path) -> None:
    state = load_state(tmp_path)

    assert state.next_iteration == 1
    assert state.champion_checkpoint == Path(
        "runs/stage-3.3/mlx-resconv-policy-value-4x4-1000.npz"
    )


def test_ledger_paths_default_to_stage_neutral_directory() -> None:
    assert LEDGER_DIR == Path("runs/az-flywheel")
    assert flywheel_state_path() == Path("runs/az-flywheel/flywheel-state.json")
    assert flywheel_history_path() == Path("runs/az-flywheel/flywheel-history.jsonl")


def test_state_round_trips_paths_and_last_evaluation(tmp_path: Path) -> None:
    state = FlywheelState(
        next_iteration=4,
        champion_checkpoint=Path("runs/stage-3.6/champion.npz"),
        latest_candidate_checkpoint=Path("runs/stage-3.6/candidate.npz"),
        last_evaluation={"iteration": 3, "decision": "promoted"},
    )

    save_state(state, tmp_path)

    assert flywheel_state_path(tmp_path).exists()
    assert load_state(tmp_path) == state


def test_completed_next_iteration_records_summary_and_advances_state(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "ledger"
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    champion = Path("runs/stage-3.3/champion.npz")
    save_state(FlywheelState(champion_checkpoint=champion), ledger_dir)
    config = FlywheelConfig(
        iteration=1,
        champion_checkpoint=champion,
        stage_dir=stage_dir,
    )
    paths = flywheel_paths(config)
    paths.eval_champion.write_text(
        "\n".join(
            [
                '{"candidatePlayer": 0, "finalScores": [5, 3], "winner": 0}',
                '{"candidatePlayer": 1, "finalScores": [2, 4], "winner": 1}',
            ]
        )
        + "\n"
    )

    state = record_completed_iteration(config, ledger_dir=ledger_dir)

    assert state.next_iteration == 2
    assert state.champion_checkpoint == champion
    assert state.latest_candidate_checkpoint == paths.checkpoint
    assert state.last_evaluation is not None
    assert state.last_evaluation["decision"] == "pending"
    assert state.last_evaluation["summary"]["wins"] == 2
    assert flywheel_history_path(ledger_dir).exists()


def test_promote_iteration_updates_champion_from_candidate(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "ledger"
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    champion = Path("runs/stage-3.3/champion.npz")
    save_state(FlywheelState(next_iteration=2, champion_checkpoint=champion), ledger_dir)
    candidate = Path(
        stage_dir / "mlx-resconv-policy-value-4x4-iter001-guided-sims250.npz"
    )
    candidate.write_text("checkpoint")
    (stage_dir / "iter001-vs-champion-sims100.jsonl").write_text(
        '{"candidatePlayer": 0, "finalScores": [5, 3], "winner": 0}\n'
    )

    state = promote_iteration(
        iteration=1,
        stage_dir=stage_dir,
        ledger_dir=ledger_dir,
        self_play_simulations=250,
        eval_simulations=100,
        reason="cleared the bar",
    )

    assert state.champion_checkpoint == candidate
    assert state.next_iteration == 2
    assert state.last_evaluation is not None
    assert state.last_evaluation["decision"] == "promoted"
    assert state.last_evaluation["reason"] == "cleared the bar"


def test_auto_promotion_policy_requires_win_rate_and_margin() -> None:
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
    assert "auto-promoted" in auto_decision_reason(
        {"winRate": 0.6, "averageScoreMargin": 0.2},
        promoted=True,
        min_win_rate=0.55,
        min_average_score_margin=0.0,
    )


def test_loop_rejects_then_promotes_and_keeps_training_from_rejected_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ledger_dir = tmp_path / "ledger"
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    champion = stage_dir / "champion.npz"
    champion.write_text("checkpoint")
    save_state(FlywheelState(next_iteration=1, champion_checkpoint=champion), ledger_dir)
    init_checkpoints: list[Path] = []

    def fake_run_iteration(config, *, dry_run, overwrite) -> None:
        init_checkpoints.append(init_checkpoint(config))
        paths = flywheel_paths(config)
        paths.checkpoint.write_text("checkpoint")
        if config.iteration == 1:
            paths.eval_champion.write_text(
                "\n".join(
                    [
                        '{"candidatePlayer": 0, "finalScores": [5, 3], "winner": 0}',
                        '{"candidatePlayer": 1, "finalScores": [2, 4], "winner": 0}',
                    ]
                )
                + "\n"
            )
        else:
            paths.eval_champion.write_text(
                "\n".join(
                    [
                        '{"candidatePlayer": 0, "finalScores": [5, 3], "winner": 0}',
                        '{"candidatePlayer": 1, "finalScores": [3, 5], "winner": 1}',
                    ]
                )
                + "\n"
            )

    monkeypatch.setattr("dots_boxes_mcts.az_flywheel.run_iteration", fake_run_iteration)

    run_loop(
        SimpleNamespace(
            iterations=2,
            min_win_rate=0.55,
            min_average_score_margin=0.0,
            ledger_dir=ledger_dir,
            stage_dir=stage_dir,
            champion_checkpoint=None,
            init_checkpoint=None,
            games=200,
            rows=4,
            cols=4,
            self_play_simulations=250,
            eval_simulations=100,
            eval_champion_games=100,
            train_epochs=10,
            batch_size=256,
            learning_rate=0.0005,
            validation_fraction=0.1,
            diagnostics_every=5,
            c_puct=1.5,
            root_dirichlet_alpha=0.3,
            root_exploration_fraction=0.25,
            temperature_moves=8,
            sampling_temperature=1.0,
            self_play_seed=None,
            eval_champion_seed=None,
            mlx_device="cpu",
            no_debug=True,
            dry_run=False,
            overwrite=False,
        )
    )

    iter001 = stage_dir / "mlx-resconv-policy-value-4x4-iter001-guided-sims250.npz"
    iter002 = stage_dir / "mlx-resconv-policy-value-4x4-iter002-guided-sims250.npz"
    state = load_state(ledger_dir)

    assert init_checkpoints == [champion, iter001]
    assert state.next_iteration == 3
    assert state.champion_checkpoint == iter002
    assert state.last_evaluation is not None
    assert state.last_evaluation["decision"] == "promoted"
