from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dots_boxes_mcts.az_checkpoint_eval import summarize_checkpoint_match_records


STAGE_DIR = Path("runs/stage-3.6")
LEDGER_DIR = Path("runs/az-flywheel")
STAGE_33_CHECKPOINT = Path("runs/stage-3.3/mlx-resconv-policy-value-4x4-1000.npz")
STATE_FILENAME = "flywheel-state.json"
HISTORY_FILENAME = "flywheel-history.jsonl"


@dataclass(frozen=True)
class FlywheelPaths:
    games: Path
    examples: Path
    checkpoint: Path
    diagnostics: Path
    eval_champion: Path


@dataclass(frozen=True)
class FlywheelConfig:
    iteration: int
    champion_checkpoint: Path = STAGE_33_CHECKPOINT
    init_checkpoint: Path | None = None
    stage_dir: Path = STAGE_DIR
    self_play_games: int = 200
    rows: int = 4
    cols: int = 4
    self_play_simulations: int = 250
    eval_simulations: int = 100
    eval_champion_games: int = 100
    train_epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 0.0005
    validation_fraction: float = 0.1
    diagnostics_every: int = 5
    c_puct: float = 1.5
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    temperature_moves: int = 8
    sampling_temperature: float = 1.0
    mlx_device: str = "gpu"
    debug: bool = True
    self_play_seed: int | None = None
    eval_champion_seed: int | None = None


@dataclass(frozen=True)
class FlywheelState:
    next_iteration: int = 1
    champion_checkpoint: Path = STAGE_33_CHECKPOINT
    latest_candidate_checkpoint: Path | None = None
    last_evaluation: dict[str, Any] | None = None

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "FlywheelState":
        return cls(
            next_iteration=int(payload.get("nextIteration", 1)),
            champion_checkpoint=Path(
                payload.get("championCheckpoint", str(STAGE_33_CHECKPOINT))
            ),
            latest_candidate_checkpoint=optional_path(
                payload.get("latestCandidateCheckpoint")
            ),
            last_evaluation=payload.get("lastEvaluation"),
        )

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": 1,
            "nextIteration": self.next_iteration,
            "championCheckpoint": str(self.champion_checkpoint),
        }
        if self.latest_candidate_checkpoint is not None:
            payload["latestCandidateCheckpoint"] = str(self.latest_candidate_checkpoint)
        if self.last_evaluation is not None:
            payload["lastEvaluation"] = self.last_evaluation
        return payload


def optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def default_self_play_seed(iteration: int) -> int:
    return 6001 + (iteration - 1) * 2000


def default_eval_champion_seed(iteration: int) -> int:
    return 7001 + (iteration - 1) * 2000


def iter_label(iteration: int) -> str:
    return f"iter{iteration:03d}"


def checkpoint_path(stage_dir: Path, iteration: int, simulations: int) -> Path:
    return (
        stage_dir
        / f"mlx-resconv-policy-value-4x4-{iter_label(iteration)}-guided-sims{simulations}.npz"
    )


def default_init_checkpoint(
    iteration: int,
    champion_checkpoint: Path,
    stage_dir: Path,
    simulations: int,
) -> Path:
    if iteration <= 1:
        return champion_checkpoint
    return checkpoint_path(stage_dir=stage_dir, iteration=iteration - 1, simulations=simulations)


def init_checkpoint(config: FlywheelConfig) -> Path:
    return config.init_checkpoint or default_init_checkpoint(
        iteration=config.iteration,
        champion_checkpoint=config.champion_checkpoint,
        stage_dir=config.stage_dir,
        simulations=config.self_play_simulations,
    )


def flywheel_paths(config: FlywheelConfig) -> FlywheelPaths:
    label = iter_label(config.iteration)
    board = f"{config.rows}x{config.cols}"
    games_suffix = f"{board}-{label}-games{config.self_play_games}-sims{config.self_play_simulations}"
    checkpoint = checkpoint_path(
        stage_dir=config.stage_dir,
        iteration=config.iteration,
        simulations=config.self_play_simulations,
    )
    return FlywheelPaths(
        games=config.stage_dir / f"guided-self-play-{games_suffix}.jsonl",
        examples=config.stage_dir / f"guided-examples-{games_suffix}.jsonl",
        checkpoint=checkpoint,
        diagnostics=checkpoint.with_name(f"{checkpoint.stem}-diagnostics.jsonl"),
        eval_champion=config.stage_dir
        / f"iter{config.iteration:03d}-vs-champion-sims{config.eval_simulations}.jsonl",
    )


def flywheel_state_path(ledger_dir: Path = LEDGER_DIR) -> Path:
    return ledger_dir / STATE_FILENAME


def flywheel_history_path(ledger_dir: Path = LEDGER_DIR) -> Path:
    return ledger_dir / HISTORY_FILENAME


def load_state(ledger_dir: Path = LEDGER_DIR) -> FlywheelState:
    path = flywheel_state_path(ledger_dir)
    if not path.exists():
        return FlywheelState()
    return FlywheelState.from_json(json.loads(path.read_text()))


def save_state(state: FlywheelState, ledger_dir: Path = LEDGER_DIR) -> None:
    path = flywheel_state_path(ledger_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_json(), indent=2, sort_keys=True) + "\n")


def append_history(event: dict[str, Any], ledger_dir: Path = LEDGER_DIR) -> None:
    path = flywheel_history_path(ledger_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def example_inputs(config: FlywheelConfig) -> list[Path]:
    return [flywheel_paths(config).examples]


def command_plan(config: FlywheelConfig) -> list[list[str]]:
    paths = flywheel_paths(config)
    training_init_checkpoint = init_checkpoint(config)
    self_play_seed = config.self_play_seed or default_self_play_seed(config.iteration)
    eval_champion_seed = config.eval_champion_seed or default_eval_champion_seed(config.iteration)

    guided_self_play = [
        sys.executable,
        "-m",
        "dots_boxes_mcts.az_guided_self_play",
        "--checkpoint",
        str(config.champion_checkpoint),
        "--iteration",
        str(config.iteration),
        "--games",
        str(config.self_play_games),
        "--rows",
        str(config.rows),
        "--cols",
        str(config.cols),
        "--simulations",
        str(config.self_play_simulations),
        "--seed",
        str(self_play_seed),
        "--root-dirichlet-alpha",
        str(config.root_dirichlet_alpha),
        "--root-exploration-fraction",
        str(config.root_exploration_fraction),
        "--temperature-moves",
        str(config.temperature_moves),
        "--sampling-temperature",
        str(config.sampling_temperature),
        "--c-puct",
        str(config.c_puct),
        "--mlx-device",
        config.mlx_device,
    ]
    if config.debug:
        guided_self_play.append("--debug")

    convert_examples = [
        sys.executable,
        "-m",
        "dots_boxes_mcts.train",
        str(paths.games),
        "--out",
        str(paths.examples),
    ]

    train = [
        sys.executable,
        "-m",
        "dots_boxes_mcts.train",
        str(paths.examples),
        "--init-checkpoint",
        str(training_init_checkpoint),
        "--train-epochs",
        str(config.train_epochs),
        "--batch-size",
        str(config.batch_size),
        "--learning-rate",
        str(config.learning_rate),
        "--validation-fraction",
        str(config.validation_fraction),
        "--diagnostics-every",
        str(config.diagnostics_every),
        "--mlx-device",
        config.mlx_device,
        "--diagnostics-out",
        str(paths.diagnostics),
        "--checkpoint-out",
        str(paths.checkpoint),
    ]

    eval_champion = checkpoint_eval_command(
        candidate=paths.checkpoint,
        baseline=config.champion_checkpoint,
        games=config.eval_champion_games,
        rows=config.rows,
        cols=config.cols,
        simulations=config.eval_simulations,
        seed=eval_champion_seed,
        mlx_device=config.mlx_device,
        out=paths.eval_champion,
    )
    return [guided_self_play, convert_examples, train, eval_champion]


def checkpoint_eval_command(
    candidate: Path,
    baseline: Path,
    games: int,
    rows: int,
    cols: int,
    simulations: int,
    seed: int,
    mlx_device: str,
    out: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "dots_boxes_mcts.az_checkpoint_eval",
        "--candidate",
        str(candidate),
        "--baseline",
        str(baseline),
        "--games",
        str(games),
        "--rows",
        str(rows),
        "--cols",
        str(cols),
        "--simulations",
        str(simulations),
        "--seed",
        str(seed),
        "--mlx-device",
        mlx_device,
        "--out",
        str(out),
    ]


def validate_inputs(config: FlywheelConfig) -> None:
    missing = [config.champion_checkpoint, init_checkpoint(config)]
    missing = [path for path in missing if not path.exists()]
    if missing:
        paths = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required flywheel input(s):\n{paths}")


def validate_outputs(paths: FlywheelPaths, overwrite: bool) -> None:
    if overwrite:
        return
    outputs = [
        paths.games,
        paths.games.with_name(f"{paths.games.stem}.meta.json"),
        paths.examples,
        paths.checkpoint,
        paths.diagnostics,
        paths.eval_champion,
    ]
    existing = [path for path in outputs if path.exists()]
    if existing:
        paths_text = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing flywheel output(s). "
            f"Pass --overwrite to replace them:\n{paths_text}"
        )


def run_commands(commands: list[list[str]], dry_run: bool) -> None:
    for index, command in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {format_command(command)}", flush=True)
        if not dry_run:
            subprocess.run(command, check=True)


def format_command(command: list[str]) -> str:
    return " ".join(shell_quote(part) for part in command)


def shell_quote(value: str) -> str:
    if value and all(char.isalnum() or char in "-_./:=+" for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def read_evaluation_summary(path: Path) -> dict[str, Any]:
    records = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return summarize_checkpoint_match_records(records)


def iteration_evaluation(
    *,
    iteration: int,
    candidate: Path,
    baseline: Path,
    evaluation_path: Path,
    summary: dict[str, Any],
    promoted: bool | None,
    decision: str,
    reason: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "iteration": iteration,
        "candidateCheckpoint": str(candidate),
        "baselineCheckpoint": str(baseline),
        "evaluationPath": str(evaluation_path),
        "summary": summary,
        "promoted": promoted,
        "decision": decision,
    }
    if reason:
        payload["reason"] = reason
    return payload


def record_completed_iteration(
    config: FlywheelConfig,
    ledger_dir: Path = LEDGER_DIR,
) -> FlywheelState:
    paths = flywheel_paths(config)
    summary = read_evaluation_summary(paths.eval_champion)
    had_state = flywheel_state_path(ledger_dir).exists()
    state = load_state(ledger_dir)
    champion_checkpoint = state.champion_checkpoint if had_state else config.champion_checkpoint
    evaluation = iteration_evaluation(
        iteration=config.iteration,
        candidate=paths.checkpoint,
        baseline=config.champion_checkpoint,
        evaluation_path=paths.eval_champion,
        summary=summary,
        promoted=None,
        decision="pending",
    )
    next_state = FlywheelState(
        next_iteration=max(state.next_iteration, config.iteration + 1),
        champion_checkpoint=champion_checkpoint,
        latest_candidate_checkpoint=paths.checkpoint,
        last_evaluation=evaluation,
    )
    save_state(next_state, ledger_dir)
    append_history(
        {
            "event": "iteration_completed",
            "iteration": config.iteration,
            "championCheckpoint": str(config.champion_checkpoint),
            "initCheckpoint": str(init_checkpoint(config)),
            "outputs": {
                "games": str(paths.games),
                "examples": str(paths.examples),
                "checkpoint": str(paths.checkpoint),
                "diagnostics": str(paths.diagnostics),
                "evalChampion": str(paths.eval_champion),
            },
            "evaluation": evaluation,
        },
        ledger_dir,
    )
    return next_state


def promote_iteration(
    *,
    iteration: int,
    stage_dir: Path,
    ledger_dir: Path,
    self_play_simulations: int,
    eval_simulations: int,
    reason: str | None,
) -> FlywheelState:
    state = load_state(ledger_dir)
    candidate = checkpoint_path(
        stage_dir=stage_dir,
        iteration=iteration,
        simulations=self_play_simulations,
    )
    evaluation_path = stage_dir / f"{iter_label(iteration)}-vs-champion-sims{eval_simulations}.jsonl"
    if not candidate.exists():
        raise FileNotFoundError(f"Missing candidate checkpoint: {candidate}")
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Missing evaluation output: {evaluation_path}")

    previous_champion = state.champion_checkpoint
    summary = read_evaluation_summary(evaluation_path)
    evaluation = iteration_evaluation(
        iteration=iteration,
        candidate=candidate,
        baseline=previous_champion,
        evaluation_path=evaluation_path,
        summary=summary,
        promoted=True,
        decision="promoted",
        reason=reason,
    )
    next_state = FlywheelState(
        next_iteration=max(state.next_iteration, iteration + 1),
        champion_checkpoint=candidate,
        latest_candidate_checkpoint=candidate,
        last_evaluation=evaluation,
    )
    save_state(next_state, ledger_dir)
    append_history(
        {
            "event": "candidate_promoted",
            "iteration": iteration,
            "previousChampionCheckpoint": str(previous_champion),
            "championCheckpoint": str(candidate),
            "evaluation": evaluation,
        },
        ledger_dir,
    )
    return next_state


def reject_iteration(
    *,
    iteration: int,
    stage_dir: Path,
    ledger_dir: Path,
    self_play_simulations: int,
    eval_simulations: int,
    reason: str | None,
) -> FlywheelState:
    state = load_state(ledger_dir)
    candidate = checkpoint_path(
        stage_dir=stage_dir,
        iteration=iteration,
        simulations=self_play_simulations,
    )
    evaluation_path = stage_dir / f"{iter_label(iteration)}-vs-champion-sims{eval_simulations}.jsonl"
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Missing evaluation output: {evaluation_path}")

    summary = read_evaluation_summary(evaluation_path)
    evaluation = iteration_evaluation(
        iteration=iteration,
        candidate=candidate,
        baseline=state.champion_checkpoint,
        evaluation_path=evaluation_path,
        summary=summary,
        promoted=False,
        decision="rejected",
        reason=reason,
    )
    next_state = FlywheelState(
        next_iteration=max(state.next_iteration, iteration + 1),
        champion_checkpoint=state.champion_checkpoint,
        latest_candidate_checkpoint=candidate,
        last_evaluation=evaluation,
    )
    save_state(next_state, ledger_dir)
    append_history(
        {
            "event": "candidate_rejected",
            "iteration": iteration,
            "championCheckpoint": str(state.champion_checkpoint),
            "evaluation": evaluation,
        },
        ledger_dir,
    )
    return next_state


def should_promote(
    summary: dict[str, Any],
    *,
    min_win_rate: float,
    min_average_score_margin: float,
) -> bool:
    return (
        float(summary.get("winRate", 0.0)) >= min_win_rate
        and float(summary.get("averageScoreMargin", 0.0)) >= min_average_score_margin
    )


def auto_decision_reason(
    summary: dict[str, Any],
    *,
    promoted: bool,
    min_win_rate: float,
    min_average_score_margin: float,
) -> str:
    decision = "auto-promoted" if promoted else "auto-rejected"
    return (
        f"{decision}: winRate={float(summary.get('winRate', 0.0)):.3f} "
        f"minWinRate={min_win_rate:.3f}; "
        f"averageScoreMargin={float(summary.get('averageScoreMargin', 0.0)):.3f} "
        f"minAverageScoreMargin={min_average_score_margin:.3f}"
    )


def print_status(ledger_dir: Path) -> None:
    state = load_state(ledger_dir)
    print(f"Flywheel state: {flywheel_state_path(ledger_dir)}")
    print(f"Next iteration: {state.next_iteration}")
    print(f"Champion checkpoint: {state.champion_checkpoint}")
    if state.latest_candidate_checkpoint is not None:
        print(f"Latest candidate checkpoint: {state.latest_candidate_checkpoint}")
    if state.last_evaluation is not None:
        print("Last evaluation:")
        print(json.dumps(state.last_evaluation, indent=2, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in {"next", "loop", "status", "init-state", "promote", "reject"}:
        return parse_subcommand_args(argv)
    return parse_legacy_args(argv)


def parse_legacy_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one champion-gated AlphaZero-style candidate iteration."
    )
    parser.set_defaults(command="run")
    parser.add_argument("--iteration", type=int, required=True)
    add_flywheel_options(parser)
    return parser.parse_args(argv)


def add_flywheel_options(
    parser: argparse.ArgumentParser,
    *,
    champion_default: Path | None = STAGE_33_CHECKPOINT,
) -> None:
    if champion_default is None:
        parser.add_argument("--champion-checkpoint", type=Path)
    else:
        parser.add_argument("--champion-checkpoint", type=Path, default=champion_default)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help=(
            "Checkpoint to continue training from. Defaults to the champion for "
            "iteration 1, otherwise the previous iteration candidate."
        ),
    )
    parser.add_argument("--stage-dir", type=Path, default=STAGE_DIR)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--self-play-simulations", type=int, default=250)
    parser.add_argument("--eval-simulations", type=int, default=100)
    parser.add_argument("--eval-champion-games", type=int, default=100)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--diagnostics-every", type=int, default=5)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25)
    parser.add_argument("--temperature-moves", type=int, default=8)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--self-play-seed", type=int)
    parser.add_argument("--eval-champion-seed", type=int)
    parser.add_argument("--mlx-device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def parse_subcommand_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and track champion-gated AlphaZero-style candidate iterations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    next_parser = subparsers.add_parser("next", help="Run the next tracked iteration.")
    add_flywheel_options(next_parser, champion_default=None)
    next_parser.add_argument("--ledger-dir", type=Path, default=LEDGER_DIR)

    loop_parser = subparsers.add_parser(
        "loop",
        help="Run multiple tracked iterations with a fixed automatic promotion policy.",
    )
    add_flywheel_options(loop_parser, champion_default=None)
    loop_parser.add_argument("--ledger-dir", type=Path, default=LEDGER_DIR)
    loop_parser.add_argument("--iterations", type=int, required=True)
    loop_parser.add_argument("--min-win-rate", type=float, default=0.55)
    loop_parser.add_argument("--min-average-score-margin", type=float, default=0.0)

    status_parser = subparsers.add_parser("status", help="Show tracked flywheel state.")
    status_parser.add_argument("--ledger-dir", type=Path, default=LEDGER_DIR)

    init_parser = subparsers.add_parser(
        "init-state",
        help="Initialize or reset the tracked flywheel state.",
    )
    init_parser.add_argument("--ledger-dir", type=Path, default=LEDGER_DIR)
    init_parser.add_argument("--champion-checkpoint", type=Path, default=STAGE_33_CHECKPOINT)
    init_parser.add_argument("--next-iteration", type=int, default=1)
    init_parser.add_argument("--overwrite", action="store_true")

    for name, help_text in [
        ("promote", "Promote an evaluated candidate to champion."),
        ("reject", "Record that an evaluated candidate was not promoted."),
    ]:
        action_parser = subparsers.add_parser(name, help=help_text)
        action_parser.add_argument("--iteration", type=int, required=True)
        action_parser.add_argument("--stage-dir", type=Path, default=STAGE_DIR)
        action_parser.add_argument("--ledger-dir", type=Path, default=LEDGER_DIR)
        action_parser.add_argument("--self-play-simulations", type=int, default=250)
        action_parser.add_argument("--eval-simulations", type=int, default=100)
        action_parser.add_argument("--reason")

    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> FlywheelConfig:
    if args.iteration < 1:
        raise SystemExit("--iteration must be at least 1")
    return FlywheelConfig(
        iteration=args.iteration,
        champion_checkpoint=args.champion_checkpoint,
        init_checkpoint=args.init_checkpoint,
        stage_dir=args.stage_dir,
        self_play_games=args.games,
        rows=args.rows,
        cols=args.cols,
        self_play_simulations=args.self_play_simulations,
        eval_simulations=args.eval_simulations,
        eval_champion_games=args.eval_champion_games,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        diagnostics_every=args.diagnostics_every,
        c_puct=args.c_puct,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_exploration_fraction=args.root_exploration_fraction,
        temperature_moves=args.temperature_moves,
        sampling_temperature=args.sampling_temperature,
        mlx_device=args.mlx_device,
        debug=not args.no_debug,
        self_play_seed=args.self_play_seed,
        eval_champion_seed=args.eval_champion_seed,
    )


def tracked_iteration_config(
    args: argparse.Namespace,
    *,
    allow_champion_override: bool = True,
) -> FlywheelConfig:
    state = load_state(args.ledger_dir)
    args.iteration = state.next_iteration
    if allow_champion_override and args.champion_checkpoint is not None:
        args.champion_checkpoint = args.champion_checkpoint
    else:
        args.champion_checkpoint = state.champion_checkpoint
    return config_from_args(args)


def run_iteration(config: FlywheelConfig, *, dry_run: bool, overwrite: bool) -> None:
    paths = flywheel_paths(config)
    if not dry_run:
        validate_inputs(config)
        validate_outputs(paths, overwrite=overwrite)

    commands = command_plan(config)
    if overwrite:
        commands[0].append("--overwrite")
    print(f"Flywheel iteration {config.iteration}")
    print(f"Champion checkpoint: {config.champion_checkpoint}")
    print(f"Training init checkpoint: {init_checkpoint(config)}")
    print(f"Self-play seed: {config.self_play_seed or default_self_play_seed(config.iteration)}")
    print(f"Evaluation seed: {config.eval_champion_seed or default_eval_champion_seed(config.iteration)}")
    run_commands(commands, dry_run=dry_run)


def run_loop(args: argparse.Namespace) -> None:
    if args.iterations < 1:
        raise SystemExit("--iterations must be at least 1")
    if not 0.0 <= args.min_win_rate <= 1.0:
        raise SystemExit("--min-win-rate must be in [0, 1]")

    for loop_index in range(1, args.iterations + 1):
        config = tracked_iteration_config(args, allow_champion_override=False)
        print(f"\nLoop iteration {loop_index}/{args.iterations}", flush=True)
        run_iteration(config, dry_run=args.dry_run, overwrite=args.overwrite)
        if args.dry_run:
            print("Dry-run loop stops after the next planned tracked iteration.")
            return

        state = record_completed_iteration(config, ledger_dir=args.ledger_dir)
        if state.last_evaluation is None:
            raise RuntimeError("Completed iteration did not record an evaluation.")
        summary = state.last_evaluation["summary"]
        promoted = should_promote(
            summary,
            min_win_rate=args.min_win_rate,
            min_average_score_margin=args.min_average_score_margin,
        )
        reason = auto_decision_reason(
            summary,
            promoted=promoted,
            min_win_rate=args.min_win_rate,
            min_average_score_margin=args.min_average_score_margin,
        )
        if promoted:
            next_state = promote_iteration(
                iteration=config.iteration,
                stage_dir=config.stage_dir,
                ledger_dir=args.ledger_dir,
                self_play_simulations=config.self_play_simulations,
                eval_simulations=config.eval_simulations,
                reason=reason,
            )
            print(f"Auto-promoted iteration {config.iteration}: {next_state.champion_checkpoint}")
        else:
            next_state = reject_iteration(
                iteration=config.iteration,
                stage_dir=config.stage_dir,
                ledger_dir=args.ledger_dir,
                self_play_simulations=config.self_play_simulations,
                eval_simulations=config.eval_simulations,
                reason=reason,
            )
            print(
                f"Auto-rejected iteration {config.iteration}. "
                f"Champion remains: {next_state.champion_checkpoint}"
            )
            print(
                "Training will still initialize from this rejected candidate on "
                "the next iteration unless --init-checkpoint is provided."
            )


def main() -> None:
    args = parse_args()
    if args.command == "status":
        print_status(args.ledger_dir)
        return
    if args.command == "init-state":
        path = flywheel_state_path(args.ledger_dir)
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"State already exists. Pass --overwrite to replace it: {path}")
        state = FlywheelState(
            next_iteration=args.next_iteration,
            champion_checkpoint=args.champion_checkpoint,
        )
        save_state(state, args.ledger_dir)
        append_history(
            {
                "event": "state_initialized",
                "nextIteration": args.next_iteration,
                "championCheckpoint": str(args.champion_checkpoint),
            },
            args.ledger_dir,
        )
        print_status(args.ledger_dir)
        return
    if args.command == "promote":
        state = promote_iteration(
            iteration=args.iteration,
            stage_dir=args.stage_dir,
            ledger_dir=args.ledger_dir,
            self_play_simulations=args.self_play_simulations,
            eval_simulations=args.eval_simulations,
            reason=args.reason,
        )
        print(f"Promoted iteration {args.iteration}: {state.champion_checkpoint}")
        return
    if args.command == "reject":
        state = reject_iteration(
            iteration=args.iteration,
            stage_dir=args.stage_dir,
            ledger_dir=args.ledger_dir,
            self_play_simulations=args.self_play_simulations,
            eval_simulations=args.eval_simulations,
            reason=args.reason,
        )
        print(f"Rejected iteration {args.iteration}. Champion remains: {state.champion_checkpoint}")
        return
    if args.command == "loop":
        run_loop(args)
        return

    if args.command == "next":
        config = tracked_iteration_config(args)
    else:
        config = config_from_args(args)

    run_iteration(config, dry_run=args.dry_run, overwrite=args.overwrite)
    if args.command == "next" and not args.dry_run:
        next_state = record_completed_iteration(config, ledger_dir=args.ledger_dir)
        print(f"Recorded iteration {config.iteration}. Next iteration: {next_state.next_iteration}")


if __name__ == "__main__":
    main()
