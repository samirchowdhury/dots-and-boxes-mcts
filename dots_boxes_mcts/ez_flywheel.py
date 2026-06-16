from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dots_boxes_mcts.ez_checkpoint_eval import summarize_checkpoint_match_records
from dots_boxes_mcts.encoding import CHANNEL_NAMES, action_ids, board_shape
from dots_boxes_mcts.train import MlxPolicyValueNetwork

EZ_FLYWHEEL_DIR = Path("runs/ez-flywheel")
STATE_FILENAME = "ez-flywheel-state.json"
HISTORY_FILENAME = "ez-flywheel-history.jsonl"
DEFAULT_TACTICAL_SUITE = Path("runs/stage-3.8/papg-stage3.6-unsafe-opener-positions.jsonl")
DEFAULT_EZ_FLYWHEEL_SIMULATIONS = 2_000
DEFAULT_EVALUATOR_CACHE_ENTRIES = 500_000
_DURATION_PART_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>[smhd])")


@dataclass(frozen=True)
class EzFlywheelPaths:
    games: Path
    examples: Path
    checkpoint: Path
    diagnostics: Path
    strategic_summary: Path
    unsafe_positions: Path
    tactical_probe: Path
    eval_champion: Path


@dataclass(frozen=True)
class EzFlywheelConfig:
    iteration: int
    run_dir: Path = EZ_FLYWHEEL_DIR
    games: int = 25
    rows: int = 4
    cols: int = 4
    simulations: int = DEFAULT_EZ_FLYWHEEL_SIMULATIONS
    seed: int | None = None
    train_epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 0.0005
    validation_fraction: float = 0.1
    diagnostics_every: int = 5
    hidden_size: int = 64
    residual_blocks: int = 4
    mlx_device: str = "gpu"
    debug: bool = True
    init_checkpoint: Path | None = None
    run_strategic_eval: bool = False
    tactical_suite: Path | None = None
    tactical_probe_seed: int = 1
    eval_champion_games: int = 20
    eval_champion_simulations: int = 2000
    eval_champion_seed: int | None = None
    c_puct: float = 1.5
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    temperature_moves: int = 8
    sampling_temperature: float = 1.0
    evaluator_cache_entries: int = DEFAULT_EVALUATOR_CACHE_ENTRIES


@dataclass(frozen=True)
class EzFlywheelState:
    next_iteration: int = 1
    champion_checkpoint: Path | None = None
    latest_candidate_checkpoint: Path | None = None
    last_evaluation: dict[str, Any] | None = None

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "EzFlywheelState":
        return cls(
            next_iteration=int(payload.get("nextIteration", 1)),
            champion_checkpoint=optional_path(payload.get("championCheckpoint")),
            latest_candidate_checkpoint=optional_path(payload.get("latestCandidateCheckpoint")),
            last_evaluation=payload.get("lastEvaluation"),
        )

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": 1,
            "nextIteration": self.next_iteration,
        }
        if self.champion_checkpoint is not None:
            payload["championCheckpoint"] = str(self.champion_checkpoint)
        if self.latest_candidate_checkpoint is not None:
            payload["latestCandidateCheckpoint"] = str(self.latest_candidate_checkpoint)
        if self.last_evaluation is not None:
            payload["lastEvaluation"] = self.last_evaluation
        return payload


def optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def iter_label(iteration: int) -> str:
    return f"iter{iteration:03d}"


def default_seed(iteration: int) -> int:
    return 42_001 + (iteration - 1) * 2_000


def default_eval_seed(iteration: int) -> int:
    return 43_001 + (iteration - 1) * 2_000


def parse_duration_seconds(text: str | None) -> float | None:
    if text is None:
        return None
    value = text.strip().lower()
    if not value:
        raise ValueError("Duration cannot be empty.")
    if value.replace(".", "", 1).isdigit():
        seconds = float(value)
    else:
        position = 0
        seconds = 0.0
        multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86_400}
        for match in _DURATION_PART_RE.finditer(value):
            if match.start() != position:
                raise ValueError(f"Invalid duration: {text!r}. Use forms like 30m, 12h, or 1h30m.")
            seconds += float(match.group("value")) * multipliers[match.group("unit")]
            position = match.end()
        if position != len(value):
            raise ValueError(f"Invalid duration: {text!r}. Use forms like 30m, 12h, or 1h30m.")
    if seconds <= 0:
        raise ValueError("--duration must be greater than zero.")
    return seconds


def ez_flywheel_state_path(run_dir: Path = EZ_FLYWHEEL_DIR) -> Path:
    return run_dir / STATE_FILENAME


def ez_flywheel_history_path(run_dir: Path = EZ_FLYWHEEL_DIR) -> Path:
    return run_dir / HISTORY_FILENAME


def random_checkpoint_path(
    *,
    rows: int = 4,
    cols: int = 4,
    seed: int = 1,
    run_dir: Path = EZ_FLYWHEEL_DIR,
) -> Path:
    return run_dir / f"ez-random-policy-value-{rows}x{cols}-seed{seed}.npz"


def create_random_checkpoint(
    path: Path,
    *,
    rows: int,
    cols: int,
    hidden_size: int,
    residual_blocks: int,
    seed: int,
    device: str,
    overwrite: bool = False,
) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Random checkpoint already exists: {path}. Pass --overwrite to replace it."
        )
    height, width = board_shape(rows, cols)
    model = MlxPolicyValueNetwork(
        board_height=height,
        board_width=width,
        channels=len(CHANNEL_NAMES),
        action_count=len(action_ids(rows, cols)),
        hidden_size=hidden_size,
        residual_blocks=residual_blocks,
        seed=seed,
        device=device,
    )
    model.save(path)
    return path


def load_state(run_dir: Path = EZ_FLYWHEEL_DIR) -> EzFlywheelState:
    path = ez_flywheel_state_path(run_dir)
    if not path.exists():
        return EzFlywheelState()
    return EzFlywheelState.from_json(json.loads(path.read_text(encoding="utf8")))


def save_state(state: EzFlywheelState, run_dir: Path = EZ_FLYWHEEL_DIR) -> None:
    path = ez_flywheel_state_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf8")


def append_history(event: dict[str, Any], run_dir: Path = EZ_FLYWHEEL_DIR) -> None:
    path = ez_flywheel_history_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **event,
                },
                sort_keys=True,
            )
            + "\n"
        )


def ez_flywheel_paths(config: EzFlywheelConfig) -> EzFlywheelPaths:
    label = iter_label(config.iteration)
    board = f"{config.rows}x{config.cols}"
    suffix = f"{board}-{label}-games{config.games}-sims{config.simulations}"
    checkpoint = (
        config.run_dir
        / f"ez-policy-value-{board}-{label}-sims{config.simulations}.npz"
    )
    return EzFlywheelPaths(
        games=config.run_dir / f"ez-self-play-{suffix}.jsonl",
        examples=config.run_dir / f"ez-examples-{suffix}.jsonl",
        checkpoint=checkpoint,
        diagnostics=checkpoint.with_name(f"{checkpoint.stem}-diagnostics.jsonl"),
        strategic_summary=config.run_dir / f"{label}-strategic-summary.json",
        unsafe_positions=config.run_dir / f"{label}-unsafe-opener-positions.jsonl",
        tactical_probe=config.run_dir / f"{label}-tactical-probe",
        eval_champion=config.run_dir
        / f"{label}-vs-champion-sims{config.eval_champion_simulations}.jsonl",
    )


def current_training_checkpoint(config: EzFlywheelConfig, state: EzFlywheelState) -> Path | None:
    if config.init_checkpoint is not None:
        return config.init_checkpoint
    return state.latest_candidate_checkpoint or state.champion_checkpoint


def training_init_checkpoint(config: EzFlywheelConfig, state: EzFlywheelState) -> Path | None:
    return current_training_checkpoint(config, state)


def self_play_checkpoint(config: EzFlywheelConfig, state: EzFlywheelState) -> Path | None:
    return current_training_checkpoint(config, state)


def command_plan(config: EzFlywheelConfig, state: EzFlywheelState | None = None) -> list[list[str]]:
    state = state or load_state(config.run_dir)
    paths = ez_flywheel_paths(config)
    seed = config.seed or default_seed(config.iteration)
    checkpoint = self_play_checkpoint(config, state)
    if checkpoint is None:
        raise ValueError(
            "EpsilonZero network-guided self-play requires a checkpoint. "
            "Initialize EpsilonZero with --champion-checkpoint or pass --init-checkpoint."
        )
    commands: list[list[str]] = [
        [
            sys.executable,
            "-m",
            "dots_boxes_mcts.ez_guided_self_play",
            "--checkpoint",
            str(checkpoint),
            "--iteration",
            str(config.iteration),
            "--games",
            str(config.games),
            "--rows",
            str(config.rows),
            "--cols",
            str(config.cols),
            "--simulations",
            str(config.simulations),
            "--seed",
            str(seed),
            "--c-puct",
            str(config.c_puct),
            "--root-dirichlet-alpha",
            str(config.root_dirichlet_alpha),
            "--root-exploration-fraction",
            str(config.root_exploration_fraction),
            "--temperature-moves",
            str(config.temperature_moves),
            "--sampling-temperature",
            str(config.sampling_temperature),
            "--mlx-device",
            config.mlx_device,
            "--evaluator-cache-entries",
            str(config.evaluator_cache_entries),
            "--out",
            str(paths.games),
            "--overwrite",
        ],
        [
            sys.executable,
            "-m",
            "dots_boxes_mcts.train",
            str(paths.games),
            "--out",
            str(paths.examples),
        ],
    ]
    if config.debug:
        commands[0].append("--debug")

    train = [
        sys.executable,
        "-m",
        "dots_boxes_mcts.train",
        str(paths.examples),
        "--train-epochs",
        str(config.train_epochs),
        "--batch-size",
        str(config.batch_size),
        "--learning-rate",
        str(config.learning_rate),
        "--hidden-size",
        str(config.hidden_size),
        "--residual-blocks",
        str(config.residual_blocks),
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
    init = training_init_checkpoint(config, state)
    if init is not None:
        train[4:4] = ["--init-checkpoint", str(init)]
    commands.append(train)

    if config.run_strategic_eval:
        commands.append(
            [
                sys.executable,
                "-m",
                "dots_boxes_mcts.strategic_eval",
                str(paths.games),
                "--summary-out",
                str(paths.strategic_summary),
                "--suite-out",
                str(paths.unsafe_positions),
            ]
        )

    if config.tactical_suite is not None:
        commands.append(
            [
                sys.executable,
                "-m",
                "dots_boxes_mcts.ez_mcts_simulation_probe",
                str(config.tactical_suite),
                "--checkpoint",
                str(paths.checkpoint),
                "--inputs-are-positions",
                "--simulations",
                str(config.simulations),
                "--seed",
                str(config.tactical_probe_seed),
                "--mlx-device",
                config.mlx_device,
                "--cache-entries",
                str(config.evaluator_cache_entries),
                "--out-dir",
                str(paths.tactical_probe),
            ]
        )

    if state.champion_checkpoint is not None:
        commands.append(
            [
                sys.executable,
                "-m",
                "dots_boxes_mcts.ez_checkpoint_eval",
                "--candidate",
                str(paths.checkpoint),
                "--baseline",
                str(state.champion_checkpoint),
                "--games",
                str(config.eval_champion_games),
                "--rows",
                str(config.rows),
                "--cols",
                str(config.cols),
                "--simulations",
                str(config.eval_champion_simulations),
                "--seed",
                str(config.eval_champion_seed or default_eval_seed(config.iteration)),
                "--mlx-device",
                config.mlx_device,
                "--evaluator-cache-entries",
                str(config.evaluator_cache_entries),
                "--out",
                str(paths.eval_champion),
            ]
        )
    return commands


def validate_inputs(config: EzFlywheelConfig, state: EzFlywheelState) -> None:
    missing: list[Path] = []
    init = training_init_checkpoint(config, state)
    if init is not None and not init.exists():
        missing.append(init)
    self_play = self_play_checkpoint(config, state)
    if self_play is None:
        raise FileNotFoundError(
            "Missing required EpsilonZero input: a network-guided self-play checkpoint. "
            "Run init-state with --champion-checkpoint or pass --init-checkpoint."
        )
    if not self_play.exists():
        missing.append(self_play)
    if config.tactical_suite is not None and not config.tactical_suite.exists():
        missing.append(config.tactical_suite)
    if missing:
        paths = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required EpsilonZero input(s):\n{paths}")


def validate_outputs(config: EzFlywheelConfig, paths: EzFlywheelPaths, overwrite: bool) -> None:
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
    if config.run_strategic_eval:
        outputs.extend([paths.strategic_summary, paths.unsafe_positions])
    existing = [path for path in outputs if path.exists()]
    if config.tactical_suite is not None and paths.tactical_probe.exists():
        existing.append(paths.tactical_probe)
    if existing:
        text = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing EpsilonZero flywheel output(s). "
            f"Pass --overwrite to replace them:\n{text}"
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


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf8"))


def read_evaluation_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    records = []
    with path.open(encoding="utf8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return summarize_checkpoint_match_records(records)


def record_completed_iteration(config: EzFlywheelConfig) -> EzFlywheelState:
    paths = ez_flywheel_paths(config)
    state = load_state(config.run_dir)
    evaluation = {
        "iteration": config.iteration,
        "candidateCheckpoint": str(paths.checkpoint),
        "strategicSummary": read_json(paths.strategic_summary)
        if config.run_strategic_eval
        else None,
        "tacticalProbe": read_json(paths.tactical_probe / "summary.json")
        if config.tactical_suite is not None
        else None,
        "championSummary": read_evaluation_summary(paths.eval_champion),
        "decision": "pending",
        "promoted": None,
    }
    next_state = EzFlywheelState(
        next_iteration=max(state.next_iteration, config.iteration + 1),
        champion_checkpoint=state.champion_checkpoint,
        latest_candidate_checkpoint=paths.checkpoint,
        last_evaluation=evaluation,
    )
    save_state(next_state, config.run_dir)
    append_history(
        {
            "event": "iteration_completed",
            "iteration": config.iteration,
            "outputs": {
                "games": str(paths.games),
                "selfPlayMetadata": str(paths.games.with_name(f"{paths.games.stem}.meta.json")),
                "examples": str(paths.examples),
                "checkpoint": str(paths.checkpoint),
                "diagnostics": str(paths.diagnostics),
                "strategicSummary": str(paths.strategic_summary)
                if config.run_strategic_eval
                else None,
                "unsafePositions": str(paths.unsafe_positions)
                if config.run_strategic_eval
                else None,
                "tacticalProbe": str(paths.tactical_probe) if config.tactical_suite else None,
                "evalChampion": str(paths.eval_champion),
            },
            "evaluation": evaluation,
        },
        config.run_dir,
    )
    return next_state


def promote_iteration(iteration: int, run_dir: Path, reason: str | None = None) -> EzFlywheelState:
    state = load_state(run_dir)
    candidate = state.latest_candidate_checkpoint
    if candidate is None:
        candidate = ez_flywheel_paths(EzFlywheelConfig(iteration=iteration, run_dir=run_dir)).checkpoint
    if not candidate.exists():
        raise FileNotFoundError(f"Missing EpsilonZero candidate checkpoint: {candidate}")
    evaluation = state.last_evaluation or {
        "iteration": iteration,
        "candidateCheckpoint": str(candidate),
    }
    evaluation = {
        **evaluation,
        "decision": "promoted",
        "promoted": True,
    }
    if reason:
        evaluation["reason"] = reason
    next_state = EzFlywheelState(
        next_iteration=max(state.next_iteration, iteration + 1),
        champion_checkpoint=candidate,
        latest_candidate_checkpoint=candidate,
        last_evaluation=evaluation,
    )
    save_state(next_state, run_dir)
    append_history(
        {
            "event": "candidate_promoted",
            "iteration": iteration,
            "championCheckpoint": str(candidate),
            "evaluation": evaluation,
        },
        run_dir,
    )
    return next_state


def reject_iteration(iteration: int, run_dir: Path, reason: str | None = None) -> EzFlywheelState:
    state = load_state(run_dir)
    candidate = state.latest_candidate_checkpoint
    if candidate is None:
        candidate = ez_flywheel_paths(EzFlywheelConfig(iteration=iteration, run_dir=run_dir)).checkpoint
    evaluation = state.last_evaluation or {
        "iteration": iteration,
        "candidateCheckpoint": str(candidate),
    }
    evaluation = {
        **evaluation,
        "decision": "rejected",
        "promoted": False,
    }
    if reason:
        evaluation["reason"] = reason
    next_state = EzFlywheelState(
        next_iteration=max(state.next_iteration, iteration + 1),
        champion_checkpoint=state.champion_checkpoint,
        latest_candidate_checkpoint=candidate,
        last_evaluation=evaluation,
    )
    save_state(next_state, run_dir)
    append_history(
        {
            "event": "candidate_rejected",
            "iteration": iteration,
            "championCheckpoint": str(state.champion_checkpoint),
            "candidateCheckpoint": str(candidate),
            "evaluation": evaluation,
        },
        run_dir,
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


def unsafe_selection_rate(evaluation: dict[str, Any] | None, simulations: int) -> float | None:
    if evaluation is None:
        return None
    tactical_probe = evaluation.get("tacticalProbe")
    if not isinstance(tactical_probe, list) or not tactical_probe:
        return None
    selected = None
    for item in tactical_probe:
        if isinstance(item, dict) and int(item.get("simulations", -1)) == simulations:
            selected = item
            break
    if selected is None:
        selected = tactical_probe[-1]
    if not isinstance(selected, dict) or "unsafeOpenerSelectionRate" not in selected:
        return None
    return float(selected["unsafeOpenerSelectionRate"])


def status_payload(run_dir: Path) -> dict[str, Any]:
    state = load_state(run_dir)
    return {
        "stageDir": str(run_dir),
        **state.to_json(),
    }


def run_next_iteration(
    config: EzFlywheelConfig,
    state: EzFlywheelState,
    *,
    dry_run: bool,
    overwrite: bool,
) -> EzFlywheelState | None:
    paths = ez_flywheel_paths(config)
    validate_inputs(config, state)
    validate_outputs(config, paths, overwrite=overwrite or dry_run)
    commands = command_plan(config, state=state)
    run_commands(commands, dry_run=dry_run)
    if dry_run:
        return None
    return record_completed_iteration(config)


def run_loop(args: argparse.Namespace) -> None:
    duration_seconds = parse_duration_seconds(args.duration)
    if args.iterations is None and duration_seconds is None:
        raise SystemExit("Pass --iterations or --duration.")
    if args.iterations is not None and args.iterations < 1:
        raise SystemExit("--iterations must be at least 1")
    if not 0.0 <= args.min_win_rate <= 1.0:
        raise SystemExit("--min-win-rate must be in [0, 1]")
    started_at = time.monotonic()
    deadline = None if duration_seconds is None else started_at + duration_seconds
    loop_index = 0
    while args.iterations is None or loop_index < args.iterations:
        if deadline is not None and time.monotonic() >= deadline:
            if loop_index == 0:
                print("Duration elapsed before any EpsilonZero iteration started.", flush=True)
            return
        loop_index += 1
        state = load_state(args.run_dir)
        iteration = state.next_iteration
        config = config_from_args(args, iteration=iteration)
        loop_total = str(args.iterations) if args.iterations is not None else f"duration {args.duration}"
        print(
            f"\nEpsilonZero loop iteration {loop_index}/{loop_total}: iter{iteration:03d}",
            flush=True,
        )
        next_state = run_next_iteration(
            config,
            state,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        if args.dry_run:
            print("Dry-run loop stops after the next planned EpsilonZero iteration.")
            return
        if next_state is None:
            raise RuntimeError("Completed iteration did not record EpsilonZero flywheel state.")
        evaluation = next_state.last_evaluation
        if evaluation is None:
            raise RuntimeError("Completed iteration did not record an evaluation.")
        summary = evaluation.get("championSummary")
        if not isinstance(summary, dict):
            raise RuntimeError("Completed iteration did not record a champion-match summary.")
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
        gate_text = (
            "Champion gate: "
            f"winRate={float(summary.get('winRate', 0.0)):.3f} "
            f"averageScoreMargin={float(summary.get('averageScoreMargin', 0.0)):.3f}"
        )
        unsafe_rate = unsafe_selection_rate(evaluation, config.simulations)
        if unsafe_rate is not None:
            gate_text += f" unsafeOpenerSelectionRate={unsafe_rate:.3f}"
        print(gate_text, flush=True)
        if promoted:
            promoted_state = promote_iteration(
                iteration=iteration,
                run_dir=args.run_dir,
                reason=reason,
            )
            print(f"Auto-promoted iteration {iteration}: {promoted_state.champion_checkpoint}")
        else:
            rejected_state = reject_iteration(
                iteration=iteration,
                run_dir=args.run_dir,
                reason=reason,
            )
            print(
                f"Auto-rejected iteration {iteration}. "
                f"Champion remains: {rejected_state.champion_checkpoint}"
            )


def add_shared_next_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--games", type=int, default=25)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=DEFAULT_EZ_FLYWHEEL_SIMULATIONS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--diagnostics-every", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--mlx-device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--run-dir", type=Path, default=EZ_FLYWHEEL_DIR)
    parser.add_argument(
        "--strategic-eval",
        action="store_true",
        help="Also summarize avoidable box giveaways and write an unsafe-opener suite.",
    )
    parser.add_argument(
        "--tactical-suite",
        type=Path,
        help="Optional unsafe-opener suite to probe with network-guided MCTS.",
    )
    parser.add_argument("--tactical-probe-seed", type=int, default=1)
    parser.add_argument("--eval-champion-games", type=int, default=20)
    parser.add_argument("--eval-champion-simulations", type=int, default=2000)
    parser.add_argument("--eval-champion-seed", type=int)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25)
    parser.add_argument("--temperature-moves", type=int, default=8)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument(
        "--evaluator-cache-entries",
        type=int,
        default=DEFAULT_EVALUATOR_CACHE_ENTRIES,
    )
    parser.add_argument("--quiet", action="store_true")


def config_from_args(args: argparse.Namespace, *, iteration: int) -> EzFlywheelConfig:
    return EzFlywheelConfig(
        iteration=iteration,
        run_dir=args.run_dir,
        games=args.games,
        rows=args.rows,
        cols=args.cols,
        simulations=args.simulations,
        seed=args.seed,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        diagnostics_every=args.diagnostics_every,
        hidden_size=args.hidden_size,
        residual_blocks=args.residual_blocks,
        mlx_device=args.mlx_device,
        debug=not args.quiet,
        init_checkpoint=args.init_checkpoint,
        run_strategic_eval=args.strategic_eval,
        tactical_suite=args.tactical_suite,
        tactical_probe_seed=args.tactical_probe_seed,
        eval_champion_games=args.eval_champion_games,
        eval_champion_simulations=args.eval_champion_simulations,
        eval_champion_seed=args.eval_champion_seed,
        c_puct=args.c_puct,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_exploration_fraction=args.root_exploration_fraction,
        temperature_moves=args.temperature_moves,
        sampling_temperature=args.sampling_temperature,
        evaluator_cache_entries=args.evaluator_cache_entries,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the independent EpsilonZero pure-restart pipeline.")
    subparsers = parser.add_subparsers(dest="command")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--run-dir", type=Path, default=EZ_FLYWHEEL_DIR)

    init_parser = subparsers.add_parser("init-state")
    init_parser.add_argument("--run-dir", type=Path, default=EZ_FLYWHEEL_DIR)
    init_parser.add_argument("--champion-checkpoint", type=Path)
    init_parser.add_argument(
        "--random-checkpoint",
        action="store_true",
        help="Create and use a random EpsilonZero policy/value checkpoint as the initial network.",
    )
    init_parser.add_argument("--rows", type=int, default=4)
    init_parser.add_argument("--cols", type=int, default=4)
    init_parser.add_argument("--hidden-size", type=int, default=64)
    init_parser.add_argument("--residual-blocks", type=int, default=4)
    init_parser.add_argument("--random-seed", type=int, default=1)
    init_parser.add_argument("--mlx-device", choices=["cpu", "gpu"], default="cpu")
    init_parser.add_argument("--checkpoint-out", type=Path)
    init_parser.add_argument("--overwrite", action="store_true")

    next_parser = subparsers.add_parser("next")
    add_shared_next_args(next_parser)
    next_parser.add_argument("--iteration", type=int)
    next_parser.add_argument("--dry-run", action="store_true")
    next_parser.add_argument("--overwrite", action="store_true")

    loop_parser = subparsers.add_parser("loop")
    add_shared_next_args(loop_parser)
    loop_parser.add_argument("--iterations", type=int)
    loop_parser.add_argument(
        "--duration",
        help="Run whole flywheel iterations until this time budget is reached, e.g. 30m, 12h, or 1h30m.",
    )
    loop_parser.add_argument("--min-win-rate", type=float, default=0.55)
    loop_parser.add_argument("--min-average-score-margin", type=float, default=0.0)
    loop_parser.add_argument("--dry-run", action="store_true")
    loop_parser.add_argument("--overwrite", action="store_true")

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--iteration", type=int, required=True)
    promote_parser.add_argument("--run-dir", type=Path, default=EZ_FLYWHEEL_DIR)
    promote_parser.add_argument("--reason")

    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(status_payload(args.run_dir), sort_keys=True))
        return

    if args.command == "init-state":
        path = ez_flywheel_state_path(args.run_dir)
        if path.exists() and not args.overwrite:
            raise SystemExit(
                f"EpsilonZero flywheel state already exists: {path}. Pass --overwrite to replace it."
            )
        if args.random_checkpoint and args.champion_checkpoint is not None:
            raise SystemExit("Use either --random-checkpoint or --champion-checkpoint, not both.")
        champion_checkpoint = args.champion_checkpoint
        if args.random_checkpoint:
            champion_checkpoint = args.checkpoint_out or random_checkpoint_path(
                rows=args.rows,
                cols=args.cols,
                seed=args.random_seed,
                run_dir=args.run_dir,
            )
            create_random_checkpoint(
                champion_checkpoint,
                rows=args.rows,
                cols=args.cols,
                hidden_size=args.hidden_size,
                residual_blocks=args.residual_blocks,
                seed=args.random_seed,
                device=args.mlx_device,
                overwrite=args.overwrite,
            )
        state = EzFlywheelState(champion_checkpoint=champion_checkpoint)
        save_state(state, args.run_dir)
        event = {
            "event": "state_initialized",
            "state": state.to_json(),
        }
        if args.random_checkpoint:
            event["randomCheckpoint"] = {
                "path": str(champion_checkpoint),
                "rows": args.rows,
                "cols": args.cols,
                "hiddenSize": args.hidden_size,
                "residualBlocks": args.residual_blocks,
                "seed": args.random_seed,
                "mlxDevice": args.mlx_device,
            }
        append_history(event, args.run_dir)
        print(json.dumps(state.to_json(), sort_keys=True))
        return

    if args.command == "next":
        state = load_state(args.run_dir)
        iteration = args.iteration or state.next_iteration
        config = config_from_args(args, iteration=iteration)
        next_state = run_next_iteration(
            config,
            state,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        if next_state is not None:
            print(json.dumps(next_state.to_json(), sort_keys=True))
        return

    if args.command == "loop":
        run_loop(args)
        return

    if args.command == "promote":
        state = promote_iteration(args.iteration, args.run_dir, reason=args.reason)
        print(json.dumps(state.to_json(), sort_keys=True))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
