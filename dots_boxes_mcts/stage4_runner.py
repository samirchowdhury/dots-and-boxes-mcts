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

STAGE4_DIR = Path("runs/stage-4")
STATE_FILENAME = "stage4-state.json"
HISTORY_FILENAME = "stage4-history.jsonl"
DEFAULT_TACTICAL_SUITE = Path("runs/stage-3.8/papg-stage3.6-unsafe-opener-positions.jsonl")


@dataclass(frozen=True)
class Stage4Paths:
    games: Path
    examples: Path
    checkpoint: Path
    diagnostics: Path
    strategic_summary: Path
    unsafe_positions: Path
    tactical_probe: Path
    eval_previous: Path


@dataclass(frozen=True)
class Stage4Config:
    iteration: int
    stage_dir: Path = STAGE4_DIR
    games: int = 25
    rows: int = 4
    cols: int = 4
    simulations: int = 50_000
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
    tactical_suite: Path | None = DEFAULT_TACTICAL_SUITE
    tactical_probe_seeds: str = "1,2,3"
    eval_previous_games: int = 100
    eval_previous_simulations: int = 100
    eval_previous_seed: int | None = None


@dataclass(frozen=True)
class Stage4State:
    next_iteration: int = 1
    champion_checkpoint: Path | None = None
    latest_candidate_checkpoint: Path | None = None
    last_evaluation: dict[str, Any] | None = None

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Stage4State":
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


def stage4_state_path(stage_dir: Path = STAGE4_DIR) -> Path:
    return stage_dir / STATE_FILENAME


def stage4_history_path(stage_dir: Path = STAGE4_DIR) -> Path:
    return stage_dir / HISTORY_FILENAME


def load_state(stage_dir: Path = STAGE4_DIR) -> Stage4State:
    path = stage4_state_path(stage_dir)
    if not path.exists():
        return Stage4State()
    return Stage4State.from_json(json.loads(path.read_text(encoding="utf8")))


def save_state(state: Stage4State, stage_dir: Path = STAGE4_DIR) -> None:
    path = stage4_state_path(stage_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf8")


def append_history(event: dict[str, Any], stage_dir: Path = STAGE4_DIR) -> None:
    path = stage4_history_path(stage_dir)
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


def stage4_paths(config: Stage4Config) -> Stage4Paths:
    label = iter_label(config.iteration)
    board = f"{config.rows}x{config.cols}"
    suffix = f"{board}-{label}-games{config.games}-sims{config.simulations}"
    checkpoint = (
        config.stage_dir
        / f"mlx-resconv-policy-value-{board}-{label}-pure-restart-sims{config.simulations}.npz"
    )
    return Stage4Paths(
        games=config.stage_dir / f"stage4-self-play-{suffix}.jsonl",
        examples=config.stage_dir / f"stage4-examples-{suffix}.jsonl",
        checkpoint=checkpoint,
        diagnostics=checkpoint.with_name(f"{checkpoint.stem}-diagnostics.jsonl"),
        strategic_summary=config.stage_dir / f"{label}-strategic-summary.json",
        unsafe_positions=config.stage_dir / f"{label}-unsafe-opener-positions.jsonl",
        tactical_probe=config.stage_dir / f"{label}-tactical-probe",
        eval_previous=config.stage_dir
        / f"{label}-vs-previous-stage4-sims{config.eval_previous_simulations}.jsonl",
    )


def training_init_checkpoint(config: Stage4Config, state: Stage4State) -> Path | None:
    if config.init_checkpoint is not None:
        return config.init_checkpoint
    return state.champion_checkpoint


def command_plan(config: Stage4Config, state: Stage4State | None = None) -> list[list[str]]:
    state = state or load_state(config.stage_dir)
    paths = stage4_paths(config)
    seed = config.seed or default_seed(config.iteration)
    commands: list[list[str]] = [
        [
            sys.executable,
            "-m",
            "dots_boxes_mcts.stage4_self_play",
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
            "--out",
            str(paths.games),
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
                "dots_boxes_mcts.mcts_simulation_probe",
                str(config.tactical_suite),
                "--inputs-are-positions",
                "--backend",
                "numba",
                "--simulations",
                str(config.simulations),
                "--seeds",
                config.tactical_probe_seeds,
                "--out-dir",
                str(paths.tactical_probe),
            ]
        )

    if state.champion_checkpoint is not None:
        commands.append(
            [
                sys.executable,
                "-m",
                "dots_boxes_mcts.az_checkpoint_eval",
                "--candidate",
                str(paths.checkpoint),
                "--baseline",
                str(state.champion_checkpoint),
                "--games",
                str(config.eval_previous_games),
                "--rows",
                str(config.rows),
                "--cols",
                str(config.cols),
                "--simulations",
                str(config.eval_previous_simulations),
                "--seed",
                str(config.eval_previous_seed or default_eval_seed(config.iteration)),
                "--mlx-device",
                config.mlx_device,
                "--out",
                str(paths.eval_previous),
            ]
        )
    return commands


def validate_inputs(config: Stage4Config, state: Stage4State) -> None:
    missing: list[Path] = []
    init = training_init_checkpoint(config, state)
    if init is not None and not init.exists():
        missing.append(init)
    if config.tactical_suite is not None and not config.tactical_suite.exists():
        missing.append(config.tactical_suite)
    if missing:
        paths = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required Stage 4 input(s):\n{paths}")


def validate_outputs(paths: Stage4Paths, overwrite: bool) -> None:
    if overwrite:
        return
    outputs = [
        paths.games,
        paths.examples,
        paths.checkpoint,
        paths.diagnostics,
        paths.strategic_summary,
        paths.unsafe_positions,
        paths.eval_previous,
    ]
    existing = [path for path in outputs if path.exists()]
    if paths.tactical_probe.exists():
        existing.append(paths.tactical_probe)
    if existing:
        text = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing Stage 4 output(s). "
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


def record_completed_iteration(config: Stage4Config) -> Stage4State:
    paths = stage4_paths(config)
    state = load_state(config.stage_dir)
    evaluation = {
        "iteration": config.iteration,
        "candidateCheckpoint": str(paths.checkpoint),
        "strategicSummary": read_json(paths.strategic_summary),
        "tacticalProbe": read_json(paths.tactical_probe / "summary.json"),
        "previousStage4Summary": read_evaluation_summary(paths.eval_previous),
        "decision": "pending",
        "promoted": None,
    }
    next_state = Stage4State(
        next_iteration=max(state.next_iteration, config.iteration + 1),
        champion_checkpoint=state.champion_checkpoint,
        latest_candidate_checkpoint=paths.checkpoint,
        last_evaluation=evaluation,
    )
    save_state(next_state, config.stage_dir)
    append_history(
        {
            "event": "iteration_completed",
            "iteration": config.iteration,
            "outputs": {
                "games": str(paths.games),
                "examples": str(paths.examples),
                "checkpoint": str(paths.checkpoint),
                "diagnostics": str(paths.diagnostics),
                "strategicSummary": str(paths.strategic_summary),
                "unsafePositions": str(paths.unsafe_positions),
                "tacticalProbe": str(paths.tactical_probe),
                "evalPrevious": str(paths.eval_previous),
            },
            "evaluation": evaluation,
        },
        config.stage_dir,
    )
    return next_state


def promote_iteration(iteration: int, stage_dir: Path, reason: str | None = None) -> Stage4State:
    state = load_state(stage_dir)
    candidate = state.latest_candidate_checkpoint
    if candidate is None:
        candidate = stage4_paths(Stage4Config(iteration=iteration, stage_dir=stage_dir)).checkpoint
    if not candidate.exists():
        raise FileNotFoundError(f"Missing Stage 4 candidate checkpoint: {candidate}")
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
    next_state = Stage4State(
        next_iteration=max(state.next_iteration, iteration + 1),
        champion_checkpoint=candidate,
        latest_candidate_checkpoint=candidate,
        last_evaluation=evaluation,
    )
    save_state(next_state, stage_dir)
    append_history(
        {
            "event": "candidate_promoted",
            "iteration": iteration,
            "championCheckpoint": str(candidate),
            "evaluation": evaluation,
        },
        stage_dir,
    )
    return next_state


def status_payload(stage_dir: Path) -> dict[str, Any]:
    state = load_state(stage_dir)
    return {
        "stageDir": str(stage_dir),
        **state.to_json(),
    }


def add_shared_next_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--games", type=int, default=25)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=50_000)
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
    parser.add_argument("--stage-dir", type=Path, default=STAGE4_DIR)
    parser.add_argument("--tactical-suite", type=Path, default=DEFAULT_TACTICAL_SUITE)
    parser.add_argument("--no-tactical-probe", action="store_true")
    parser.add_argument("--tactical-probe-seeds", default="1,2,3")
    parser.add_argument("--eval-previous-games", type=int, default=100)
    parser.add_argument("--eval-previous-simulations", type=int, default=100)
    parser.add_argument("--eval-previous-seed", type=int)
    parser.add_argument("--quiet", action="store_true")


def config_from_args(args: argparse.Namespace, *, iteration: int) -> Stage4Config:
    return Stage4Config(
        iteration=iteration,
        stage_dir=args.stage_dir,
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
        tactical_suite=None if args.no_tactical_probe else args.tactical_suite,
        tactical_probe_seeds=args.tactical_probe_seeds,
        eval_previous_games=args.eval_previous_games,
        eval_previous_simulations=args.eval_previous_simulations,
        eval_previous_seed=args.eval_previous_seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the independent Stage 4 pure-restart pipeline.")
    subparsers = parser.add_subparsers(dest="command")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--stage-dir", type=Path, default=STAGE4_DIR)

    init_parser = subparsers.add_parser("init-state")
    init_parser.add_argument("--stage-dir", type=Path, default=STAGE4_DIR)
    init_parser.add_argument("--champion-checkpoint", type=Path)
    init_parser.add_argument("--overwrite", action="store_true")

    next_parser = subparsers.add_parser("next")
    add_shared_next_args(next_parser)
    next_parser.add_argument("--iteration", type=int)
    next_parser.add_argument("--dry-run", action="store_true")
    next_parser.add_argument("--overwrite", action="store_true")

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--iteration", type=int, required=True)
    promote_parser.add_argument("--stage-dir", type=Path, default=STAGE4_DIR)
    promote_parser.add_argument("--reason")

    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(status_payload(args.stage_dir), sort_keys=True))
        return

    if args.command == "init-state":
        path = stage4_state_path(args.stage_dir)
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Stage 4 state already exists: {path}. Pass --overwrite to replace it.")
        state = Stage4State(champion_checkpoint=args.champion_checkpoint)
        save_state(state, args.stage_dir)
        append_history({"event": "state_initialized", "state": state.to_json()}, args.stage_dir)
        print(json.dumps(state.to_json(), sort_keys=True))
        return

    if args.command == "next":
        state = load_state(args.stage_dir)
        iteration = args.iteration or state.next_iteration
        config = config_from_args(args, iteration=iteration)
        paths = stage4_paths(config)
        validate_inputs(config, state)
        validate_outputs(paths, overwrite=args.overwrite or args.dry_run)
        commands = command_plan(config, state=state)
        run_commands(commands, dry_run=args.dry_run)
        if not args.dry_run:
            next_state = record_completed_iteration(config)
            print(json.dumps(next_state.to_json(), sort_keys=True))
        return

    if args.command == "promote":
        state = promote_iteration(args.iteration, args.stage_dir, reason=args.reason)
        print(json.dumps(state.to_json(), sort_keys=True))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
