from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


STAGE_DIR = Path("runs/stage-3.6")
STAGE_33_CHECKPOINT = Path("runs/stage-3.3/mlx-resconv-policy-value-4x4-1000.npz")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one champion-gated AlphaZero-style candidate iteration."
    )
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--champion-checkpoint", type=Path, default=STAGE_33_CHECKPOINT)
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
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    paths = flywheel_paths(config)
    if not args.dry_run:
        validate_inputs(config)
        validate_outputs(paths, overwrite=args.overwrite)

    commands = command_plan(config)
    if args.overwrite:
        commands[0].append("--overwrite")
    print(f"Flywheel iteration {config.iteration}")
    print(f"Champion checkpoint: {config.champion_checkpoint}")
    print(f"Training init checkpoint: {init_checkpoint(config)}")
    print(f"Self-play seed: {config.self_play_seed or default_self_play_seed(config.iteration)}")
    print(f"Evaluation seed: {config.eval_champion_seed or default_eval_champion_seed(config.iteration)}")
    run_commands(commands, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
