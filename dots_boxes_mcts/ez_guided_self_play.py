from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Callable

from dots_boxes_mcts.ez_mcts import CachedNetworkEvaluator, NetworkEvaluator, NetworkGuidedMCTS
from dots_boxes_mcts.game import GameState, apply_move, new_game, state_snapshot
from dots_boxes_mcts.mcts import SearchResult, result_payload
from dots_boxes_mcts.self_play import write_jsonl

ProgressLogger = Callable[[str], None]
DEFAULT_TEMPERATURE_MOVES = 8
DEFAULT_SAMPLING_TEMPERATURE = 1.0
DEFAULT_EVALUATOR_CACHE_ENTRIES = 500_000


def play_guided_self_play_game(
    checkpoint: Path,
    rows: int = 4,
    cols: int = 4,
    simulations: int = 25,
    seed: int = 1,
    c_puct: float = 1.5,
    root_dirichlet_alpha: float = 0.3,
    root_exploration_fraction: float = 0.25,
    temperature_moves: int = DEFAULT_TEMPERATURE_MOVES,
    sampling_temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
    device: str = "cpu",
    reuse_tree: bool = False,
    evaluator_cache_entries: int = DEFAULT_EVALUATOR_CACHE_ENTRIES,
    progress_logger: ProgressLogger | None = None,
    game_index: int | None = None,
    total_games: int | None = None,
) -> dict:
    if temperature_moves < 0:
        raise ValueError("temperature_moves must be non-negative")
    if sampling_temperature <= 0:
        raise ValueError("sampling_temperature must be positive")

    move_rng = random.Random(seed)
    network_evaluator = NetworkEvaluator(checkpoint=checkpoint, device=device)
    evaluator = (
        CachedNetworkEvaluator(network_evaluator, max_entries=evaluator_cache_entries)
        if evaluator_cache_entries > 0
        else network_evaluator
    )
    searcher = NetworkGuidedMCTS(
        evaluator=evaluator,
        simulations=simulations,
        c_puct=c_puct,
        seed=seed,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_exploration_fraction=root_exploration_fraction,
    )
    state = new_game(rows=rows, cols=cols)
    moves: list[str] = []
    decisions: list[dict] = []
    game_start = time.perf_counter()

    while not state.terminal:
        turn_start = time.perf_counter()
        result = searcher.search_reusing_tree(state) if reuse_tree else searcher.search(state)
        search_seconds = time.perf_counter() - turn_start
        move, move_selection = select_self_play_move(
            result=result,
            turn=len(moves),
            rng=move_rng,
            temperature_moves=temperature_moves,
            sampling_temperature=sampling_temperature,
        )
        player = state.current_player
        scores_before = state.scores
        decisions.append(
            {
                "turn": len(moves),
                "player": player,
                "state": state_snapshot(state),
                "search": result_payload(result),
                "selectedMove": move,
                "searchPreferredMove": result.move,
                "moveSelection": move_selection,
                "samplingTemperature": sampling_temperature,
                "temperatureMoves": temperature_moves,
            }
        )
        moves.append(move)
        next_state = apply_move(state, move)
        if reuse_tree:
            searcher.advance_tree(move, next_state)
        state = next_state
        if progress_logger is not None:
            progress_logger(
                format_turn_progress(
                    game_index=game_index,
                    total_games=total_games,
                    seed=seed,
                    turn=len(moves) - 1,
                    player=player,
                    move=move,
                    search_seconds=search_seconds,
                    simulations=simulations,
                    legal_moves=len(result.stats),
                    scores_before=scores_before,
                    scores_after=state.scores,
                )
            )

    if progress_logger is not None:
        progress_logger(
            format_game_progress(
                game_index=game_index,
                total_games=total_games,
                seed=seed,
                moves=len(moves),
                elapsed_seconds=time.perf_counter() - game_start,
                final_scores=state.scores,
                winner=state.winner,
            )
        )

    return guided_self_play_record(
        state=state,
        moves=moves,
        seed=seed,
        checkpoint=checkpoint,
        simulations=simulations,
        c_puct=c_puct,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_exploration_fraction=root_exploration_fraction,
        temperature_moves=temperature_moves,
        sampling_temperature=sampling_temperature,
        reuse_tree=reuse_tree,
        decisions=decisions,
    )


def generate_guided_self_play_games(
    checkpoint: Path,
    games: int,
    rows: int = 4,
    cols: int = 4,
    simulations: int = 25,
    seed: int = 1,
    c_puct: float = 1.5,
    root_dirichlet_alpha: float = 0.3,
    root_exploration_fraction: float = 0.25,
    temperature_moves: int = DEFAULT_TEMPERATURE_MOVES,
    sampling_temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
    device: str = "cpu",
    reuse_tree: bool = False,
    evaluator_cache_entries: int = DEFAULT_EVALUATOR_CACHE_ENTRIES,
    progress_logger: ProgressLogger | None = None,
) -> list[dict]:
    records: list[dict] = []
    for game_index in range(games):
        record = play_guided_self_play_game(
            checkpoint=checkpoint,
            rows=rows,
            cols=cols,
            simulations=simulations,
            seed=seed + game_index,
            c_puct=c_puct,
            root_dirichlet_alpha=root_dirichlet_alpha,
            root_exploration_fraction=root_exploration_fraction,
            temperature_moves=temperature_moves,
            sampling_temperature=sampling_temperature,
            device=device,
            reuse_tree=reuse_tree,
            evaluator_cache_entries=evaluator_cache_entries,
            progress_logger=progress_logger,
            game_index=game_index,
            total_games=games,
        )
        record["gameIndex"] = game_index
        records.append(record)
    return records


def select_self_play_move(
    result: SearchResult,
    turn: int,
    rng: random.Random,
    temperature_moves: int = DEFAULT_TEMPERATURE_MOVES,
    sampling_temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
) -> tuple[str, str]:
    if turn >= temperature_moves:
        return result.move, "max_visit"

    weighted_moves = [
        (stat.move, float(stat.visits) ** (1.0 / sampling_temperature))
        for stat in result.stats
        if stat.visits > 0
    ]
    total_weight = sum(weight for _, weight in weighted_moves)
    if total_weight <= 0:
        return result.move, "max_visit"

    threshold = rng.random() * total_weight
    cumulative = 0.0
    for move, weight in weighted_moves:
        cumulative += weight
        if cumulative >= threshold:
            return move, "sampled_visit_counts"
    return weighted_moves[-1][0], "sampled_visit_counts"


def format_turn_progress(
    game_index: int | None,
    total_games: int | None,
    seed: int,
    turn: int,
    player: int,
    move: str,
    search_seconds: float,
    simulations: int,
    legal_moves: int,
    scores_before: tuple[int, int],
    scores_after: tuple[int, int],
) -> str:
    scored = scores_after != scores_before
    return (
        f"[guided-self-play] {format_game_label(game_index, total_games)} "
        f"seed={seed} turn={turn} player={player} move={move} "
        f"search={search_seconds:.3f}s simulations={simulations} "
        f"legal={legal_moves} scores={scores_after[0]}-{scores_after[1]} "
        f"scored={str(scored).lower()}"
    )


def format_game_progress(
    game_index: int | None,
    total_games: int | None,
    seed: int,
    moves: int,
    elapsed_seconds: float,
    final_scores: tuple[int, int],
    winner: int | str | None,
) -> str:
    return (
        f"[guided-self-play] {format_game_label(game_index, total_games)} "
        f"seed={seed} complete moves={moves} elapsed={elapsed_seconds:.3f}s "
        f"final={final_scores[0]}-{final_scores[1]} winner={winner}"
    )


def format_game_label(game_index: int | None, total_games: int | None) -> str:
    if game_index is None:
        return "game=?"
    if total_games is None:
        return f"game={game_index + 1}"
    return f"game={game_index + 1}/{total_games}"


def default_output_path(
    rows: int,
    cols: int,
    games: int,
    simulations: int,
    iteration: int | None = None,
) -> Path:
    parts = [
        "guided-self-play",
        f"{rows}x{cols}",
    ]
    if iteration is not None:
        parts.append(f"iter{iteration:03d}")
    parts.extend([f"games{games}", f"sims{simulations}"])
    return Path("runs/stage-3.6") / ("-".join(parts) + ".jsonl")


def metadata_output_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}.meta.json")


def run_metadata(
    out_path: Path,
    checkpoint: Path,
    games: int,
    rows: int,
    cols: int,
    simulations: int,
    iteration: int | None,
    seed: int,
    c_puct: float,
    root_dirichlet_alpha: float,
    root_exploration_fraction: float,
    temperature_moves: int,
    sampling_temperature: float,
    device: str,
    debug: bool,
    reuse_tree: bool,
    evaluator_cache_entries: int,
) -> dict:
    return {
        "output": str(out_path),
        "checkpoint": str(checkpoint),
        "games": games,
        "rows": rows,
        "cols": cols,
        "simulations": simulations,
        "iteration": iteration,
        "seed": seed,
        "cPuct": c_puct,
        "rootDirichletAlpha": root_dirichlet_alpha,
        "rootExplorationFraction": root_exploration_fraction,
        "temperatureMoves": temperature_moves,
        "samplingTemperature": sampling_temperature,
        "mlxDevice": device,
        "debug": debug,
        "reuseTree": reuse_tree,
        "evaluatorCacheEntries": evaluator_cache_entries,
    }


def write_metadata(metadata: dict, out_path: Path) -> Path:
    metadata_path = metadata_output_path(out_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return metadata_path


def ensure_outputs_do_not_exist(out_path: Path, overwrite: bool = False) -> None:
    if overwrite:
        return
    metadata_path = metadata_output_path(out_path)
    existing = [path for path in (out_path, metadata_path) if path.exists()]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing output: {paths}. "
            "Use --overwrite or choose a different --iteration/--out."
        )


def summarize_records(records: list[dict]) -> dict:
    if not records:
        return {
            "games": 0,
            "player0Wins": 0,
            "player1Wins": 0,
            "draws": 0,
            "averageScoreMarginForPlayer0": 0.0,
            "averageDecisionsPerGame": 0.0,
        }
    margins = [record["finalScores"][0] - record["finalScores"][1] for record in records]
    return {
        "games": len(records),
        "player0Wins": sum(1 for record in records if record["winner"] == 0),
        "player1Wins": sum(1 for record in records if record["winner"] == 1),
        "draws": sum(1 for record in records if record["winner"] == "draw"),
        "averageScoreMarginForPlayer0": sum(margins) / len(margins),
        "averageDecisionsPerGame": sum(len(record["decisions"]) for record in records)
        / len(records),
    }


def guided_self_play_record(
    state: GameState,
    moves: list[str],
    seed: int,
    checkpoint: Path,
    simulations: int,
    c_puct: float,
    root_dirichlet_alpha: float,
    root_exploration_fraction: float,
    temperature_moves: int,
    sampling_temperature: float,
    reuse_tree: bool,
    decisions: list[dict],
) -> dict:
    return {
        "seed": seed,
        "rows": state.rows,
        "cols": state.cols,
        "players": {
            "0": "network_guided_mcts",
            "1": "network_guided_mcts",
        },
        "dataSource": "network_guided_self_play",
        "checkpoint": str(checkpoint),
        "simulations": simulations,
        "cPuct": c_puct,
        "rootDirichletAlpha": root_dirichlet_alpha,
        "rootExplorationFraction": root_exploration_fraction,
        "temperatureMoves": temperature_moves,
        "samplingTemperature": sampling_temperature,
        "reuseTree": reuse_tree,
        "moves": moves,
        "decisions": decisions,
        "finalScores": [state.scores[0], state.scores[1]],
        "winner": state.winner,
        "terminal": state.terminal,
        "state": state_snapshot(state),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate network-guided MCTS self-play data.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=25)
    parser.add_argument("--iteration", type=int, help="Optional AlphaZero loop iteration label.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25)
    parser.add_argument("--temperature-moves", type=int, default=DEFAULT_TEMPERATURE_MOVES)
    parser.add_argument("--sampling-temperature", type=float, default=DEFAULT_SAMPLING_TEMPERATURE)
    parser.add_argument("--mlx-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--enable-tree-reuse",
        action="store_true",
        help="Retain the played child subtree between moves and run a full fresh budget.",
    )
    parser.add_argument(
        "--disable-tree-reuse",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--evaluator-cache-entries",
        type=int,
        default=DEFAULT_EVALUATOR_CACHE_ENTRIES,
        help="Maximum per-game network evaluation cache entries. Use 0 to disable.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-turn search timing and per-game completion progress to stderr.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output JSONL path. Defaults to a parameter-derived path under runs/stage-3.6/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing JSONL output and metadata sidecar.",
    )
    args = parser.parse_args()

    if args.games < 1:
        raise SystemExit("--games must be at least 1")
    if args.iteration is not None and args.iteration < 0:
        raise SystemExit("--iteration must be non-negative")
    if args.temperature_moves < 0:
        raise SystemExit("--temperature-moves must be non-negative")
    if args.sampling_temperature <= 0:
        raise SystemExit("--sampling-temperature must be positive")
    if args.evaluator_cache_entries < 0:
        raise SystemExit("--evaluator-cache-entries must be non-negative")

    out_path = args.out or default_output_path(
        rows=args.rows,
        cols=args.cols,
        games=args.games,
        simulations=args.simulations,
        iteration=args.iteration,
    )
    try:
        ensure_outputs_do_not_exist(out_path, overwrite=args.overwrite)
    except FileExistsError as error:
        raise SystemExit(str(error)) from error

    metadata = run_metadata(
        out_path=out_path,
        checkpoint=args.checkpoint,
        games=args.games,
        rows=args.rows,
        cols=args.cols,
        simulations=args.simulations,
        iteration=args.iteration,
        seed=args.seed,
        c_puct=args.c_puct,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_exploration_fraction=args.root_exploration_fraction,
        temperature_moves=args.temperature_moves,
        sampling_temperature=args.sampling_temperature,
        device=args.mlx_device,
        debug=args.debug,
        reuse_tree=args.enable_tree_reuse and not args.disable_tree_reuse,
        evaluator_cache_entries=args.evaluator_cache_entries,
    )
    records = generate_guided_self_play_games(
        checkpoint=args.checkpoint,
        games=args.games,
        rows=args.rows,
        cols=args.cols,
        simulations=args.simulations,
        seed=args.seed,
        c_puct=args.c_puct,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_exploration_fraction=args.root_exploration_fraction,
        temperature_moves=args.temperature_moves,
        sampling_temperature=args.sampling_temperature,
        device=args.mlx_device,
        reuse_tree=args.enable_tree_reuse and not args.disable_tree_reuse,
        evaluator_cache_entries=args.evaluator_cache_entries,
        progress_logger=(lambda message: print(message, file=sys.stderr)) if args.debug else None,
    )
    write_jsonl(records, out_path)
    metadata_path = write_metadata(metadata, out_path)
    print(json.dumps(summarize_records(records), sort_keys=True))
    print(f"Wrote {len(records)} games to {out_path}")
    print(f"Wrote run metadata to {metadata_path}")


if __name__ == "__main__":
    main()
