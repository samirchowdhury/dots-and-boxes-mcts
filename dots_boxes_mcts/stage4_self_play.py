from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

from dots_boxes_mcts.fast_mcts import FastUCTMCTS
from dots_boxes_mcts.game import GameState, apply_move, new_game, state_snapshot
from dots_boxes_mcts.mcts import result_payload
from dots_boxes_mcts.self_play import write_jsonl

ProgressLogger = Callable[[str], None]
DEFAULT_STAGE4_SIMULATIONS = 50_000


def play_stage4_self_play_game(
    rows: int = 4,
    cols: int = 4,
    simulations: int = DEFAULT_STAGE4_SIMULATIONS,
    seed: int = 1,
    exploration_constant: float = math.sqrt(2),
    progress_logger: ProgressLogger | None = None,
    game_index: int | None = None,
    total_games: int | None = None,
) -> dict:
    searchers = {
        0: FastUCTMCTS(
            simulations=simulations,
            exploration_constant=exploration_constant,
            seed=seed * 2,
        ),
        1: FastUCTMCTS(
            simulations=simulations,
            exploration_constant=exploration_constant,
            seed=seed * 2 + 1,
        ),
    }
    state = new_game(rows=rows, cols=cols)
    moves: list[str] = []
    decisions: list[dict] = []
    game_start = time.perf_counter()

    while not state.terminal:
        player = state.current_player
        turn_start = time.perf_counter()
        result = searchers[player].search(state)
        search_seconds = time.perf_counter() - turn_start
        move = result.move
        scores_before = state.scores
        decisions.append(
            {
                "turn": len(moves),
                "player": player,
                "state": state_snapshot(state),
                "search": result_payload(result),
                "selectedMove": move,
                "searchBackend": "numba_uct",
            }
        )
        moves.append(move)
        state = apply_move(state, move)
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

    return stage4_self_play_record(
        state=state,
        moves=moves,
        seed=seed,
        simulations=simulations,
        exploration_constant=exploration_constant,
        decisions=decisions,
    )


def generate_stage4_self_play_games(
    games: int,
    rows: int = 4,
    cols: int = 4,
    simulations: int = DEFAULT_STAGE4_SIMULATIONS,
    seed: int = 1,
    exploration_constant: float = math.sqrt(2),
    progress_logger: ProgressLogger | None = None,
) -> list[dict]:
    records: list[dict] = []
    for game_index in range(games):
        game_seed = seed + game_index
        record = play_stage4_self_play_game(
            rows=rows,
            cols=cols,
            simulations=simulations,
            seed=game_seed,
            exploration_constant=exploration_constant,
            progress_logger=progress_logger,
            game_index=game_index,
            total_games=games,
        )
        record["gameIndex"] = game_index
        records.append(record)
    return records


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


def stage4_self_play_record(
    state: GameState,
    moves: list[str],
    seed: int,
    simulations: int,
    exploration_constant: float,
    decisions: list[dict],
) -> dict:
    return {
        "seed": seed,
        "rows": state.rows,
        "cols": state.cols,
        "players": {
            "0": "stage4_numba_uct",
            "1": "stage4_numba_uct",
        },
        "dataSource": "stage4_numba_uct_self_play",
        "simulations": simulations,
        "explorationConstant": exploration_constant,
        "moves": moves,
        "decisions": decisions,
        "finalScores": [state.scores[0], state.scores[1]],
        "winner": state.winner,
        "terminal": state.terminal,
        "state": state_snapshot(state),
    }


def format_turn_progress(
    game_index: int | None,
    total_games: int | None,
    seed: int,
    turn: int,
    player: int,
    move: str,
    search_seconds: float,
    simulations: int,
    scores_before: tuple[int, int],
    scores_after: tuple[int, int],
) -> str:
    scored = scores_after != scores_before
    return (
        f"[stage4-self-play] {format_game_label(game_index, total_games)} "
        f"seed={seed} turn={turn} player={player} move={move} "
        f"search={search_seconds:.3f}s simulations={simulations} "
        f"scores={scores_after[0]}-{scores_after[1]} scored={str(scored).lower()}"
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
        f"[stage4-self-play] {format_game_label(game_index, total_games)} "
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
    *,
    rows: int,
    cols: int,
    games: int,
    simulations: int,
    iteration: int | None,
) -> Path:
    iteration_part = "bootstrap" if iteration is None else f"iter{iteration:03d}"
    return (
        Path("runs/stage-4")
        / f"stage4-self-play-{rows}x{cols}-{iteration_part}-games{games}-sims{simulations}.jsonl"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 4 high-simulation self-play data.")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=DEFAULT_STAGE4_SIMULATIONS)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--seed", type=int, default=42_001)
    parser.add_argument("--exploration-constant", type=float, default=math.sqrt(2))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.games < 1:
        raise SystemExit("--games must be at least 1")
    if args.simulations < 1:
        raise SystemExit("--simulations must be at least 1")

    out_path = args.out or default_output_path(
        rows=args.rows,
        cols=args.cols,
        games=args.games,
        simulations=args.simulations,
        iteration=args.iteration,
    )
    records = generate_stage4_self_play_games(
        games=args.games,
        rows=args.rows,
        cols=args.cols,
        simulations=args.simulations,
        seed=args.seed,
        exploration_constant=args.exploration_constant,
        progress_logger=(lambda message: print(message, file=sys.stderr, flush=True))
        if args.debug
        else None,
    )
    write_jsonl(records, out_path)
    print(json.dumps(summarize_records(records), sort_keys=True))
    print(f"Wrote {len(records)} games to {out_path}")


if __name__ == "__main__":
    main()
