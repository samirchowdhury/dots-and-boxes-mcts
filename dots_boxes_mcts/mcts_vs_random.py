from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Literal

from dots_boxes_mcts.fast_mcts import FastUCTMCTS
from dots_boxes_mcts.game import GameState, apply_move, legal_moves, new_game, state_snapshot
from dots_boxes_mcts.mcts import UCTMCTS, result_payload
from dots_boxes_mcts.self_play import write_jsonl
from dots_boxes_mcts.strategic_eval import summarize_strategic_records

Backend = Literal["python", "numba"]


def make_searcher(
    *,
    backend: Backend,
    simulations: int,
    exploration_constant: float,
    seed: int,
) -> UCTMCTS | FastUCTMCTS:
    if backend == "python":
        return UCTMCTS(
            simulations=simulations,
            exploration_constant=exploration_constant,
            seed=seed,
        )
    if backend == "numba":
        return FastUCTMCTS(
            simulations=simulations,
            exploration_constant=exploration_constant,
            seed=seed,
        )
    raise ValueError(f"Unknown backend: {backend}")


def play_mcts_vs_random_game(
    rows: int = 3,
    cols: int = 3,
    simulations: int = 100,
    seed: int = 1,
    mcts_player: int = 0,
    exploration_constant: float = 2**0.5,
    backend: Backend = "python",
) -> dict:
    if mcts_player not in {0, 1}:
        raise ValueError("mcts_player must be 0 or 1")

    rng = random.Random(seed)
    mcts = make_searcher(
        backend=backend,
        simulations=simulations,
        exploration_constant=exploration_constant,
        seed=seed,
    )
    state = new_game(rows=rows, cols=cols)
    moves: list[str] = []
    decisions: list[dict] = []

    while not state.terminal:
        if state.current_player == mcts_player:
            result = mcts.search(state)
            move = result.move
            decisions.append(
                {
                    "turn": len(moves),
                    "player": state.current_player,
                    "state": state_snapshot(state),
                    "search": result_payload(result),
                }
            )
        else:
            move = rng.choice(legal_moves(state))

        moves.append(move)
        state = apply_move(state, move)

    return mcts_game_record(
        state=state,
        moves=moves,
        seed=seed,
        mcts_player=mcts_player,
        simulations=simulations,
        exploration_constant=exploration_constant,
        decisions=decisions,
    )


def generate_mcts_vs_random_games(
    games: int,
    rows: int = 3,
    cols: int = 3,
    simulations: int = 100,
    seed: int = 1,
    mcts_player: int = 0,
    exploration_constant: float = 2**0.5,
    backend: Backend = "python",
) -> list[dict]:
    records: list[dict] = []
    for game_index in range(games):
        game_seed = seed + game_index
        record = play_mcts_vs_random_game(
            rows=rows,
            cols=cols,
            simulations=simulations,
            seed=game_seed,
            mcts_player=mcts_player,
            exploration_constant=exploration_constant,
            backend=backend,
        )
        record["gameIndex"] = game_index
        records.append(record)
    return records


def summarize_records(records: list[dict], mcts_player: int = 0) -> dict:
    if not records:
        return {
            "games": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "winRate": 0.0,
            "averageScoreMargin": 0.0,
            "strategic": summarize_strategic_records([], perspective_player=mcts_player),
        }

    opponent = 1 if mcts_player == 0 else 0
    margins = [
        record["finalScores"][mcts_player] - record["finalScores"][opponent]
        for record in records
    ]
    wins = sum(1 for record in records if record["winner"] == mcts_player)
    draws = sum(1 for record in records if record["winner"] == "draw")
    losses = len(records) - wins - draws
    return {
        "games": len(records),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "winRate": wins / len(records),
        "averageScoreMargin": sum(margins) / len(margins),
        "strategic": summarize_strategic_records(records, perspective_player=mcts_player),
    }


def mcts_game_record(
    state: GameState,
    moves: list[str],
    seed: int,
    mcts_player: int,
    simulations: int,
    exploration_constant: float,
    decisions: list[dict],
) -> dict:
    return {
        "seed": seed,
        "rows": state.rows,
        "cols": state.cols,
        "players": {
            str(mcts_player): "uct_mcts",
            str(1 if mcts_player == 0 else 0): "random",
        },
        "mctsPlayer": mcts_player,
        "simulations": simulations,
        "explorationConstant": exploration_constant,
        "moves": moves,
        "decisions": decisions,
        "finalScores": [state.scores[0], state.scores[1]],
        "winner": state.winner,
        "terminal": state.terminal,
        "state": state_snapshot(state),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCTS against random play.")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--backend", choices=["python", "numba"], default="python")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mcts-player", type=int, choices=[0, 1], default=0)
    parser.add_argument("--exploration-constant", type=float, default=2**0.5)
    parser.add_argument("--out", type=Path, default=Path("runs/mcts-vs-random.jsonl"))
    args = parser.parse_args()

    if args.games < 1:
        raise SystemExit("--games must be at least 1")

    records = generate_mcts_vs_random_games(
        games=args.games,
        rows=args.rows,
        cols=args.cols,
        simulations=args.simulations,
        seed=args.seed,
        mcts_player=args.mcts_player,
        exploration_constant=args.exploration_constant,
        backend=args.backend,
    )
    write_jsonl(records, args.out)
    print(json.dumps(summarize_records(records, mcts_player=args.mcts_player), sort_keys=True))
    print(f"Wrote {len(records)} games to {args.out}")


if __name__ == "__main__":
    main()
