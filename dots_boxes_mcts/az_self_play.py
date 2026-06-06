from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_boxes_mcts.game import GameState, apply_move, new_game, state_snapshot
from dots_boxes_mcts.mcts import UCTMCTS, result_payload
from dots_boxes_mcts.self_play import write_jsonl


def play_mcts_self_play_game(
    rows: int = 3,
    cols: int = 3,
    simulations: int = 25,
    seed: int = 1,
    exploration_constant: float = 2**0.5,
) -> dict:
    searchers = {
        0: UCTMCTS(
            simulations=simulations,
            exploration_constant=exploration_constant,
            seed=seed * 2,
        ),
        1: UCTMCTS(
            simulations=simulations,
            exploration_constant=exploration_constant,
            seed=seed * 2 + 1,
        ),
    }
    state = new_game(rows=rows, cols=cols)
    moves: list[str] = []
    decisions: list[dict] = []

    while not state.terminal:
        result = searchers[state.current_player].search(state)
        move = result.move
        decisions.append(
            {
                "turn": len(moves),
                "player": state.current_player,
                "state": state_snapshot(state),
                "search": result_payload(result),
            }
        )
        moves.append(move)
        state = apply_move(state, move)

    return mcts_self_play_record(
        state=state,
        moves=moves,
        seed=seed,
        simulations=simulations,
        exploration_constant=exploration_constant,
        decisions=decisions,
    )


def generate_mcts_self_play_games(
    games: int,
    rows: int = 3,
    cols: int = 3,
    simulations: int = 25,
    seed: int = 1,
    exploration_constant: float = 2**0.5,
) -> list[dict]:
    records: list[dict] = []
    for game_index in range(games):
        game_seed = seed + game_index
        record = play_mcts_self_play_game(
            rows=rows,
            cols=cols,
            simulations=simulations,
            seed=game_seed,
            exploration_constant=exploration_constant,
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


def mcts_self_play_record(
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
            "0": "uct_mcts",
            "1": "uct_mcts",
        },
        "dataSource": "mcts_self_play",
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
    parser = argparse.ArgumentParser(description="Generate MCTS-vs-MCTS self-play data.")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exploration-constant", type=float, default=2**0.5)
    parser.add_argument("--out", type=Path, default=Path("runs/stage-3.2/self-play-3x3.jsonl"))
    args = parser.parse_args()

    if args.games < 1:
        raise SystemExit("--games must be at least 1")

    records = generate_mcts_self_play_games(
        games=args.games,
        rows=args.rows,
        cols=args.cols,
        simulations=args.simulations,
        seed=args.seed,
        exploration_constant=args.exploration_constant,
    )
    write_jsonl(records, args.out)
    print(json.dumps(summarize_records(records), sort_keys=True))
    print(f"Wrote {len(records)} games to {args.out}")


if __name__ == "__main__":
    main()
