from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dots_boxes_mcts.az_mcts import NetworkEvaluator, NetworkGuidedMCTS
from dots_boxes_mcts.evaluate import summarize_records
from dots_boxes_mcts.game import GameState, apply_move, legal_moves, new_game, state_snapshot
from dots_boxes_mcts.mcts import UCTMCTS, result_payload
from dots_boxes_mcts.self_play import write_jsonl


def play_guided_vs_baseline_game(
    checkpoint: Path,
    opponent: str = "random",
    rows: int = 4,
    cols: int = 4,
    simulations: int = 25,
    opponent_simulations: int = 25,
    seed: int = 1,
    guided_player: int = 0,
    c_puct: float = 1.5,
    device: str = "cpu",
) -> dict:
    if guided_player not in {0, 1}:
        raise ValueError("guided_player must be 0 or 1")
    if opponent not in {"random", "plain_mcts"}:
        raise ValueError("opponent must be 'random' or 'plain_mcts'")

    rng = random.Random(seed)
    evaluator = NetworkEvaluator(checkpoint=checkpoint, device=device)
    guided = NetworkGuidedMCTS(evaluator=evaluator, simulations=simulations, c_puct=c_puct)
    plain = UCTMCTS(simulations=opponent_simulations, seed=seed) if opponent == "plain_mcts" else None
    state = new_game(rows=rows, cols=cols)
    moves: list[str] = []
    decisions: list[dict] = []

    while not state.terminal:
        if state.current_player == guided_player:
            result = guided.search(state)
            move = result.move
            decisions.append(
                {
                    "turn": len(moves),
                    "player": state.current_player,
                    "state": state_snapshot(state),
                    "search": result_payload(result),
                }
            )
        elif opponent == "plain_mcts":
            assert plain is not None
            move = plain.choose_move(state)
        else:
            move = rng.choice(legal_moves(state))

        moves.append(move)
        state = apply_move(state, move)

    return guided_game_record(
        state=state,
        moves=moves,
        seed=seed,
        checkpoint=checkpoint,
        opponent=opponent,
        guided_player=guided_player,
        simulations=simulations,
        opponent_simulations=opponent_simulations,
        c_puct=c_puct,
        decisions=decisions,
    )


def generate_guided_vs_baseline_games(
    checkpoint: Path,
    games: int,
    opponent: str = "random",
    rows: int = 4,
    cols: int = 4,
    simulations: int = 25,
    opponent_simulations: int = 25,
    seed: int = 1,
    guided_player: int = 0,
    c_puct: float = 1.5,
    device: str = "cpu",
) -> list[dict]:
    records: list[dict] = []
    for game_index in range(games):
        record = play_guided_vs_baseline_game(
            checkpoint=checkpoint,
            opponent=opponent,
            rows=rows,
            cols=cols,
            simulations=simulations,
            opponent_simulations=opponent_simulations,
            seed=seed + game_index,
            guided_player=guided_player,
            c_puct=c_puct,
            device=device,
        )
        record["gameIndex"] = game_index
        records.append(record)
    return records


def guided_game_record(
    state: GameState,
    moves: list[str],
    seed: int,
    checkpoint: Path,
    opponent: str,
    guided_player: int,
    simulations: int,
    opponent_simulations: int,
    c_puct: float,
    decisions: list[dict],
) -> dict:
    opponent_player = 1 if guided_player == 0 else 0
    opponent_name = "random" if opponent == "random" else "uct_mcts"
    return {
        "seed": seed,
        "rows": state.rows,
        "cols": state.cols,
        "players": {
            str(guided_player): "network_guided_mcts",
            str(opponent_player): opponent_name,
        },
        "checkpoint": str(checkpoint),
        "guidedPlayer": guided_player,
        "opponent": opponent,
        "simulations": simulations,
        "opponentSimulations": opponent_simulations,
        "cPuct": c_puct,
        "moves": moves,
        "decisions": decisions,
        "finalScores": [state.scores[0], state.scores[1]],
        "winner": state.winner,
        "terminal": state.terminal,
        "state": state_snapshot(state),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate network-guided MCTS against a baseline.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opponent", choices=["random", "plain_mcts"], default="random")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=25)
    parser.add_argument("--opponent-simulations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--guided-player", type=int, choices=[0, 1], default=0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--mlx-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--out", type=Path, default=Path("runs/stage-3.5/eval.jsonl"))
    args = parser.parse_args()

    records = generate_guided_vs_baseline_games(
        checkpoint=args.checkpoint,
        games=args.games,
        opponent=args.opponent,
        rows=args.rows,
        cols=args.cols,
        simulations=args.simulations,
        opponent_simulations=args.opponent_simulations,
        seed=args.seed,
        guided_player=args.guided_player,
        c_puct=args.c_puct,
        device=args.mlx_device,
    )
    write_jsonl(records, args.out)
    print(json.dumps(summarize_records(records, mcts_player=args.guided_player), sort_keys=True))
    print(f"Wrote {len(records)} games to {args.out}")


if __name__ == "__main__":
    main()
