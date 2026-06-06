from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_boxes_mcts.az_mcts import NetworkEvaluator, NetworkGuidedMCTS
from dots_boxes_mcts.evaluate import summarize_records
from dots_boxes_mcts.game import GameState, apply_move, new_game, state_snapshot
from dots_boxes_mcts.mcts import result_payload
from dots_boxes_mcts.self_play import write_jsonl


def play_checkpoint_match_game(
    candidate_checkpoint: Path,
    baseline_checkpoint: Path,
    rows: int = 4,
    cols: int = 4,
    simulations: int = 100,
    seed: int = 1,
    candidate_player: int = 0,
    c_puct: float = 1.5,
    device: str = "cpu",
) -> dict:
    if candidate_player not in {0, 1}:
        raise ValueError("candidate_player must be 0 or 1")

    evaluators = {
        candidate_player: NetworkEvaluator(checkpoint=candidate_checkpoint, device=device),
        1 - candidate_player: NetworkEvaluator(checkpoint=baseline_checkpoint, device=device),
    }
    searchers = {
        player: NetworkGuidedMCTS(
            evaluator=evaluator,
            simulations=simulations,
            c_puct=c_puct,
            seed=seed * 2 + player,
        )
        for player, evaluator in evaluators.items()
    }
    state = new_game(rows=rows, cols=cols)
    moves: list[str] = []
    decisions: list[dict] = []

    while not state.terminal:
        result = searchers[state.current_player].search(state)
        decisions.append(
            {
                "turn": len(moves),
                "player": state.current_player,
                "checkpointRole": (
                    "candidate" if state.current_player == candidate_player else "baseline"
                ),
                "state": state_snapshot(state),
                "search": result_payload(result),
            }
        )
        moves.append(result.move)
        state = apply_move(state, result.move)

    return checkpoint_match_record(
        state=state,
        moves=moves,
        seed=seed,
        candidate_checkpoint=candidate_checkpoint,
        baseline_checkpoint=baseline_checkpoint,
        candidate_player=candidate_player,
        simulations=simulations,
        c_puct=c_puct,
        decisions=decisions,
    )


def generate_checkpoint_match_games(
    candidate_checkpoint: Path,
    baseline_checkpoint: Path,
    games: int,
    rows: int = 4,
    cols: int = 4,
    simulations: int = 100,
    seed: int = 1,
    c_puct: float = 1.5,
    device: str = "cpu",
    alternate_colors: bool = True,
) -> list[dict]:
    records: list[dict] = []
    for game_index in range(games):
        candidate_player = game_index % 2 if alternate_colors else 0
        record = play_checkpoint_match_game(
            candidate_checkpoint=candidate_checkpoint,
            baseline_checkpoint=baseline_checkpoint,
            rows=rows,
            cols=cols,
            simulations=simulations,
            seed=seed + game_index,
            candidate_player=candidate_player,
            c_puct=c_puct,
            device=device,
        )
        record["gameIndex"] = game_index
        records.append(record)
    return records


def summarize_checkpoint_match_records(records: list[dict]) -> dict:
    if not records:
        return summarize_records([], mcts_player=0)

    candidate_margins = []
    wins = 0
    draws = 0
    for record in records:
        candidate_player = int(record["candidatePlayer"])
        baseline_player = 1 - candidate_player
        candidate_margins.append(
            record["finalScores"][candidate_player] - record["finalScores"][baseline_player]
        )
        if record["winner"] == candidate_player:
            wins += 1
        elif record["winner"] == "draw":
            draws += 1

    losses = len(records) - wins - draws
    return {
        "games": len(records),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "winRate": wins / len(records),
        "averageScoreMargin": sum(candidate_margins) / len(candidate_margins),
    }


def checkpoint_match_record(
    state: GameState,
    moves: list[str],
    seed: int,
    candidate_checkpoint: Path,
    baseline_checkpoint: Path,
    candidate_player: int,
    simulations: int,
    c_puct: float,
    decisions: list[dict],
) -> dict:
    baseline_player = 1 - candidate_player
    return {
        "seed": seed,
        "rows": state.rows,
        "cols": state.cols,
        "players": {
            str(candidate_player): "candidate_network_guided_mcts",
            str(baseline_player): "baseline_network_guided_mcts",
        },
        "dataSource": "checkpoint_match",
        "candidateCheckpoint": str(candidate_checkpoint),
        "baselineCheckpoint": str(baseline_checkpoint),
        "candidatePlayer": candidate_player,
        "simulations": simulations,
        "cPuct": c_puct,
        "moves": moves,
        "decisions": decisions,
        "finalScores": [state.scores[0], state.scores[1]],
        "winner": state.winner,
        "terminal": state.terminal,
        "state": state_snapshot(state),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a candidate checkpoint head-to-head against a baseline checkpoint."
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--mlx-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--no-alternate-colors", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/stage-3.6/checkpoint-match.jsonl"))
    args = parser.parse_args()

    if args.games < 1:
        raise SystemExit("--games must be at least 1")

    records = generate_checkpoint_match_games(
        candidate_checkpoint=args.candidate,
        baseline_checkpoint=args.baseline,
        games=args.games,
        rows=args.rows,
        cols=args.cols,
        simulations=args.simulations,
        seed=args.seed,
        c_puct=args.c_puct,
        device=args.mlx_device,
        alternate_colors=not args.no_alternate_colors,
    )
    write_jsonl(records, args.out)
    print(json.dumps(summarize_checkpoint_match_records(records), sort_keys=True))
    print(f"Wrote {len(records)} games to {args.out}")


if __name__ == "__main__":
    main()
