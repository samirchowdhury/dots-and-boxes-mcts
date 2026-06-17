from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dots_boxes_mcts.ez_mcts import CachedNetworkEvaluator, NetworkEvaluator, NetworkGuidedMCTS
from dots_boxes_mcts.fast_ez_mcts import FastNetworkGuidedMCTS
from dots_boxes_mcts.game import GameState, apply_move, legal_moves, new_game, state_snapshot
from dots_boxes_mcts.mcts import result_payload
from dots_boxes_mcts.mcts_vs_random import summarize_records
from dots_boxes_mcts.self_play import write_jsonl
from dots_boxes_mcts.strategic_eval import summarize_strategic_records

DEFAULT_EVALUATOR_CACHE_ENTRIES = 500_000


def play_checkpoint_match_game(
    candidate_checkpoint: Path,
    baseline_checkpoint: Path,
    rows: int = 4,
    cols: int = 4,
    simulations: int = 100,
    seed: int = 1,
    candidate_player: int = 0,
    c_puct: float = 1.5,
    opening_random_plies: int = 2,
    device: str = "cpu",
    reuse_tree: bool = False,
    evaluator_cache_entries: int = DEFAULT_EVALUATOR_CACHE_ENTRIES,
    mcts_backend: str = "python",
    mcts_batch_size: int = 8,
    virtual_loss: float = 1.0,
) -> dict:
    if candidate_player not in {0, 1}:
        raise ValueError("candidate_player must be 0 or 1")
    if opening_random_plies < 0:
        raise ValueError("opening_random_plies must be non-negative")

    rng = random.Random(seed)
    network_evaluators = {
        candidate_player: NetworkEvaluator(checkpoint=candidate_checkpoint, device=device),
        1 - candidate_player: NetworkEvaluator(checkpoint=baseline_checkpoint, device=device),
    }
    evaluators = {
        player: (
            CachedNetworkEvaluator(evaluator, max_entries=evaluator_cache_entries)
            if evaluator_cache_entries > 0
            else evaluator
        )
        for player, evaluator in network_evaluators.items()
    }
    searchers = {
        player: make_network_guided_searcher(
            evaluator=evaluator,
            backend=mcts_backend,
            simulations=simulations,
            c_puct=c_puct,
            seed=seed * 2 + player,
            batch_size=mcts_batch_size,
            virtual_loss=virtual_loss,
        )
        for player, evaluator in evaluators.items()
    }
    state = new_game(rows=rows, cols=cols)
    moves: list[str] = []
    opening_moves: list[str] = []
    decisions: list[dict] = []

    def advance_searchers(move: str, next_state: GameState) -> None:
        if reuse_tree:
            for searcher in searchers.values():
                searcher.advance_tree(move, next_state)

    for _ in range(opening_random_plies):
        if state.terminal:
            break
        move = rng.choice(legal_moves(state))
        opening_moves.append(move)
        moves.append(move)
        next_state = apply_move(state, move)
        advance_searchers(move, next_state)
        state = next_state

    while not state.terminal:
        searcher = searchers[state.current_player]
        result = searcher.search_reusing_tree(state) if reuse_tree else searcher.search(state)
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
        next_state = apply_move(state, result.move)
        advance_searchers(result.move, next_state)
        state = next_state

    return checkpoint_match_record(
        state=state,
        moves=moves,
        seed=seed,
        candidate_checkpoint=candidate_checkpoint,
        baseline_checkpoint=baseline_checkpoint,
        candidate_player=candidate_player,
        simulations=simulations,
        c_puct=c_puct,
        opening_random_plies=opening_random_plies,
        opening_moves=opening_moves,
        reuse_tree=reuse_tree,
        evaluator_cache_entries=evaluator_cache_entries,
        mcts_backend=mcts_backend,
        mcts_batch_size=mcts_batch_size,
        virtual_loss=virtual_loss,
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
    opening_random_plies: int = 2,
    device: str = "cpu",
    alternate_colors: bool = True,
    reuse_tree: bool = False,
    evaluator_cache_entries: int = DEFAULT_EVALUATOR_CACHE_ENTRIES,
    mcts_backend: str = "python",
    mcts_batch_size: int = 8,
    virtual_loss: float = 1.0,
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
            opening_random_plies=opening_random_plies,
            device=device,
            reuse_tree=reuse_tree,
            evaluator_cache_entries=evaluator_cache_entries,
            mcts_backend=mcts_backend,
            mcts_batch_size=mcts_batch_size,
            virtual_loss=virtual_loss,
        )
        record["gameIndex"] = game_index
        records.append(record)
    return records


def make_network_guided_searcher(
    *,
    evaluator,
    backend: str,
    simulations: int,
    c_puct: float,
    seed: int,
    batch_size: int,
    virtual_loss: float,
):
    if backend == "python":
        return NetworkGuidedMCTS(
            evaluator=evaluator,
            simulations=simulations,
            c_puct=c_puct,
            seed=seed,
        )
    if backend == "cpp":
        return FastNetworkGuidedMCTS(
            evaluator=evaluator,
            simulations=simulations,
            c_puct=c_puct,
            seed=seed,
            batch_size=batch_size,
            virtual_loss=virtual_loss,
        )
    raise ValueError(f"Unknown network-guided MCTS backend: {backend}")


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
        "strategic": summarize_strategic_records(
            records,
            perspective_player=lambda record: int(record["candidatePlayer"]),
        ),
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
    opening_random_plies: int = 0,
    opening_moves: list[str] | None = None,
    reuse_tree: bool = False,
    evaluator_cache_entries: int = DEFAULT_EVALUATOR_CACHE_ENTRIES,
    mcts_backend: str = "python",
    mcts_batch_size: int = 8,
    virtual_loss: float = 1.0,
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
        "reuseTree": reuse_tree,
        "evaluatorCacheEntries": evaluator_cache_entries,
        "mctsBackend": mcts_backend,
        "mctsBatchSize": mcts_batch_size,
        "virtualLoss": virtual_loss,
        "openingRandomPlies": opening_random_plies,
        "openingMoves": opening_moves or [],
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
    parser.add_argument("--opening-random-plies", type=int, default=2)
    parser.add_argument("--mlx-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--mcts-backend", choices=["python", "cpp"], default="python")
    parser.add_argument("--mcts-batch-size", type=int, default=8)
    parser.add_argument("--virtual-loss", type=float, default=1.0)
    parser.add_argument("--no-alternate-colors", action="store_true")
    parser.add_argument(
        "--enable-tree-reuse",
        action="store_true",
        help="Retain played subtrees between moves and run a full fresh budget.",
    )
    parser.add_argument("--disable-tree-reuse", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--evaluator-cache-entries", type=int, default=DEFAULT_EVALUATOR_CACHE_ENTRIES)
    parser.add_argument("--out", type=Path, default=Path("runs/stage-3.6/checkpoint-match.jsonl"))
    args = parser.parse_args()

    if args.games < 1:
        raise SystemExit("--games must be at least 1")
    if args.evaluator_cache_entries < 0:
        raise SystemExit("--evaluator-cache-entries must be non-negative")
    if args.mcts_batch_size < 1:
        raise SystemExit("--mcts-batch-size must be at least 1")
    if args.virtual_loss < 0:
        raise SystemExit("--virtual-loss must be non-negative")

    records = generate_checkpoint_match_games(
        candidate_checkpoint=args.candidate,
        baseline_checkpoint=args.baseline,
        games=args.games,
        rows=args.rows,
        cols=args.cols,
        simulations=args.simulations,
        seed=args.seed,
        c_puct=args.c_puct,
        opening_random_plies=args.opening_random_plies,
        device=args.mlx_device,
        alternate_colors=not args.no_alternate_colors,
        reuse_tree=args.enable_tree_reuse and not args.disable_tree_reuse,
        evaluator_cache_entries=args.evaluator_cache_entries,
        mcts_backend=args.mcts_backend,
        mcts_batch_size=args.mcts_batch_size,
        virtual_loss=args.virtual_loss,
    )
    write_jsonl(records, args.out)
    print(json.dumps(summarize_checkpoint_match_records(records), sort_keys=True))
    print(f"Wrote {len(records)} games to {args.out}")


if __name__ == "__main__":
    main()
