from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from dots_boxes_mcts.ez_mcts import CachedNetworkEvaluator, NetworkEvaluator, NetworkGuidedMCTS
from dots_boxes_mcts.fast_ez_mcts import CPP_IMPORT_ERROR, FastNetworkGuidedMCTS
from dots_boxes_mcts.game import GameState, apply_move, legal_moves, new_game, state_snapshot
from dots_boxes_mcts.mcts import SearchResult


class UniformEvaluator:
    def evaluate(self, state: GameState) -> tuple[dict[str, float], float]:
        moves = legal_moves(state)
        return {move: 1.0 / len(moves) for move in moves}, 0.0

    def evaluate_snapshot(self, snapshot: dict) -> tuple[dict[str, float], float]:
        state = game_state_from_snapshot_for_profile(snapshot)
        return self.evaluate(state)

    def evaluate_many_snapshots(
        self,
        snapshots: list[dict],
    ) -> list[tuple[dict[str, float], float]]:
        return [self.evaluate_snapshot(snapshot) for snapshot in snapshots]


@dataclass(frozen=True)
class ProfileRecord:
    backend: str
    seconds: float
    searches: int
    simulations: int
    simulations_per_second: float
    selected_move: str
    evaluator_hits: int | None = None
    evaluator_misses: int | None = None


def game_state_from_snapshot_for_profile(snapshot: dict) -> GameState:
    return GameState(
        rows=int(snapshot["rows"]),
        cols=int(snapshot["cols"]),
        current_player=int(snapshot["currentPlayer"]),
        edges=frozenset(str(edge) for edge in snapshot["edges"]),
        edge_owners=tuple(
            (str(edge), int(owner))
            for edge, owner in snapshot.get("edgeOwners", [])
        ),
        boxes=tuple(
            tuple(None if cell is None else int(cell) for cell in row)
            for row in snapshot["boxes"]
        ),
        scores=(int(snapshot["scores"][0]), int(snapshot["scores"][1])),
        terminal=bool(snapshot.get("terminal", False)),
        winner=snapshot.get("winner"),
    )


def profiling_state(rows: int, cols: int, opening_moves: int, seed: int) -> GameState:
    state = new_game(rows=rows, cols=cols)
    if opening_moves <= 0:
        return state
    import random

    rng = random.Random(seed)
    for _ in range(opening_moves):
        if state.terminal:
            break
        state = apply_move(state, rng.choice(legal_moves(state)))
    return state


def make_evaluator(args: argparse.Namespace):
    if args.checkpoint is None:
        return UniformEvaluator()
    evaluator = NetworkEvaluator(checkpoint=args.checkpoint, device=args.mlx_device)
    if args.evaluator_cache_entries <= 0:
        return evaluator
    return CachedNetworkEvaluator(evaluator, max_entries=args.evaluator_cache_entries)


def run_backend(
    backend: str,
    evaluator,
    state: GameState,
    args: argparse.Namespace,
) -> ProfileRecord:
    if backend == "python":
        searcher = NetworkGuidedMCTS(
            evaluator=evaluator,
            simulations=args.simulations,
            c_puct=args.c_puct,
            seed=args.seed,
        )
    elif backend == "cpp":
        if CPP_IMPORT_ERROR is not None:
            raise ImportError(
                "C++ backend is not built. Run `python setup.py build_ext --inplace` first."
            ) from CPP_IMPORT_ERROR
        searcher = FastNetworkGuidedMCTS(
            evaluator=evaluator,
            simulations=args.simulations,
            c_puct=args.c_puct,
            seed=args.seed,
            batch_size=args.batch_size,
            virtual_loss=args.virtual_loss,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    samples: list[float] = []
    result: SearchResult | None = None
    for _ in range(args.warmup):
        searcher.search(state)
    for _ in range(args.repeat):
        start = time.perf_counter()
        result = searcher.search(state)
        samples.append(time.perf_counter() - start)

    assert result is not None
    seconds = statistics.mean(samples)
    return ProfileRecord(
        backend=backend,
        seconds=seconds,
        searches=args.repeat,
        simulations=args.simulations,
        simulations_per_second=args.simulations / seconds if seconds > 0 else float("inf"),
        selected_move=result.move,
        evaluator_hits=getattr(evaluator, "hits", None),
        evaluator_misses=getattr(evaluator, "misses", None),
    )


def record_payload(record: ProfileRecord) -> dict:
    return {
        "backend": record.backend,
        "seconds": record.seconds,
        "searches": record.searches,
        "simulations": record.simulations,
        "simulationsPerSecond": record.simulations_per_second,
        "selectedMove": record.selected_move,
        "evaluatorHits": record.evaluator_hits,
        "evaluatorMisses": record.evaluator_misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile Python and C++ network-guided MCTS backends."
    )
    parser.add_argument("--backend", choices=["python", "cpp", "both"], default="both")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mlx-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--opening-moves", type=int, default=0)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--virtual-loss", type=float, default=1.0)
    parser.add_argument("--evaluator-cache-entries", type=int, default=500_000)
    args = parser.parse_args()

    state = profiling_state(
        rows=args.rows,
        cols=args.cols,
        opening_moves=args.opening_moves,
        seed=args.seed,
    )
    backends = ["python", "cpp"] if args.backend == "both" else [args.backend]
    records = []
    for backend in backends:
        evaluator = make_evaluator(args)
        record = run_backend(backend, evaluator, state, args)
        records.append(record)
        print(json.dumps(record_payload(record), sort_keys=True))

    if len(records) == 2:
        baseline, candidate = records
        if baseline.backend == "python" and candidate.backend == "cpp":
            speedup = (
                baseline.seconds / candidate.seconds
                if candidate.seconds > 0
                else float("inf")
            )
            print(json.dumps({"cppSpeedupVsPython": speedup}, sort_keys=True))
    print(json.dumps({"state": state_snapshot(state)}, sort_keys=True))


if __name__ == "__main__":
    main()
