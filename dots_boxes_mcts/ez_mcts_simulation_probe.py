from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dots_boxes_mcts.ez_mcts import CachedNetworkEvaluator, NetworkEvaluator, NetworkGuidedMCTS
from dots_boxes_mcts.game import GameState
from dots_boxes_mcts.mcts import SearchResult
from dots_boxes_mcts.mcts_simulation_probe import (
    classify_move,
    compact_position,
    load_positions,
    parse_int_list,
    state_from_snapshot,
    summarize_probe,
    write_json,
    write_jsonl,
    write_svg_plot,
)

DEFAULT_SIMULATIONS = [5_000, 10_000, 15_000, 20_000, 30_000, 40_000, 50_000, 100_000]


def write_guided_summary_csv(path: Path, summaries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "simulations",
        "positions",
        "seeds",
        "trials",
        "safeOrScoringSelectionRate",
        "unsafeOpenerSelectionRate",
        "originalUnsafeMoveSelectionRate",
        "averageSafeOrScoringVisitShare",
        "averageUnsafeOpenerVisitShare",
        "elapsedSeconds",
        "cumulativeElapsedSeconds",
        "cacheHits",
        "cacheMisses",
    ]
    lines = [",".join(columns)]
    for summary in summaries:
        lines.append(",".join(str(summary.get(column, "")) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def result_record(
    position: dict,
    *,
    simulations: int,
    seed: int,
    state: GameState,
    result: SearchResult,
) -> dict:
    original_move = str(position.get("move", ""))
    category = classify_move(state, result.move)
    visits_by_category = {
        "safe": 0,
        "scoring": 0,
        "unsafe_opener": 0,
        "forced_opener": 0,
    }
    value_by_category = {key: [] for key in visits_by_category}
    for stat in result.stats:
        stat_category = classify_move(state, stat.move)
        visits_by_category[stat_category] += stat.visits
        value_by_category[stat_category].append(stat.mean_value)

    total_visits = sum(visits_by_category.values())
    safe_or_scoring_visits = visits_by_category["safe"] + visits_by_category["scoring"]
    unsafe_visits = visits_by_category["unsafe_opener"]
    return {
        "position": compact_position(position),
        "simulations": simulations,
        "backend": "network_guided",
        "trials": [
            {
                "seed": seed,
                "move": result.move,
                "category": category,
                "isSafeOrScoring": category in {"safe", "scoring"},
                "isUnsafeOpener": category == "unsafe_opener",
                "matchesOriginalUnsafeMove": bool(original_move and result.move == original_move),
                "safeOrScoringVisitShare": safe_or_scoring_visits / total_visits
                if total_visits
                else 0.0,
                "unsafeOpenerVisitShare": unsafe_visits / total_visits if total_visits else 0.0,
                "bestSafeOrScoringMeanValue": max(
                    value_by_category["safe"] + value_by_category["scoring"],
                    default=None,
                ),
                "bestUnsafeOpenerMeanValue": max(
                    value_by_category["unsafe_opener"],
                    default=None,
                ),
            }
        ],
    }


def probe_position(
    position: dict,
    *,
    simulations: int,
    seed: int,
    c_puct: float,
    evaluator: CachedNetworkEvaluator,
) -> dict:
    state = state_from_snapshot(position["state"])
    searcher = NetworkGuidedMCTS(
        evaluator=evaluator,  # type: ignore[arg-type]
        simulations=simulations,
        c_puct=c_puct,
        seed=seed,
    )
    result = searcher.search(state)
    return result_record(
        position,
        simulations=simulations,
        seed=seed,
        state=state,
        result=result,
    )


def probe_position_many(
    position: dict,
    *,
    simulations: list[int],
    seed: int,
    c_puct: float,
    evaluator: CachedNetworkEvaluator,
) -> tuple[list[dict], list[dict]]:
    state = state_from_snapshot(position["state"])
    searcher = NetworkGuidedMCTS(
        evaluator=evaluator,  # type: ignore[arg-type]
        simulations=max(simulations),
        c_puct=c_puct,
        seed=seed,
    )
    records: list[dict] = []
    timings: list[dict] = []
    started = time.perf_counter()
    last_time = started
    last_hits = evaluator.hits
    last_misses = evaluator.misses

    def on_budget(budget: int, result: SearchResult) -> None:
        nonlocal last_time, last_hits, last_misses
        now = time.perf_counter()
        records.append(
            result_record(
                position,
                simulations=budget,
                seed=seed,
                state=state,
                result=result,
            )
        )
        timings.append(
            {
                "simulations": budget,
                "elapsedSeconds": now - last_time,
                "cumulativeElapsedSeconds": now - started,
                "cacheHits": evaluator.hits - last_hits,
                "cacheMisses": evaluator.misses - last_misses,
            }
        )
        last_time = now
        last_hits = evaluator.hits
        last_misses = evaluator.misses

    searcher.search_many(state, simulations, on_budget=on_budget)
    return records, timings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe network-guided MCTS on known unsafe opener positions."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--inputs-are-positions",
        action="store_true",
        help="Treat inputs as JSONL files already produced by strategic_eval --suite-out.",
    )
    parser.add_argument("--simulations", type=parse_int_list, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit-positions", type=int)
    parser.add_argument("--position-index", type=int)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--mlx-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--cache-entries", type=int, default=500_000)
    parser.add_argument("--mlx-clear-cache-every", type=int, default=0)
    parser.add_argument("--mlx-cache-limit", type=int)
    parser.add_argument("--no-reuse-search-tree", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/stage-4/network-guided-unsafe-opener-probe"),
    )
    args = parser.parse_args()

    positions = load_positions(args.inputs, already_positions=args.inputs_are_positions)
    if args.position_index is not None:
        if args.position_index < 0 or args.position_index >= len(positions):
            raise SystemExit("--position-index is outside the available position range")
        positions = [positions[args.position_index]]
    if args.limit_positions is not None:
        positions = positions[: args.limit_positions]
    if not positions:
        raise SystemExit("No unsafe opener positions found.")

    network_evaluator = NetworkEvaluator(checkpoint=args.checkpoint, device=args.mlx_device)
    if args.mlx_cache_limit is not None:
        network_evaluator.mx.set_cache_limit(args.mlx_cache_limit)
    evaluator = CachedNetworkEvaluator(
        network_evaluator,
        max_entries=args.cache_entries,
        clear_mlx_cache_every=args.mlx_clear_cache_every,
    )
    results: list[dict] = []
    timings_by_budget: dict[int, dict] = {}
    if args.no_reuse_search_tree:
        for simulations in args.simulations:
            started = time.perf_counter()
            before_hits = evaluator.hits
            before_misses = evaluator.misses
            for position in positions:
                results.append(
                    probe_position(
                        position,
                        simulations=simulations,
                        seed=args.seed,
                        c_puct=args.c_puct,
                        evaluator=evaluator,
                    )
                )
            elapsed = time.perf_counter() - started
            timings_by_budget[simulations] = {
                "simulations": simulations,
                "elapsedSeconds": elapsed,
                "cumulativeElapsedSeconds": elapsed,
                "cacheHits": evaluator.hits - before_hits,
                "cacheMisses": evaluator.misses - before_misses,
            }
    else:
        for position in positions:
            position_results, position_timings = probe_position_many(
                position,
                simulations=args.simulations,
                seed=args.seed,
                c_puct=args.c_puct,
                evaluator=evaluator,
            )
            results.extend(position_results)
            for timing in position_timings:
                budget_timing = timings_by_budget.setdefault(
                    timing["simulations"],
                    {
                        "simulations": timing["simulations"],
                        "elapsedSeconds": 0.0,
                        "cumulativeElapsedSeconds": 0.0,
                        "cacheHits": 0,
                        "cacheMisses": 0,
                    },
                )
                budget_timing["elapsedSeconds"] += timing["elapsedSeconds"]
                budget_timing["cumulativeElapsedSeconds"] += timing["cumulativeElapsedSeconds"]
                budget_timing["cacheHits"] += timing["cacheHits"]
                budget_timing["cacheMisses"] += timing["cacheMisses"]

    timings = [
        timings_by_budget[simulations]
        for simulations in sorted(timings_by_budget)
    ]

    summaries = summarize_probe(results, positions=len(positions), seeds=[args.seed])
    timing_by_budget = {item["simulations"]: item for item in timings}
    for summary in summaries:
        summary.update(timing_by_budget.get(summary["simulations"], {}))

    payload = {
        "inputs": [str(path) for path in args.inputs],
        "inputsArePositions": args.inputs_are_positions,
        "checkpoint": str(args.checkpoint),
        "positions": len(positions),
        "positionIndex": args.position_index,
        "simulations": args.simulations,
        "seed": args.seed,
        "cPuct": args.c_puct,
        "mlxDevice": args.mlx_device,
        "backend": "network_guided",
        "cacheEntries": args.cache_entries,
        "mlxClearCacheEvery": args.mlx_clear_cache_every,
        "mlxCacheLimit": args.mlx_cache_limit,
        "reuseSearchTree": not args.no_reuse_search_tree,
        "summary": summaries,
    }

    write_json(args.out_dir / "summary.json", payload)
    write_jsonl(args.out_dir / "trials.jsonl", results)
    write_guided_summary_csv(args.out_dir / "summary.csv", summaries)
    write_svg_plot(
        args.out_dir / "curve.svg",
        summaries,
        title=f"Network-guided MCTS safety probe across {len(positions)} unsafe-opener positions",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
