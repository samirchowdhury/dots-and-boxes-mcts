from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from dots_boxes_mcts.game import GameState, apply_move
from dots_boxes_mcts.fast_mcts import FastUCTMCTS
from dots_boxes_mcts.mcts import UCTMCTS
from dots_boxes_mcts.strategic_eval import (
    extract_unsafe_opener_positions,
    legal_move_profile,
    load_jsonl_records,
    new_three_sided_box_count,
)

DEFAULT_SIMULATIONS = [10, 25, 50, 100, 250, 500, 1000]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]


def state_from_snapshot(snapshot: dict) -> GameState:
    return GameState(
        rows=int(snapshot["rows"]),
        cols=int(snapshot["cols"]),
        current_player=int(snapshot["currentPlayer"]),
        edges=frozenset(str(edge) for edge in snapshot["edges"]),
        edge_owners=tuple((str(edge), int(owner)) for edge, owner in snapshot["edgeOwners"]),
        boxes=tuple(
            tuple(None if cell is None else int(cell) for cell in row)
            for row in snapshot["boxes"]
        ),
        scores=(int(snapshot["scores"][0]), int(snapshot["scores"][1])),
        terminal=bool(snapshot.get("terminal", False)),
        winner=snapshot.get("winner"),
    )


def load_positions(paths: list[Path], *, already_positions: bool) -> list[dict]:
    records = load_jsonl_records(paths)
    if already_positions:
        return records
    return extract_unsafe_opener_positions(records)


def classify_move(state: GameState, move: str) -> str:
    profile = legal_move_profile(state)
    next_state = apply_move(state, move)
    scored_boxes = len(next_state.history[-1]["boxes"])
    if scored_boxes:
        return "scoring"
    if new_three_sided_box_count(state, next_state) == 0:
        return "safe"
    if profile["safeMoves"]:
        return "unsafe_opener"
    return "forced_opener"


def probe_position(
    position: dict,
    *,
    simulations: int,
    seeds: list[int],
    exploration_constant: float,
    backend: str,
) -> dict:
    state = state_from_snapshot(position["state"])
    original_move = str(position.get("move", ""))
    trials: list[dict] = []
    for seed in seeds:
        searcher = make_searcher(
            backend=backend,
            simulations=simulations,
            exploration_constant=exploration_constant,
            seed=seed,
        )
        result = searcher.search(state)
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
        trials.append(
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
        )
    return {
        "position": compact_position(position),
        "simulations": simulations,
        "backend": backend,
        "trials": trials,
    }


def make_searcher(
    *,
    backend: str,
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


def compact_position(position: dict) -> dict:
    return {
        key: position[key]
        for key in (
            "recordIndex",
            "ply",
            "player",
            "move",
            "newThreeSidedBoxes",
            "safeMoves",
            "scoringMoves",
            "_path",
            "_line",
            "checkpoint",
            "bot",
            "opponent",
        )
        if key in position
    }


def summarize_probe(results: list[dict], *, positions: int, seeds: list[int]) -> list[dict]:
    summaries: list[dict] = []
    by_budget: dict[int, list[dict]] = {}
    for result in results:
        by_budget.setdefault(int(result["simulations"]), []).extend(result["trials"])

    for simulations in sorted(by_budget):
        trials = by_budget[simulations]
        total = len(trials)
        safe_or_scoring = sum(1 for trial in trials if trial["isSafeOrScoring"])
        unsafe = sum(1 for trial in trials if trial["isUnsafeOpener"])
        original = sum(1 for trial in trials if trial["matchesOriginalUnsafeMove"])
        summaries.append(
            {
                "simulations": simulations,
                "positions": positions,
                "seeds": len(seeds),
                "trials": total,
                "safeOrScoringSelections": safe_or_scoring,
                "unsafeOpenerSelections": unsafe,
                "originalUnsafeMoveSelections": original,
                "safeOrScoringSelectionRate": safe_or_scoring / total if total else 0.0,
                "unsafeOpenerSelectionRate": unsafe / total if total else 0.0,
                "originalUnsafeMoveSelectionRate": original / total if total else 0.0,
                "averageSafeOrScoringVisitShare": average(
                    trial["safeOrScoringVisitShare"] for trial in trials
                ),
                "averageUnsafeOpenerVisitShare": average(
                    trial["unsafeOpenerVisitShare"] for trial in trials
                ),
            }
        )
    return summaries


def average(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            output.write("\n")


def write_summary_csv(path: Path, summaries: list[dict]) -> None:
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
    ]
    lines = [",".join(columns)]
    for summary in summaries:
        lines.append(",".join(str(summary[column]) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def write_svg_plot(path: Path, summaries: list[dict], *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 860
    height = 520
    left = 80
    right = 40
    top = 55
    bottom = 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    budgets = [summary["simulations"] for summary in summaries]
    if not budgets:
        path.write_text("", encoding="utf8")
        return
    x_denominator = max(len(budgets) - 1, 1)

    def x_at(index: int) -> float:
        return left + plot_width * index / x_denominator

    def y_at(rate: float) -> float:
        return top + plot_height * (1.0 - rate)

    def points(key: str) -> str:
        return " ".join(
            f"{x_at(index):.1f},{y_at(float(summary[key])):.1f}"
            for index, summary in enumerate(summaries)
        )

    y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    x_tick_svg = []
    for index, budget in enumerate(budgets):
        x = x_at(index)
        x_tick_svg.append(
            f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" '
            f'y2="{top + plot_height + 6}" stroke="#334155" />'
        )
        x_tick_svg.append(
            f'<text x="{x:.1f}" y="{height - 35}" text-anchor="middle">{budget}</text>'
        )
    y_tick_svg = []
    for tick in y_ticks:
        y = y_at(tick)
        y_tick_svg.append(
            f'<line x1="{left - 6}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" '
            f'stroke="#334155" />'
        )
        y_tick_svg.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="#e2e8f0" />'
        )
        y_tick_svg.append(
            f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end">{tick:.0%}</text>'
        )
    safe_points = points("safeOrScoringSelectionRate")
    unsafe_points = points("unsafeOpenerSelectionRate")
    safe_dots = "\n".join(
        f'<circle cx="{x_at(index):.1f}" cy="{y_at(float(summary["safeOrScoringSelectionRate"])):.1f}" '
        'r="4" fill="#0f766e" />'
        for index, summary in enumerate(summaries)
    )
    unsafe_dots = "\n".join(
        f'<circle cx="{x_at(index):.1f}" cy="{y_at(float(summary["unsafeOpenerSelectionRate"])):.1f}" '
        'r="4" fill="#b91c1c" />'
        for index, summary in enumerate(summaries)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #0f172a; font-size: 13px; }}
    .title {{ font-size: 20px; font-weight: 700; }}
    .label {{ font-size: 14px; font-weight: 600; }}
    .legend {{ font-size: 13px; }}
  </style>
  <text class="title" x="{left}" y="30">{escape_xml(title)}</text>
  {''.join(y_tick_svg)}
  {''.join(x_tick_svg)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#334155" />
  <line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#334155" />
  <polyline points="{safe_points}" fill="none" stroke="#0f766e" stroke-width="3" />
  <polyline points="{unsafe_points}" fill="none" stroke="#b91c1c" stroke-width="3" />
  {safe_dots}
  {unsafe_dots}
  <text class="label" x="{width / 2:.1f}" y="{height - 8}" text-anchor="middle">MCTS simulations per decision</text>
  <text class="label" x="20" y="{top + plot_height / 2:.1f}" transform="rotate(-90 20 {top + plot_height / 2:.1f})" text-anchor="middle">Selection rate</text>
  <line x1="{width - 290}" y1="28" x2="{width - 260}" y2="28" stroke="#0f766e" stroke-width="3" />
  <text class="legend" x="{width - 250}" y="33">safe or scoring move</text>
  <line x1="{width - 290}" y1="50" x2="{width - 260}" y2="50" stroke="#b91c1c" stroke-width="3" />
  <text class="legend" x="{width - 250}" y="55">unsafe opener</text>
</svg>
'''
    path.write_text(svg, encoding="utf8")


def escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def limit_positions(positions: list[dict], limit: int | None) -> list[dict]:
    if limit is None:
        return positions
    return positions[:limit]


def parse_int_list(value: str) -> list[int]:
    parsed = [int(item) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe whether higher plain MCTS simulation budgets avoid known unsafe opener positions."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--inputs-are-positions",
        action="store_true",
        help="Treat inputs as JSONL files already produced by strategic_eval --suite-out.",
    )
    parser.add_argument("--simulations", type=parse_int_list, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--seeds", type=parse_int_list, default=DEFAULT_SEEDS)
    parser.add_argument("--limit-positions", type=int)
    parser.add_argument("--exploration-constant", type=float, default=2**0.5)
    parser.add_argument("--backend", choices=["python", "numba"], default="python")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/stage-3.8/mcts-simulation-probe"))
    args = parser.parse_args()

    positions = limit_positions(
        load_positions(args.inputs, already_positions=args.inputs_are_positions),
        args.limit_positions,
    )
    if not positions:
        raise SystemExit("No unsafe opener positions found.")

    results: list[dict] = []
    for position in positions:
        for simulations in args.simulations:
            results.append(
                probe_position(
                    position,
                    simulations=simulations,
                    seeds=args.seeds,
                    exploration_constant=args.exploration_constant,
                    backend=args.backend,
                )
            )

    summaries = summarize_probe(results, positions=len(positions), seeds=args.seeds)
    payload = {
        "inputs": [str(path) for path in args.inputs],
        "inputsArePositions": args.inputs_are_positions,
        "positions": len(positions),
        "simulations": args.simulations,
        "seeds": args.seeds,
        "explorationConstant": args.exploration_constant,
        "backend": args.backend,
        "summary": summaries,
    }

    write_json(args.out_dir / "summary.json", payload)
    write_jsonl(args.out_dir / "trials.jsonl", results)
    write_summary_csv(args.out_dir / "summary.csv", summaries)
    write_svg_plot(
        args.out_dir / "curve.svg",
        summaries,
        title=f"MCTS safety probe across {len(positions)} unsafe-opener positions",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
