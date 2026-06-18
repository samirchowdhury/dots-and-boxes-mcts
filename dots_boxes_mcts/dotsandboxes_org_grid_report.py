from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_GRID_DIR = Path("runs/dotsandboxes-org/ez-flywheel-grid")


@dataclass(frozen=True)
class GameRow:
    iteration: int
    checkpoint: str
    simulations: int
    site_think_time: float
    our_player: int
    seed: int
    win: int
    draw: int
    loss: int
    score_margin: int
    our_score: int
    opponent_score: int
    move_count: int
    decision_count: int
    winner: str
    grid_cell_key: str


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf8") as input_file:
        for line in input_file:
            if line.strip():
                records.append(json.loads(line))
    return records


def game_row(record: dict[str, Any]) -> GameRow:
    grid = record["grid"]
    our_player = int(record["ourPlayer"])
    opponent = 1 - our_player
    final_scores = record["finalScores"]
    winner = record["winner"]
    win = int(winner == our_player)
    draw = int(winner == "draw")
    loss = int(not win and not draw)
    return GameRow(
        iteration=int(grid["iteration"]),
        checkpoint=str(grid["checkpoint"]),
        simulations=int(grid["simulations"]),
        site_think_time=float(grid["siteThinkTime"]),
        our_player=our_player,
        seed=int(grid["seed"]),
        win=win,
        draw=draw,
        loss=loss,
        score_margin=int(final_scores[our_player]) - int(final_scores[opponent]),
        our_score=int(final_scores[our_player]),
        opponent_score=int(final_scores[opponent]),
        move_count=len(record["moves"]),
        decision_count=len(record.get("decisions", [])),
        winner=str(winner),
        grid_cell_key=str(record["gridCellKey"]),
    )


def write_game_csv(rows: list[GameRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(GameRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def cell_rows(rows: list[GameRow]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[int, int, float], dict[int, GameRow]] = defaultdict(dict)
    for row in rows:
        by_cell[(row.iteration, row.simulations, row.site_think_time)][row.our_player] = row

    output: list[dict[str, Any]] = []
    for (iteration, simulations, site_think_time), by_player in sorted(
        by_cell.items(),
        key=lambda item: (-item[0][0], item[0][1], item[0][2]),
    ):
        first = by_player.get(0)
        second = by_player.get(1)
        first_win = first.win if first else ""
        second_win = second.win if second else ""
        completed_roles = int(first is not None) + int(second is not None)
        wins = (first.win if first else 0) + (second.win if second else 0)
        output.append(
            {
                "iteration": iteration,
                "simulations": simulations,
                "site_think_time": site_think_time,
                "completed_roles": completed_roles,
                "first_player_win": first_win,
                "second_player_win": second_win,
                "combined_win_rate": wins / completed_roles if completed_roles else "",
                "first_player_margin": first.score_margin if first else "",
                "second_player_margin": second.score_margin if second else "",
            }
        )
    return output


def write_dict_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    with path.open("w", encoding="utf8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def frontier_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        grouped[(int(row["iteration"]), float(row["site_think_time"]))].append(row)

    output: list[dict[str, Any]] = []
    for (iteration, site_think_time), rows in sorted(grouped.items(), reverse=True):
        ordered = sorted(rows, key=lambda row: int(row["simulations"]))
        output.append(
            {
                "iteration": iteration,
                "site_think_time": site_think_time,
                "first_player_min_win_simulations": min_simulations(ordered, "first_player_win"),
                "second_player_min_win_simulations": min_simulations(
                    ordered,
                    "second_player_win",
                ),
                "both_roles_min_win_simulations": min_simulations(
                    ordered,
                    "combined_win_rate",
                    target=1.0,
                ),
                "any_role_min_win_simulations": min_simulations(
                    ordered,
                    "combined_win_rate",
                    target=0.5,
                ),
            }
        )
    return output


def min_simulations(rows: list[dict[str, Any]], field: str, target: float = 1.0) -> int | str:
    for row in rows:
        value = row[field]
        if value != "" and float(value) >= target:
            return int(row["simulations"])
    return ""


def write_combined_heatmaps(cells: list[dict[str, Any]], path: Path) -> None:
    iterations = sorted({int(row["iteration"]) for row in cells}, reverse=True)
    simulations = sorted({int(row["simulations"]) for row in cells})
    think_times = sorted({float(row["site_think_time"]) for row in cells})
    by_key = {
        (int(row["iteration"]), int(row["simulations"]), float(row["site_think_time"])): row
        for row in cells
    }

    panel_w = 320
    panel_h = 260
    cols = 3
    rows = (len(iterations) + cols - 1) // cols
    width = cols * panel_w + 60
    height = rows * panel_h + 100
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<text x="30" y="38" font-family="Arial" font-size="24" font-weight="700" '
        'fill="#26272b">dotsandboxes.org combined role win rate</text>',
        '<text x="30" y="64" font-family="Arial" font-size="14" fill="#686c73">'
        "Rows: site think time. Columns: EpsilonZero simulations. "
        "Red = 0 roles won, amber = 1 role won, green = both roles won.</text>",
    ]
    for index, iteration in enumerate(iterations):
        origin_x = 30 + (index % cols) * panel_w
        origin_y = 92 + (index // cols) * panel_h
        parts.append(
            f'<text x="{origin_x}" y="{origin_y - 16}" font-family="Arial" '
            f'font-size="15" font-weight="700" fill="#26272b">ITER={iteration:03d}</text>'
        )
        draw_heatmap_panel(
            parts=parts,
            origin_x=origin_x,
            origin_y=origin_y,
            simulations=simulations,
            think_times=think_times,
            value_for=lambda sims, think, iteration=iteration: by_key[
                (iteration, sims, think)
            ]["combined_win_rate"],
        )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf8")


def draw_heatmap_panel(
    *,
    parts: list[str],
    origin_x: int,
    origin_y: int,
    simulations: list[int],
    think_times: list[float],
    value_for,
) -> None:
    cell_w = 42
    cell_h = 30
    label_w = 48
    for row_index, think in enumerate(think_times):
        y = origin_y + row_index * cell_h
        parts.append(
            f'<text x="{origin_x}" y="{y + 20}" font-family="Arial" font-size="11" '
            f'fill="#686c73">{think:g}s</text>'
        )
        for col_index, sims in enumerate(simulations):
            x = origin_x + label_w + col_index * cell_w
            value = float(value_for(sims, think))
            color = heat_color(value)
            label = "0" if value == 0 else ("1/2" if value == 0.5 else "1")
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" '
                f'fill="{color}" rx="2"/>'
            )
            parts.append(
                f'<text x="{x + 11}" y="{y + 19}" font-family="Arial" font-size="11" '
                f'fill="white">{label}</text>'
            )
    label_y = origin_y + len(think_times) * cell_h + 16
    for col_index, sims in enumerate(simulations):
        x = origin_x + label_w + col_index * cell_w
        parts.append(
            f'<text x="{x}" y="{label_y}" font-family="Arial" font-size="9" '
            f'fill="#686c73">{sims}</text>'
        )


def heat_color(value: float) -> str:
    if value == 0:
        return "#c94f4f"
    if value == 0.5:
        return "#d99b3d"
    return "#3f995d"


def write_report(*, games_path: Path, out_dir: Path) -> None:
    records = read_jsonl(games_path)
    rows = [game_row(record) for record in records]
    cells = cell_rows(rows)
    frontier = frontier_rows(cells)
    write_game_csv(rows, out_dir / "games.csv")
    write_dict_csv(cells, out_dir / "cells.csv")
    write_dict_csv(frontier, out_dir / "frontier.csv")
    write_combined_heatmaps(cells, out_dir / "combined_heatmaps.svg")
    print(f"Wrote {len(rows)} game rows to {out_dir / 'games.csv'}")
    print(f"Wrote {len(cells)} cell rows to {out_dir / 'cells.csv'}")
    print(f"Wrote {len(frontier)} frontier rows to {out_dir / 'frontier.csv'}")
    print(f"Wrote heatmaps to {out_dir / 'combined_heatmaps.svg'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten and visualize a dotsandboxes.org grid evaluation."
    )
    parser.add_argument("--games", type=Path, default=DEFAULT_GRID_DIR / "games.jsonl")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_GRID_DIR / "report")
    args = parser.parse_args()

    if not args.games.exists():
        raise SystemExit(f"Input games JSONL does not exist: {args.games}")
    write_report(games_path=args.games, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
