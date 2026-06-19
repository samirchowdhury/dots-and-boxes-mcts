from __future__ import annotations

import argparse
import csv
import json
import math
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
            value_for=lambda sims, think, iteration=iteration: (
                by_key.get((iteration, sims, think), {}).get("combined_win_rate", "")
            ),
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
            raw_value = value_for(sims, think)
            if raw_value == "":
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" '
                    'fill="#ece8df" rx="2"/>'
                )
                parts.append(
                    f'<text x="{x + 14}" y="{y + 19}" font-family="Arial" font-size="11" '
                    'fill="#9a948b">-</text>'
                )
                continue
            value = float(raw_value)
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


def role_color(value: int | str) -> str:
    return "#3f995d" if value == 1 else "#c94f4f"


def write_latest_checkpoint_heatmap(cells: list[dict[str, Any]], path: Path) -> None:
    latest_iteration = max(int(row["iteration"]) for row in cells)
    subset = [row for row in cells if int(row["iteration"]) == latest_iteration]
    simulations = sorted({int(row["simulations"]) for row in subset})
    think_times = sorted({float(row["site_think_time"]) for row in subset})
    by_key = {
        (int(row["simulations"]), float(row["site_think_time"])): row
        for row in subset
    }
    width, height = 620, 360
    parts = svg_header(
        width,
        height,
        "Latest checkpoint combined win rate",
        f"ITER={latest_iteration:03d}; rows are site think time, columns are simulations.",
    )
    draw_heatmap_panel(
        parts=parts,
        origin_x=64,
        origin_y=96,
        simulations=simulations,
        think_times=think_times,
        value_for=lambda sims, think: by_key.get((sims, think), {}).get("combined_win_rate", ""),
    )
    parts.append("</svg>")
    write_svg(path, parts)


def write_role_separated_heatmaps(cells: list[dict[str, Any]], path: Path) -> None:
    latest_iteration = max(int(row["iteration"]) for row in cells)
    subset = [row for row in cells if int(row["iteration"]) == latest_iteration]
    simulations = sorted({int(row["simulations"]) for row in subset})
    think_times = sorted({float(row["site_think_time"]) for row in subset})
    by_key = {
        (int(row["simulations"]), float(row["site_think_time"])): row
        for row in subset
    }
    width, height = 960, 380
    parts = svg_header(
        width,
        height,
        "Role-separated win maps",
        f"ITER={latest_iteration:03d}; green means EpsilonZero won that role.",
    )
    draw_binary_heatmap_panel(
        parts=parts,
        origin_x=64,
        origin_y=118,
        title="As Player 1",
        simulations=simulations,
        think_times=think_times,
        value_for=lambda sims, think: by_key.get((sims, think), {}).get("first_player_win", ""),
    )
    draw_binary_heatmap_panel(
        parts=parts,
        origin_x=500,
        origin_y=118,
        title="As Player 2",
        simulations=simulations,
        think_times=think_times,
        value_for=lambda sims, think: by_key.get((sims, think), {}).get("second_player_win", ""),
    )
    parts.append("</svg>")
    write_svg(path, parts)


def write_threshold_curves(frontier: list[dict[str, Any]], path: Path) -> None:
    iterations = sorted({int(row["iteration"]) for row in frontier})
    think_times = sorted({float(row["site_think_time"]) for row in frontier})
    highlighted = select_curve_iterations(iterations)
    width, height = 920, 560
    plot = PlotArea(x=86, y=84, width=650, height=380)
    parts = svg_header(
        width,
        height,
        "Minimum simulations for both-role wins",
        "Each line shows the first simulation count that wins as both Player 1 and Player 2.",
    )
    draw_log_sim_axes(parts, plot, think_times)
    colors = palette(len(highlighted))
    by_key = {
        (int(row["iteration"]), float(row["site_think_time"])): row
        for row in frontier
    }
    for color, iteration in zip(colors, highlighted):
        points = []
        for think in think_times:
            value = by_key[(iteration, think)]["both_roles_min_win_simulations"]
            if value == "":
                points.append((x_for_think(plot, think_times, think), plot.y - 18))
            else:
                points.append((x_for_think(plot, think_times, think), y_for_sim(plot, int(value))))
        draw_polyline(parts, points, color)
        label_x, label_y = points[-1]
        parts.append(svg_text(label_x + 12, label_y + 4, f"ITER={iteration:03d}", 12, color))
    parts.append(svg_text(292, 518, "site think time", 13, "#26272b"))
    parts.append(svg_text(18, 72, "min simulations", 13, "#26272b"))
    parts.append("</svg>")
    write_svg(path, parts)


def write_capability_frontier(frontier: list[dict[str, Any]], path: Path) -> None:
    iterations = sorted({int(row["iteration"]) for row in frontier})
    think_times = sorted({float(row["site_think_time"]) for row in frontier})
    highlighted = select_curve_iterations(iterations)
    width, height = 920, 560
    plot = PlotArea(x=86, y=84, width=650, height=380)
    parts = svg_header(
        width,
        height,
        "Capability frontier",
        "Minimum simulations needed to win at least one role; lower lines are stronger.",
    )
    draw_log_sim_axes(parts, plot, think_times)
    colors = palette(len(highlighted))
    by_key = {
        (int(row["iteration"]), float(row["site_think_time"])): row
        for row in frontier
    }
    for color, iteration in zip(colors, highlighted):
        points = []
        for think in think_times:
            value = by_key[(iteration, think)]["any_role_min_win_simulations"]
            if value == "":
                points.append((x_for_think(plot, think_times, think), plot.y - 18))
            else:
                points.append((x_for_think(plot, think_times, think), y_for_sim(plot, int(value))))
        draw_polyline(parts, points, color)
        label_x, label_y = points[-1]
        parts.append(svg_text(label_x + 12, label_y + 4, f"ITER={iteration:03d}", 12, color))
    parts.append(svg_text(292, 518, "site think time", 13, "#26272b"))
    parts.append(svg_text(18, 72, "min simulations", 13, "#26272b"))
    parts.append("</svg>")
    write_svg(path, parts)


def write_phase_diagram(cells: list[dict[str, Any]], path: Path) -> None:
    iterations = sorted({int(row["iteration"]) for row in cells})
    simulations = sorted({int(row["simulations"]) for row in cells})
    think_times = sorted({float(row["site_think_time"]) for row in cells})
    by_key = {
        (int(row["iteration"]), int(row["simulations"]), float(row["site_think_time"])): row
        for row in cells
    }
    width, height = 980, 660
    origin_x, origin_y = 500, 560
    sx, sy, sz = 62, 42, 1.0
    min_iteration, max_iteration = min(iterations), max(iterations)
    iteration_span = max_iteration - min_iteration
    parts = svg_header(
        width,
        height,
        "Win rate against dotsandboxes.org",
        "Projected grid of all cells; color is combined role win rate.",
    )
    parts.append(line(origin_x, origin_y, origin_x + 420, origin_y - 72, "#26272b", 2))
    parts.append(line(origin_x, origin_y, origin_x - 300, origin_y - 150, "#26272b", 2))
    parts.append(line(origin_x, origin_y, origin_x, 130, "#26272b", 2))
    parts.append(svg_text(origin_x + 392, origin_y - 84, "MCTS simulations", 13, "#26272b"))
    parts.append(svg_text(origin_x - 390, origin_y - 156, "Opponent think time", 13, "#26272b"))
    parts.append(svg_text(origin_x + 12, 120, "Model iteration", 13, "#26272b"))
    for iteration in iterations:
        z = ((iteration - min_iteration) / iteration_span) * 360 if iteration_span else 0
        for think_index, think in enumerate(think_times):
            for sim_index, simulations_value in enumerate(simulations):
                row = by_key.get((iteration, simulations_value, think))
                if row is None:
                    continue
                x = origin_x + sim_index * sx - think_index * sx * 0.62
                y = origin_y - sim_index * sy * 0.18 - think_index * sy * 0.42 - z * sz
                value = float(row["combined_win_rate"])
                radius = 4.5 if value == 0 else 6.5
                parts.append(circle(x, y, radius, heat_color(value), "white"))
    legend(parts, 42, 96)
    parts.append("</svg>")
    write_svg(path, parts)


def write_turntable_html(
    cells: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    path: Path,
) -> None:
    def optional_int(value: Any) -> int | None:
        return None if value == "" else int(value)

    plot_cells = [
        {
            "iteration": int(row["iteration"]),
            "simulations": int(row["simulations"]),
            "think": float(row["site_think_time"]),
            "combined": float(row["combined_win_rate"]),
            "p1": optional_int(row["first_player_win"]),
            "p2": optional_int(row["second_player_win"]),
            "p1Margin": optional_int(row["first_player_margin"]),
            "p2Margin": optional_int(row["second_player_margin"]),
        }
        for row in cells
    ]
    data_json = json.dumps(
        {"cells": plot_cells},
        separators=(",", ":"),
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Win rate against dotsandboxes.org</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family:
        Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      background: #07090d;
      color: #f4f7fb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      overflow: hidden;
      background:
        radial-gradient(circle at 50% 45%, rgba(73, 129, 178, 0.20), transparent 34%),
        radial-gradient(circle at 18% 16%, rgba(93, 201, 151, 0.13), transparent 28%),
        linear-gradient(150deg, #05070b 0%, #0a1119 52%, #05070b 100%);
    }}
    canvas {{
      display: block;
      width: 100vw;
      height: 100vh;
    }}
    .hud {{
      position: fixed;
      inset: 28px 32px auto 32px;
      display: flex;
      justify-content: space-between;
      gap: 18px;
      pointer-events: none;
    }}
    .title {{
      max-width: 780px;
      text-shadow: 0 10px 28px rgba(0, 0, 0, 0.42);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(30px, 4vw, 58px);
      line-height: 1.02;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin-top: 9px;
      max-width: 600px;
      color: rgba(230, 237, 245, 0.70);
      font-size: 15px;
      line-height: 1.45;
    }}
    .controls {{
      width: 306px;
      display: grid;
      gap: 10px;
      pointer-events: auto;
    }}
    .panel {{
      border: 1px solid rgba(255,255,255,0.12);
      background: linear-gradient(180deg, rgba(12, 20, 31, 0.72), rgba(5, 8, 13, 0.58));
      box-shadow: 0 22px 80px rgba(0,0,0,0.44);
      backdrop-filter: blur(20px);
      border-radius: 8px;
      padding: 12px;
    }}
    .segmented {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
    }}
    button {{
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.055);
      color: rgba(244,247,251,0.76);
      border-radius: 7px;
      height: 34px;
      cursor: pointer;
      font-weight: 650;
    }}
    button.active {{
      color: #081014;
      background: #9fe7c1;
      border-color: transparent;
    }}
    label {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: rgba(244,247,251,0.72);
      font-size: 13px;
      margin-top: 10px;
    }}
    input[type="range"] {{ width: 155px; accent-color: #9ad7b1; }}
    .caption {{
      position: fixed;
      left: 32px;
      bottom: 64px;
      color: rgba(244,247,251,0.72);
      font-size: 13px;
      line-height: 1.45;
      pointer-events: none;
      text-shadow: 0 8px 24px rgba(0,0,0,0.58);
    }}
    .legend {{
      position: fixed;
      left: 32px;
      bottom: 28px;
      display: flex;
      align-items: center;
      gap: 16px;
      color: rgba(244,247,251,0.72);
      font-size: 13px;
      pointer-events: none;
    }}
    .key {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
    }}
    .hint {{
      position: fixed;
      right: 32px;
      bottom: 28px;
      color: rgba(244,247,251,0.55);
      font-size: 13px;
      pointer-events: none;
    }}
    body.capture .controls,
    body.capture .hint {{
      display: none;
    }}
    body.capture .hud {{
      inset: 34px 40px auto 40px;
    }}
    body.capture .title {{
      max-width: 700px;
    }}
    body.capture h1 {{
      font-size: clamp(34px, 5vw, 58px);
    }}
    body.capture .subtitle {{
      max-width: 590px;
      font-size: 14px;
    }}
    body.capture .caption {{
      left: 40px;
      bottom: 76px;
    }}
    body.capture .legend {{
      left: 40px;
      bottom: 38px;
    }}
  </style>
</head>
<body>
  <canvas id="phase"></canvas>
  <div class="hud">
    <div class="title">
      <h1>Win rate against dotsandboxes.org</h1>
    </div>
    <div class="controls panel">
      <div class="segmented">
        <button data-mode="combined" class="active">Combined</button>
        <button data-mode="p1">Player 1</button>
        <button data-mode="p2">Player 2</button>
      </div>
      <label>Turntable <input id="auto" type="checkbox" checked></label>
    </div>
  </div>
  <div class="caption">Opponent think time &times; MCTS simulations &times; Model iteration</div>
  <div class="legend">
    <span class="key">
      <span class="swatch" style="color:#e1605d;background:#e1605d"></span>loss
    </span>
    <span class="key">
      <span class="swatch" style="color:#e0aa48;background:#e0aa48"></span>split
    </span>
    <span class="key">
      <span class="swatch" style="color:#56ff97;background:#56ff97"></span>win (both Player 1 and Player 2)
    </span>
  </div>
  <div class="hint">drag to rotate</div>
  <script>
    const DATA = {data_json};
    const PARAMS = new URLSearchParams(location.search);
    const captureMode = PARAMS.has("capture");
    if (captureMode) document.body.classList.add("capture");
    const canvas = document.getElementById("phase");
    const ctx = canvas.getContext("2d");
    const controls = {{
      mode: "combined",
      auto: true,
      yaw: captureMode ? -0.54 : -0.62,
      pitch: captureMode ? 0.50 : 0.54,
    }};
    const viewScale = captureMode ? 1.08 : 1;
    const ranges = {{
      iter: extent(DATA.cells.map(d => d.iteration)),
      sim: extent(DATA.cells.map(d => Math.log10(d.simulations))),
      think: extent(DATA.cells.map(d => Math.log10(d.think))),
      thinkLinear: extent(DATA.cells.map(d => d.think)),
    }};
    const ticks = {{
      think: uniqueSorted(DATA.cells.map(d => d.think)),
      simulations: uniqueSorted(DATA.cells.map(d => d.simulations)),
      iterations: selectTicks(uniqueSorted(DATA.cells.map(d => d.iteration)), 6),
    }};
    const annotationTargets = DATA.cells.filter(cell =>
      cell.iteration === 542 &&
      cell.simulations >= 5000 &&
      cell.combined >= 1
    );

    let pointer = null;
    let frame = 0;
    let lastDraw = 0;
    const targetFrameMs = captureMode ? 1000 / 30 : 1000 / 24;

    document.querySelectorAll("button[data-mode]").forEach(button => {{
      button.addEventListener("click", () => {{
        document.querySelectorAll("button[data-mode]").forEach(item => {{
          item.classList.remove("active");
        }});
        button.classList.add("active");
        controls.mode = button.dataset.mode;
      }});
    }});
    document.getElementById("auto").addEventListener(
      "change",
      event => controls.auto = event.target.checked
    );

    canvas.addEventListener("pointerdown", event => {{
      pointer = {{ x: event.clientX, y: event.clientY, yaw: controls.yaw, pitch: controls.pitch }};
      canvas.setPointerCapture(event.pointerId);
    }});
    canvas.addEventListener("pointermove", event => {{
      if (!pointer) return;
      controls.yaw = pointer.yaw + (event.clientX - pointer.x) * 0.008;
      controls.pitch = clamp(pointer.pitch + (event.clientY - pointer.y) * 0.006, -0.95, 1.05);
    }});
    canvas.addEventListener("pointerup", () => pointer = null);
    function extent(values) {{
      return [Math.min(...values), Math.max(...values)];
    }}
    function uniqueSorted(values) {{
      return [...new Set(values)].sort((a, b) => a - b);
    }}
    function selectTicks(values, maxCount) {{
      if (values.length <= maxCount) return values;
      const selected = [];
      for (let i = 0; i < maxCount; i++) {{
        selected.push(values[Math.round(i * (values.length - 1) / (maxCount - 1))]);
      }}
      return [...new Set(selected)];
    }}
    function clamp(value, min, max) {{
      return Math.max(min, Math.min(max, value));
    }}
    function norm(value, range) {{
      const span = range[1] - range[0];
      return span === 0 ? 0.5 : (value - range[0]) / span;
    }}
    function cellPoint(cell) {{
      return {{
        x: (norm(cell.think, ranges.thinkLinear) - 0.5) * 5.6,
        y: (norm(Math.log10(cell.simulations), ranges.sim) - 0.5) * 5.6,
        z: (norm(cell.iteration, ranges.iter) - 0.5) * 5.6,
        cell,
      }};
    }}
    function project(point) {{
      const cy = Math.cos(controls.yaw);
      const sy = Math.sin(controls.yaw);
      const cp = Math.cos(controls.pitch);
      const sp = Math.sin(controls.pitch);
      const x1 = point.x * cy - point.z * sy;
      const z1 = point.x * sy + point.z * cy;
      const y1 = point.y * cp - z1 * sp;
      const z2 = point.y * sp + z1 * cp;
      const scale = Math.min(innerWidth, innerHeight) * 0.113 * viewScale;
      return {{
        x: innerWidth * 0.52 + x1 * scale,
        y: innerHeight * 0.59 - y1 * scale,
        z: z2,
        p: 1,
      }};
    }}
    function outcome(cell) {{
      if (controls.mode === "p1") return cell.p1;
      if (controls.mode === "p2") return cell.p2;
      return cell.combined;
    }}
    function colorFor(value, alpha = 1) {{
      if (value === null || value === undefined) return `rgba(148,163,184,${{alpha}})`;
      if (value >= 1) return `rgba(86,255,151,${{alpha}})`;
      if (value >= 0.5) return `rgba(224,170,72,${{alpha}})`;
      return `rgba(225,96,93,${{alpha}})`;
    }}
    function resize() {{
      const dpr = Math.min(devicePixelRatio || 1, captureMode ? 1.5 : 1);
      canvas.width = Math.floor(innerWidth * dpr);
      canvas.height = Math.floor(innerHeight * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }}
    addEventListener("resize", resize);
    resize();

    function draw(timestamp = 0) {{
      if (timestamp - lastDraw < targetFrameMs) {{
        requestAnimationFrame(draw);
        return;
      }}
      lastDraw = timestamp;
      frame += 1;
      if (controls.auto && !pointer) controls.yaw += captureMode ? 0.0045 : 0.006;
      ctx.clearRect(0, 0, innerWidth, innerHeight);
      drawBackdrop();
      drawAxes();
      drawPoints();
      drawAnnotations();
      drawVignette();
      requestAnimationFrame(draw);
    }}
    function drawBackdrop() {{
      const gradient = ctx.createRadialGradient(
        innerWidth * 0.52,
        innerHeight * 0.45,
        0,
        innerWidth * 0.52,
        innerHeight * 0.45,
        innerWidth * 0.72
      );
      gradient.addColorStop(0, "rgba(34, 52, 72, 0.38)");
      gradient.addColorStop(1, "rgba(7, 9, 13, 0)");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, innerWidth, innerHeight);
    }}
    function drawVignette() {{
      const gradient = ctx.createRadialGradient(
        innerWidth * 0.5,
        innerHeight * 0.5,
        innerWidth * 0.18,
        innerWidth * 0.5,
        innerHeight * 0.5,
        innerWidth * 0.74
      );
      gradient.addColorStop(0, "rgba(0,0,0,0)");
      gradient.addColorStop(1, "rgba(0,0,0,0.42)");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, innerWidth, innerHeight);
    }}
    function drawAxes() {{
      const origin = {{x:-2.8,y:-2.8,z:-2.8}};
      drawCubeFrame();
      const axes = [
        {{ a: origin, b: {{x:2.8,y:-2.8,z:-2.8}}, label:"Opponent think time", color:"rgba(119, 197, 255, 0.70)" }},
        {{ a: origin, b: {{x:-2.8,y:2.8,z:-2.8}}, label:"MCTS simulations", color:"rgba(159, 231, 193, 0.74)" }},
        {{ a: origin, b: {{x:-2.8,y:-2.8,z:2.8}}, label:"Model iteration", color:"rgba(255, 205, 112, 0.72)" }},
      ];
      ctx.lineWidth = 1;
      axes.forEach(axis => {{
        const a = project(axis.a);
        const b = project(axis.b);
        ctx.strokeStyle = axis.color;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        ctx.fillStyle = "rgba(236, 243, 250, 0.72)";
        ctx.font = "600 13px Inter, system-ui, sans-serif";
        ctx.textAlign = b.x < a.x ? "right" : "left";
        ctx.fillText(axis.label, b.x + 8, b.y - 6);
      }});
      ctx.textAlign = "left";
      drawGridPlane();
      drawAxisTicks();
    }}
    function drawCubeFrame() {{
      const min = -2.8;
      const max = 2.8;
      const corners = [
        {{x:min,y:min,z:min}}, {{x:max,y:min,z:min}},
        {{x:max,y:max,z:min}}, {{x:min,y:max,z:min}},
        {{x:min,y:min,z:max}}, {{x:max,y:min,z:max}},
        {{x:max,y:max,z:max}}, {{x:min,y:max,z:max}},
      ].map(project);
      const edges = [
        [0,1], [1,2], [2,3], [3,0],
        [4,5], [5,6], [6,7], [7,4],
        [0,4], [1,5], [2,6], [3,7],
      ];
      ctx.save();
      ctx.strokeStyle = "rgba(211, 224, 238, 0.18)";
      ctx.lineWidth = 1;
      edges.forEach(([aIndex, bIndex]) => {{
        const a = corners[aIndex];
        const b = corners[bIndex];
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }});
      ctx.restore();
    }}
    function drawGridPlane() {{
      ctx.strokeStyle = "rgba(166, 190, 214, 0.13)";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 6; i++) {{
        const t = -2.8 + i * (5.6 / 6);
        const a = project({{x:t,y:-2.8,z:-2.8}});
        const b = project({{x:t,y:-2.8,z:2.8}});
        const c = project({{x:-2.8,y:-2.8,z:t}});
        const d = project({{x:2.8,y:-2.8,z:t}});
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.moveTo(c.x, c.y);
        ctx.lineTo(d.x, d.y);
        ctx.stroke();
      }}
    }}
    function drawAxisTicks() {{
      ctx.save();
      ctx.font = "11px Inter, system-ui, sans-serif";
      ctx.fillStyle = "rgba(235, 243, 250, 0.60)";
      ctx.strokeStyle = "rgba(235, 243, 250, 0.34)";
      ctx.lineWidth = 1;
      ticks.think.forEach(value => {{
        const x = (norm(value, ranges.thinkLinear) - 0.5) * 5.6;
        drawTick({{x, y:-2.8, z:-2.8}}, {{x, y:-3.0, z:-2.8}}, `${{formatThink(value)}}s`);
      }});
      ticks.simulations.forEach(value => {{
        const y = (norm(Math.log10(value), ranges.sim) - 0.5) * 5.6;
        drawTick({{x:-2.8, y, z:-2.8}}, {{x:-3.02, y, z:-2.8}}, formatCompact(value));
      }});
      ticks.iterations.forEach(value => {{
        const z = (norm(value, ranges.iter) - 0.5) * 5.6;
        drawTick({{x:-2.8, y:-2.8, z}}, {{x:-3.04, y:-2.8, z}}, String(value));
      }});
      ctx.restore();
    }}
    function drawTick(anchor, labelPoint, label) {{
      const a = project(anchor);
      const b = project(labelPoint);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.textAlign = b.x < a.x ? "right" : "left";
      ctx.fillText(label, b.x + (b.x < a.x ? -5 : 5), b.y + 4);
    }}
    function formatCompact(value) {{
      return value >= 1000 ? `${{value / 1000}}k` : String(value);
    }}
    function formatThink(value) {{
      return Number.isInteger(value) ? String(value) : String(value).replace(/0+$/, "").replace(/\\.$/, "");
    }}
    function drawAnnotations() {{
      if (controls.mode !== "combined" || annotationTargets.length === 0) return;
      const points = annotationTargets
        .map(cell => project(cellPoint(cell)))
        .sort((a, b) => a.x - b.x);
      const label = {{
        x: innerWidth * (captureMode ? 0.68 : 0.70),
        y: innerHeight * (captureMode ? 0.20 : 0.23),
      }};
      ctx.save();
      ctx.fillStyle = "rgba(5, 9, 14, 0.58)";
      ctx.strokeStyle = "rgba(86, 255, 151, 0.70)";
      ctx.lineWidth = 1;
      roundRect(label.x - 12, label.y - 30, 234, 58, 8);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "rgba(242, 250, 246, 0.92)";
      ctx.font = "700 13px Inter, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("ITER 542 breaks through", label.x, label.y - 8);
      ctx.font = "12px Inter, system-ui, sans-serif";
      ctx.fillStyle = "rgba(198, 238, 214, 0.86)";
      ctx.fillText("both-role wins at high MCTS budget", label.x, label.y + 12);
      points.forEach((point, index) => {{
        const start = {{
          x: label.x + (index === 0 ? 30 : index === 1 ? 118 : 202),
          y: label.y + 28,
        }};
        drawArrow(start.x, start.y, point.x, point.y, "rgba(86, 255, 151, 0.76)");
        ctx.fillStyle = "rgba(5, 9, 14, 0.64)";
        ctx.strokeStyle = "rgba(86, 255, 151, 0.86)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(point.x, point.y, 10, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }});
      ctx.restore();
    }}
    function drawArrow(x1, y1, x2, y2, color) {{
      const angle = Math.atan2(y2 - y1, x2 - x1);
      const endX = x2 - Math.cos(angle) * 13;
      const endY = y2 - Math.sin(angle) * 13;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(endX, endY);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(
        endX - Math.cos(angle - 0.48) * 9,
        endY - Math.sin(angle - 0.48) * 9
      );
      ctx.lineTo(
        endX - Math.cos(angle + 0.48) * 9,
        endY - Math.sin(angle + 0.48) * 9
      );
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }}
    function roundRect(x, y, width, height, radius) {{
      ctx.beginPath();
      ctx.moveTo(x + radius, y);
      ctx.arcTo(x + width, y, x + width, y + height, radius);
      ctx.arcTo(x + width, y + height, x, y + height, radius);
      ctx.arcTo(x, y + height, x, y, radius);
      ctx.arcTo(x, y, x + width, y, radius);
      ctx.closePath();
    }}
    function drawPoints() {{
      const projected = DATA.cells.map(cell => {{
        const p = project(cellPoint(cell));
        return {{...p, cell, value: outcome(cell)}};
      }}).sort((a, b) => a.z - b.z);
      for (const item of projected) {{
        const radius = (item.value === null ? 3.0 : item.value >= 1 ? 6.4 : item.value >= 0.5 ? 5.2 : 3.8) * item.p;
        ctx.save();
        ctx.globalAlpha = clamp(0.62 + item.p * 0.28, 0.58, 0.94);
        ctx.fillStyle = colorFor(item.value, 0.88);
        ctx.beginPath();
        ctx.arc(item.x, item.y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = "rgba(255,255,255,0.46)";
        ctx.lineWidth = 0.7;
        ctx.stroke();
        ctx.restore();
      }}
    }}
    requestAnimationFrame(draw);
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf8")


def write_small_multiple_movie_html(
    cells: list[dict[str, Any]],
    path: Path,
    *,
    frame_axis: str,
) -> None:
    axes = {
        "iteration": {
            "title": "Movie by model iteration",
            "frameLabel": "Model iteration",
            "x": "simulations",
            "xLabel": "MCTS simulations",
            "y": "site_think_time",
            "yLabel": "Opponent think time",
        },
        "simulations": {
            "title": "Movie by MCTS simulations",
            "frameLabel": "MCTS simulations",
            "x": "iteration",
            "xLabel": "Model iteration",
            "y": "site_think_time",
            "yLabel": "Opponent think time",
        },
        "site_think_time": {
            "title": "Movie by opponent thinking time",
            "frameLabel": "Opponent thinking time",
            "x": "simulations",
            "xLabel": "MCTS simulations",
            "y": "iteration",
            "yLabel": "Model iteration",
        },
    }
    if frame_axis not in axes:
        raise ValueError(f"Unknown movie frame axis: {frame_axis}")

    config = axes[frame_axis]
    values = {
        "iteration": sorted({int(row["iteration"]) for row in cells}),
        "simulations": sorted({int(row["simulations"]) for row in cells}),
        "site_think_time": sorted({float(row["site_think_time"]) for row in cells}),
    }
    movie_cells = [
        {
            "iteration": int(row["iteration"]),
            "simulations": int(row["simulations"]),
            "site_think_time": float(row["site_think_time"]),
            "combined": float(row["combined_win_rate"]),
            "first": row["first_player_win"],
            "second": row["second_player_win"],
            "firstMargin": row["first_player_margin"],
            "secondMargin": row["second_player_margin"],
        }
        for row in cells
    ]
    data_json = json.dumps(
        {
            "cells": movie_cells,
            "values": values,
            "frameAxis": frame_axis,
            "xAxis": config["x"],
            "yAxis": config["y"],
            "title": config["title"],
            "frameLabel": config["frameLabel"],
            "xLabel": config["xLabel"],
            "yLabel": config["yLabel"],
        },
        separators=(",", ":"),
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{config["title"]}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family:
        Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      background: #f8f7f2;
      color: #25272c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: #f8f7f2;
      overflow: hidden;
    }}
    canvas {{
      display: block;
      width: 100vw;
      height: 100vh;
    }}
    .chrome {{
      position: fixed;
      inset: 24px 28px auto 28px;
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 20px;
      pointer-events: none;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(28px, 4vw, 48px);
      line-height: 1.05;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin-top: 8px;
      max-width: 680px;
      color: #696d74;
      font-size: 14px;
      line-height: 1.45;
    }}
    .panel {{
      min-width: 292px;
      border: 1px solid rgba(38, 39, 43, 0.12);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.82);
      box-shadow: 0 18px 60px rgba(34, 32, 26, 0.14);
      padding: 12px;
      pointer-events: auto;
    }}
    .frame {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 13px;
      color: #696d74;
      margin-bottom: 10px;
    }}
    .frame strong {{
      color: #25272c;
      font-size: 16px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 38px 1fr;
      gap: 10px;
      align-items: center;
    }}
    button {{
      width: 38px;
      height: 34px;
      border-radius: 7px;
      border: 1px solid rgba(38, 39, 43, 0.14);
      background: #263238;
      color: white;
      font-size: 15px;
      cursor: pointer;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: #3f995d;
    }}
    .legend {{
      position: fixed;
      left: 32px;
      bottom: 28px;
      display: flex;
      align-items: center;
      gap: 18px;
      font-size: 13px;
      color: #696d74;
      pointer-events: none;
    }}
    .key {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
    }}
    .swatch {{
      width: 18px;
      height: 12px;
      border-radius: 2px;
    }}
    body.capture .panel {{
      display: none;
    }}
    body.capture .chrome {{
      inset: 34px 40px auto 40px;
    }}
  </style>
</head>
<body>
  <canvas id="movie"></canvas>
  <div class="chrome">
    <div>
      <h1>{config["title"]}</h1>
      <div class="subtitle">
        Each frame fixes {config["frameLabel"].lower()}; the grid shows
        {config["yLabel"].lower()} by {config["xLabel"].lower()} with combined
        role win rate in each cell.
      </div>
    </div>
    <div class="panel">
      <div class="frame">
        <span id="frameName">{config["frameLabel"]}</span>
        <strong id="frameValue"></strong>
      </div>
      <div class="controls">
        <button id="play" aria-label="Play or pause">II</button>
        <input id="scrub" type="range" min="0" max="0" step="1" value="0">
      </div>
    </div>
  </div>
  <div class="legend">
    <span class="key"><span class="swatch" style="background:#c94f4f"></span>0 roles</span>
    <span class="key"><span class="swatch" style="background:#d99b3d"></span>1 role</span>
    <span class="key"><span class="swatch" style="background:#3f995d"></span>2 roles</span>
  </div>
  <script>
    const DATA = {data_json};
    const params = new URLSearchParams(location.search);
    if (params.has("capture")) document.body.classList.add("capture");

    const canvas = document.getElementById("movie");
    const ctx = canvas.getContext("2d");
    const play = document.getElementById("play");
    const scrub = document.getElementById("scrub");
    const frameValue = document.getElementById("frameValue");
    const frameValues = DATA.values[DATA.frameAxis];
    const xValues = DATA.values[DATA.xAxis];
    const yValues = DATA.values[DATA.yAxis].slice();
    if (DATA.yAxis === "iteration") yValues.reverse();

    let frame = Number(params.get("frame") || 0);
    let playing = !params.has("paused");
    let lastStep = 0;
    const frameMs = Number(params.get("frameMs") || 900);

    scrub.max = String(Math.max(0, frameValues.length - 1));
    play.addEventListener("click", () => {{
      playing = !playing;
      play.textContent = playing ? "II" : ">";
    }});
    scrub.addEventListener("input", event => {{
      frame = Number(event.target.value);
      draw();
    }});

    function resize() {{
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(innerWidth * dpr);
      canvas.height = Math.floor(innerHeight * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }}
    addEventListener("resize", resize);

    function equalAxis(a, b) {{
      return Number(a) === Number(b);
    }}
    function label(axis, value) {{
      if (axis === "iteration") return `ITER=${{String(value).padStart(3, "0")}}`;
      if (axis === "simulations") return String(value);
      return `${{Number(value).toLocaleString(undefined, {{ maximumFractionDigits: 2 }})}}s`;
    }}
    function cellFor(x, y, frameValue) {{
      return DATA.cells.find(cell =>
        equalAxis(cell[DATA.frameAxis], frameValue) &&
        equalAxis(cell[DATA.xAxis], x) &&
        equalAxis(cell[DATA.yAxis], y)
      );
    }}
    function heatColor(value) {{
      if (value >= 1) return "#3f995d";
      if (value >= 0.5) return "#d99b3d";
      return "#c94f4f";
    }}
    function draw(timestamp = 0) {{
      if (playing && timestamp - lastStep > frameMs) {{
        frame = (frame + 1) % frameValues.length;
        lastStep = timestamp;
      }}
      scrub.value = String(frame);
      play.textContent = playing ? "II" : ">";
      const current = frameValues[frame];
      frameValue.textContent = label(DATA.frameAxis, current);

      ctx.clearRect(0, 0, innerWidth, innerHeight);
      ctx.fillStyle = "#f8f7f2";
      ctx.fillRect(0, 0, innerWidth, innerHeight);
      drawGrid(current);
      if (timestamp !== 0) requestAnimationFrame(draw);
    }}
    function drawGrid(current) {{
      const left = Math.max(84, innerWidth * 0.08);
      const top = Math.max(140, innerHeight * 0.20);
      const right = Math.min(innerWidth - 54, innerWidth * 0.94);
      const bottom = Math.min(innerHeight - 84, innerHeight * 0.86);
      const labelW = 92;
      const labelH = 34;
      const gridX = left + labelW;
      const gridY = top;
      const gridW = right - gridX;
      const gridH = bottom - gridY - labelH;
      const cellW = gridW / xValues.length;
      const cellH = gridH / yValues.length;

      ctx.fillStyle = "#686c73";
      ctx.font = "600 13px Inter, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(DATA.yLabel, left, top - 18);
      ctx.textAlign = "center";
      ctx.fillText(DATA.xLabel, gridX + gridW / 2, bottom + 34);

      yValues.forEach((y, row) => {{
        const y0 = gridY + row * cellH;
        ctx.fillStyle = "#686c73";
        ctx.textAlign = "right";
        ctx.font = "12px Inter, system-ui, sans-serif";
        ctx.fillText(label(DATA.yAxis, y), gridX - 14, y0 + cellH * 0.58);
        xValues.forEach((x, col) => {{
          const x0 = gridX + col * cellW;
          const cell = cellFor(x, y, current);
          ctx.fillStyle = cell ? heatColor(cell.combined) : "#dad8d0";
          ctx.fillRect(x0 + 2, y0 + 2, Math.max(1, cellW - 4), Math.max(1, cellH - 4));
          if (cellW >= 34 && cellH >= 24 && cell) {{
            ctx.fillStyle = "rgba(255,255,255,0.94)";
            ctx.font = "700 12px Inter, system-ui, sans-serif";
            ctx.textAlign = "center";
            const labelText = cell.combined === 0 ? "0" : cell.combined === 0.5 ? "1/2" : "1";
            ctx.fillText(labelText, x0 + cellW / 2, y0 + cellH / 2 + 4);
          }}
        }});
      }});
      xValues.forEach((x, col) => {{
        const x0 = gridX + col * cellW;
        ctx.save();
        ctx.translate(x0 + cellW / 2, bottom);
        ctx.rotate(-Math.PI / 5.5);
        ctx.fillStyle = "#686c73";
        ctx.textAlign = "right";
        ctx.font = "11px Inter, system-ui, sans-serif";
        ctx.fillText(label(DATA.xAxis, x), 0, 0);
        ctx.restore();
      }});

      ctx.strokeStyle = "rgba(38,39,43,0.20)";
      ctx.lineWidth = 1;
      ctx.strokeRect(gridX, gridY, gridW, gridH);
    }}
    resize();
    requestAnimationFrame(draw);
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf8")


def write_small_multiple_movies(cells: list[dict[str, Any]], out_dir: Path) -> None:
    write_small_multiple_movie_html(
        cells,
        out_dir / "08_movie_by_model_iteration.html",
        frame_axis="iteration",
    )
    write_small_multiple_movie_html(
        cells,
        out_dir / "09_movie_by_mcts_simulations.html",
        frame_axis="simulations",
    )
    write_small_multiple_movie_html(
        cells,
        out_dir / "10_movie_by_opponent_think_time.html",
        frame_axis="site_think_time",
    )


def draw_binary_heatmap_panel(
    *,
    parts: list[str],
    origin_x: int,
    origin_y: int,
    title: str,
    simulations: list[int],
    think_times: list[float],
    value_for,
) -> None:
    parts.append(svg_text(origin_x, origin_y - 22, title, 15, "#26272b", bold=True))
    cell_w = 42
    cell_h = 30
    label_w = 48
    for row_index, think in enumerate(think_times):
        y = origin_y + row_index * cell_h
        parts.append(svg_text(origin_x, y + 20, f"{think:g}s", 11, "#686c73"))
        for col_index, sims in enumerate(simulations):
            x = origin_x + label_w + col_index * cell_w
            value = value_for(sims, think)
            if value == "":
                parts.append(rect(x, y, cell_w - 2, cell_h - 2, "#ece8df"))
                parts.append(svg_text(x + 15, y + 19, "-", 11, "#9a948b"))
                continue
            parts.append(rect(x, y, cell_w - 2, cell_h - 2, role_color(value)))
            parts.append(svg_text(x + 15, y + 19, str(value), 11, "white"))
    for col_index, sims in enumerate(simulations):
        x = origin_x + label_w + col_index * cell_w
        parts.append(
            svg_text(
                x,
                origin_y + len(think_times) * cell_h + 16,
                str(sims),
                9,
                "#686c73",
            )
        )


@dataclass(frozen=True)
class PlotArea:
    x: int
    y: int
    width: int
    height: int


def select_curve_iterations(iterations: list[int]) -> list[int]:
    if len(iterations) <= 6:
        return iterations
    indexes = [
        0,
        len(iterations) // 5,
        2 * len(iterations) // 5,
        3 * len(iterations) // 5,
        4 * len(iterations) // 5,
        len(iterations) - 1,
    ]
    return [iterations[index] for index in sorted(set(indexes))]


def draw_log_sim_axes(parts: list[str], plot: PlotArea, think_times: list[float]) -> None:
    parts.append(line(plot.x, plot.y + plot.height, plot.x + plot.width, plot.y + plot.height))
    parts.append(line(plot.x, plot.y, plot.x, plot.y + plot.height))
    for think in think_times:
        x = x_for_think(plot, think_times, think)
        parts.append(line(x, plot.y, x, plot.y + plot.height, "#e1e2e3"))
        parts.append(svg_text(x - 12, plot.y + plot.height + 24, f"{think:g}s", 11, "#686c73"))
    for sim in [250, 500, 1000, 2000, 5000, 10000]:
        y = y_for_sim(plot, sim)
        parts.append(line(plot.x, y, plot.x + plot.width, y, "#e1e2e3"))
        parts.append(svg_text(plot.x - 54, y + 4, str(sim), 11, "#686c73"))


def x_for_think(plot: PlotArea, think_times: list[float], think: float) -> float:
    if len(think_times) == 1:
        return plot.x + plot.width / 2
    return plot.x + think_times.index(think) * plot.width / (len(think_times) - 1)


def y_for_sim(plot: PlotArea, simulations: int) -> float:
    min_log = math.log10(250)
    max_log = math.log10(10000)
    value = (math.log10(simulations) - min_log) / (max_log - min_log)
    return plot.y + plot.height - value * plot.height


def draw_polyline(parts: list[str], points: list[tuple[float, float]], color: str) -> None:
    joined = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    parts.append(
        f'<polyline points="{joined}" fill="none" stroke="{color}" '
        'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    for x, y in points:
        parts.append(circle(x, y, 4, color, "white"))


def svg_header(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        svg_text(30, 38, title, 24, "#26272b", bold=True),
        svg_text(30, 64, subtitle, 14, "#686c73"),
    ]


def svg_text(x: float, y: float, text: str, size: int, color: str, bold: bool = False) -> str:
    weight = ' font-weight="700"' if bold else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial" font-size="{size}"'
        f'{weight} fill="{color}">{text}</text>'
    )


def rect(x: float, y: float, width: float, height: float, fill: str) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{fill}" rx="2"/>'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "#26272b",
    width: int = 1,
) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}"/>'
    )


def circle(x: float, y: float, radius: float, fill: str, stroke: str) -> str:
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
    )


def palette(count: int) -> list[str]:
    colors = ["#7357a6", "#2f80a8", "#32936f", "#d98f34", "#c65353", "#5e6f8f"]
    return colors[:count]


def legend(parts: list[str], x: int, y: int) -> None:
    labels = [("0 roles", "#c94f4f"), ("1 role", "#d99b3d"), ("2 roles", "#3f995d")]
    for index, (label, color) in enumerate(labels):
        item_x = x + index * 92
        parts.append(rect(item_x, y, 16, 16, color))
        parts.append(svg_text(item_x + 24, y + 13, label, 12, "#686c73"))


def write_svg(path: Path, parts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf8")


def write_report(*, games_path: Path, out_dir: Path) -> None:
    records = read_jsonl(games_path)
    rows = [game_row(record) for record in records]
    cells = cell_rows(rows)
    frontier = frontier_rows(cells)
    write_game_csv(rows, out_dir / "games.csv")
    write_dict_csv(cells, out_dir / "cells.csv")
    write_dict_csv(frontier, out_dir / "frontier.csv")
    write_latest_checkpoint_heatmap(cells, out_dir / "01_latest_checkpoint_heatmap.svg")
    write_combined_heatmaps(cells, out_dir / "combined_heatmaps.svg")
    write_combined_heatmaps(cells, out_dir / "02_small_multiples_combined_heatmaps.svg")
    write_threshold_curves(frontier, out_dir / "03_threshold_curves.svg")
    write_role_separated_heatmaps(cells, out_dir / "04_role_separated_heatmaps.svg")
    write_phase_diagram(cells, out_dir / "05_3d_phase_diagram.svg")
    write_capability_frontier(frontier, out_dir / "06_capability_frontier.svg")
    write_turntable_html(cells, frontier, out_dir / "07_phase_turntable.html")
    write_small_multiple_movies(cells, out_dir)
    print(f"Wrote {len(rows)} game rows to {out_dir / 'games.csv'}")
    print(f"Wrote {len(cells)} cell rows to {out_dir / 'cells.csv'}")
    print(f"Wrote {len(frontier)} frontier rows to {out_dir / 'frontier.csv'}")
    print(f"Wrote plots to {out_dir}")


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
