from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dots_boxes_mcts.dotsandboxes_org_browser_eval import (
    DEFAULT_CHROME_PATH,
    DOTSANDBOXES_ORG_URL,
    RecordingOptions,
    block_nonessential_site_traffic,
    play_dotsandboxes_org_game,
    validate_dotsandboxes_org_board_size,
)
from dots_boxes_mcts.external_bot_common import summarize_external_records

DEFAULT_ITERS = (10, 20, 50, 68, 92, 110, 152, 194, 250, 300, 350, 388, 450, 500, 542)
DEFAULT_SIMULATIONS = (250, 500, 1000, 2000, 5000, 10000)
DEFAULT_SITE_THINK_TIMES = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
DEFAULT_OUT_DIR = Path("runs/dotsandboxes-org/ez-flywheel-grid")


@dataclass(frozen=True)
class GridCell:
    iteration: int
    checkpoint: Path
    simulations: int
    simulations_index: int
    site_think_time: float
    site_think_time_index: int
    our_player: int
    seed: int

    @property
    def key(self) -> str:
        return "|".join(
            [
                f"iter={self.iteration:03d}",
                f"checkpoint={self.checkpoint.as_posix()}",
                f"simulations={self.simulations}",
                f"siteThinkTime={format_float(self.site_think_time)}",
                f"ourPlayer={self.our_player}",
                f"seed={self.seed}",
            ]
        )


def format_float(value: float) -> str:
    return f"{value:g}"


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return items


def parse_float_list(value: str) -> tuple[float, ...]:
    items = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    return items


def checkpoint_for_iteration(iteration: int) -> Path:
    return Path(f"runs/ez-flywheel/ez-policy-value-4x4-iter{iteration:03d}-sims2000.npz")


def seed_for_cell(iteration: int, simulations_index: int, site_think_time_index: int) -> int:
    return 100_000 + iteration * 100 + simulations_index * 10 + site_think_time_index


def build_grid(
    *,
    iterations: tuple[int, ...],
    simulations_values: tuple[int, ...],
    site_think_times: tuple[float, ...],
) -> list[GridCell]:
    cells: list[GridCell] = []
    for iteration in sorted(iterations, reverse=True):
        checkpoint = checkpoint_for_iteration(iteration)
        for simulations_index, simulations in enumerate(simulations_values):
            for site_think_time_index, site_think_time in enumerate(site_think_times):
                base_seed = seed_for_cell(
                    iteration=iteration,
                    simulations_index=simulations_index,
                    site_think_time_index=site_think_time_index,
                )
                for our_player in (0, 1):
                    cells.append(
                        GridCell(
                            iteration=iteration,
                            checkpoint=checkpoint,
                            simulations=simulations,
                            simulations_index=simulations_index,
                            site_think_time=site_think_time,
                            site_think_time_index=site_think_time_index,
                            our_player=our_player,
                            seed=base_seed + our_player,
                        )
                    )
    return cells


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open(encoding="utf8") as input_file:
        for line in input_file:
            if line.strip():
                records.append(json.loads(line))
    return records


def append_jsonl(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as output:
        output.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
        output.write("\n")


def completed_cell_keys(records: list[dict]) -> set[str]:
    return {
        record["gridCellKey"]
        for record in records
        if record.get("gridCellKey") and record.get("terminal")
    }


def pending_cells(cells: list[GridCell], records: list[dict]) -> list[GridCell]:
    completed = completed_cell_keys(records)
    return [cell for cell in cells if cell.key not in completed]


def add_grid_metadata(record: dict, cell: GridCell) -> dict:
    enriched = dict(record)
    enriched["gridCellKey"] = cell.key
    enriched["grid"] = {
        "iteration": cell.iteration,
        "checkpoint": cell.checkpoint.as_posix(),
        "simulations": cell.simulations,
        "simulationsIndex": cell.simulations_index,
        "siteThinkTime": cell.site_think_time,
        "siteThinkTimeIndex": cell.site_think_time_index,
        "ourPlayer": cell.our_player,
        "seed": cell.seed,
    }
    return enriched


def write_summary(
    *,
    path: Path,
    cells: list[GridCell],
    records: list[dict],
    failures: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_cell_keys(records)
    pending = [cell for cell in cells if cell.key not in completed]
    terminal_records = [record for record in records if record.get("terminal")]
    payload = {
        "totalCells": len(cells),
        "completedCells": len(completed),
        "pendingCells": len(pending),
        "failures": len(failures),
        "summary": summarize_external_records(terminal_records) if terminal_records else None,
        "pending": [asdict_for_json(cell) for cell in pending],
        "failedKeys": sorted({failure["gridCellKey"] for failure in failures}),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf8")


def asdict_for_json(cell: GridCell) -> dict:
    payload = asdict(cell)
    payload["checkpoint"] = cell.checkpoint.as_posix()
    payload["key"] = cell.key
    return payload


def validate_checkpoints(cells: list[GridCell]) -> None:
    missing = sorted({cell.checkpoint for cell in cells if not cell.checkpoint.exists()})
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise SystemExit(f"Missing checkpoint file(s):\n{joined}")


def run_grid(
    *,
    cells: list[GridCell],
    out_dir: Path,
    rows: int,
    cols: int,
    c_puct: float,
    mlx_device: str,
    backend: str,
    mcts_backend: str,
    mcts_batch_size: int,
    virtual_loss: float,
    headless: bool,
    browser_executable: Path | None,
    slow_mo_ms: int,
    site_url: str,
    block_telemetry: bool,
    opening_top_k: int,
    max_retries: int,
    recording: RecordingOptions | None,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise RuntimeError(
            "The dotsandboxes.org grid runner needs Playwright. Install it with: `uv sync`."
        ) from error

    games_path = out_dir / "games.jsonl"
    failures_path = out_dir / "failures.jsonl"
    summary_path = out_dir / "summary.json"
    records = read_jsonl(games_path)
    failures = read_jsonl(failures_path)
    remaining = pending_cells(cells, records)
    write_summary(path=summary_path, cells=cells, records=records, failures=failures)
    if not remaining:
        print(f"All {len(cells)} grid games are already complete in {games_path}.")
        return

    with sync_playwright() as playwright:
        launch_kwargs: dict[str, Any] = {
            "headless": headless,
            "slow_mo": slow_mo_ms,
        }
        if browser_executable is not None:
            launch_kwargs["executable_path"] = str(browser_executable)
        browser = playwright.chromium.launch(**launch_kwargs)
        context = browser.new_context()
        if block_telemetry:
            block_nonessential_site_traffic(context)
        page = context.new_page()
        try:
            for index, cell in enumerate(remaining, start=1):
                label = (
                    f"ITER={cell.iteration:03d} sims={cell.simulations} "
                    f"think={format_float(cell.site_think_time)} player={cell.our_player}"
                )
                print(f"[{index}/{len(remaining)}] {label}")
                for attempt in range(max_retries + 1):
                    try:
                        record = play_dotsandboxes_org_game(
                            page=page,
                            checkpoint=cell.checkpoint,
                            rows=rows,
                            cols=cols,
                            simulations=cell.simulations,
                            seed=cell.seed,
                            c_puct=c_puct,
                            mlx_device=mlx_device,
                            backend=backend,
                            mcts_backend=mcts_backend,
                            mcts_batch_size=mcts_batch_size,
                            virtual_loss=virtual_loss,
                            our_player=cell.our_player,
                            site_url=site_url,
                            site_think_time=cell.site_think_time,
                            block_telemetry=block_telemetry,
                            opening_top_k=opening_top_k,
                            opening_index=0,
                            recording=recording,
                        )
                    except Exception as error:
                        failure = {
                            "gridCellKey": cell.key,
                            "grid": asdict_for_json(cell),
                            "attempt": attempt + 1,
                            "error": repr(error),
                        }
                        append_jsonl(failure, failures_path)
                        failures.append(failure)
                        write_summary(
                            path=summary_path,
                            cells=cells,
                            records=records,
                            failures=failures,
                        )
                        try:
                            page.reload(wait_until="domcontentloaded")
                        except Exception:
                            page.goto(site_url, wait_until="domcontentloaded")
                        if attempt >= max_retries:
                            print(f"  failed after {max_retries + 1} attempts: {error!r}")
                            break
                        print(f"  retrying after error: {error!r}")
                        continue

                    enriched = add_grid_metadata(record, cell)
                    append_jsonl(enriched, games_path)
                    records.append(enriched)
                    write_summary(
                        path=summary_path,
                        cells=cells,
                        records=records,
                        failures=failures,
                    )
                    break
        finally:
            context.close()
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a resumable broad-grid dotsandboxes.org EpsilonZero evaluation."
    )
    parser.add_argument("--iters", type=parse_int_list, default=DEFAULT_ITERS)
    parser.add_argument("--simulations", type=parse_int_list, default=DEFAULT_SIMULATIONS)
    parser.add_argument(
        "--site-think-times",
        type=parse_float_list,
        default=DEFAULT_SITE_THINK_TIMES,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--mlx-device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--backend", choices=["python", "numba"], default="numba")
    parser.add_argument("--mcts-backend", choices=["python", "cpp"], default="cpp")
    parser.add_argument("--mcts-batch-size", type=int, default=8)
    parser.add_argument("--virtual-loss", type=float, default=1.0)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--browser-executable", type=Path, default=DEFAULT_CHROME_PATH)
    parser.add_argument("--slow-mo-ms", type=int, default=0)
    parser.add_argument("--site-url", default=DOTSANDBOXES_ORG_URL)
    parser.add_argument("--allow-site-telemetry", action="store_true")
    parser.add_argument("--opening-top-k", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    if args.max_retries < 0:
        raise SystemExit("--max-retries must be non-negative")
    if args.opening_top_k < 1:
        raise SystemExit("--opening-top-k must be at least 1")
    if args.mcts_batch_size < 1:
        raise SystemExit("--mcts-batch-size must be at least 1")
    if args.virtual_loss < 0:
        raise SystemExit("--virtual-loss must be non-negative")
    if any(value < 1 for value in args.simulations):
        raise SystemExit("--simulations values must be positive")
    if any(value < 0 for value in args.site_think_times):
        raise SystemExit("--site-think-times values must be non-negative")
    if args.browser_executable is not None and not args.browser_executable.exists():
        raise SystemExit(f"Browser executable does not exist: {args.browser_executable}")
    validate_dotsandboxes_org_board_size(rows=args.rows, cols=args.cols)

    cells = build_grid(
        iterations=args.iters,
        simulations_values=args.simulations,
        site_think_times=args.site_think_times,
    )
    validate_checkpoints(cells)
    run_grid(
        cells=cells,
        out_dir=args.out_dir,
        rows=args.rows,
        cols=args.cols,
        c_puct=args.c_puct,
        mlx_device=args.mlx_device,
        backend=args.backend,
        mcts_backend=args.mcts_backend,
        mcts_batch_size=args.mcts_batch_size,
        virtual_loss=args.virtual_loss,
        headless=not args.headed,
        browser_executable=args.browser_executable,
        slow_mo_ms=args.slow_mo_ms,
        site_url=args.site_url,
        block_telemetry=not args.allow_site_telemetry,
        opening_top_k=args.opening_top_k,
        max_retries=args.max_retries,
        recording=None,
    )


if __name__ == "__main__":
    main()
