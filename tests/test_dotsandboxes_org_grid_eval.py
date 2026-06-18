from __future__ import annotations

import json
from pathlib import Path

from dots_boxes_mcts.dotsandboxes_org_grid_eval import (
    add_grid_metadata,
    build_grid,
    completed_cell_keys,
    pending_cells,
    read_jsonl,
    seed_for_cell,
    write_summary,
)


def test_build_grid_orders_iterations_descending_and_pairs_players() -> None:
    cells = build_grid(
        iterations=(1, 3),
        simulations_values=(250, 500),
        site_think_times=(0.1,),
    )

    assert [
        (cell.iteration, cell.simulations, cell.our_player, cell.seed)
        for cell in cells
    ] == [
        (3, 250, 0, 100300),
        (3, 250, 1, 100301),
        (3, 500, 0, 100310),
        (3, 500, 1, 100311),
        (1, 250, 0, 100100),
        (1, 250, 1, 100101),
        (1, 500, 0, 100110),
        (1, 500, 1, 100111),
    ]


def test_seed_for_cell_is_stable_and_leaves_room_for_player_offset() -> None:
    assert seed_for_cell(iteration=542, simulations_index=4, site_think_time_index=2) == 154242


def test_add_grid_metadata_adds_resume_key() -> None:
    cell = build_grid(
        iterations=(542,),
        simulations_values=(5000,),
        site_think_times=(0.25,),
    )[0]

    record = add_grid_metadata({"terminal": True, "winner": 0}, cell)

    assert record["gridCellKey"] == cell.key
    assert record["grid"] == {
        "iteration": 542,
        "checkpoint": "runs/ez-flywheel/ez-policy-value-4x4-iter542-sims2000.npz",
        "simulations": 5000,
        "simulationsIndex": 0,
        "siteThinkTime": 0.25,
        "siteThinkTimeIndex": 0,
        "ourPlayer": 0,
        "seed": 154200,
    }


def test_pending_cells_skips_terminal_records_with_matching_grid_key() -> None:
    cells = build_grid(
        iterations=(2,),
        simulations_values=(250,),
        site_think_times=(0.1,),
    )
    records = [add_grid_metadata({"terminal": True}, cells[0])]

    assert completed_cell_keys(records) == {cells[0].key}
    assert pending_cells(cells, records) == [cells[1]]


def test_write_summary_includes_pending_and_aggregate_summary(tmp_path: Path) -> None:
    cells = build_grid(
        iterations=(2,),
        simulations_values=(250,),
        site_think_times=(0.1,),
    )
    record = add_grid_metadata(
        {
            "terminal": True,
            "winner": 0,
            "ourPlayer": 0,
            "finalScores": [5, 4],
            "moves": [],
            "rows": 4,
            "cols": 4,
        },
        cells[0],
    )
    summary_path = tmp_path / "summary.json"

    write_summary(path=summary_path, cells=cells, records=[record], failures=[])

    summary = json.loads(summary_path.read_text(encoding="utf8"))
    assert summary["totalCells"] == 2
    assert summary["completedCells"] == 1
    assert summary["pendingCells"] == 1
    assert summary["summary"]["wins"] == 1
    assert summary["pending"][0]["key"] == cells[1].key


def test_read_jsonl_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert read_jsonl(tmp_path / "missing.jsonl") == []
