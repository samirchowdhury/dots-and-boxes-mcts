from __future__ import annotations

from dots_boxes_mcts.dotsandboxes_org_grid_report import cell_rows, frontier_rows, game_row


def record(
    *,
    iteration: int,
    simulations: int,
    think: float,
    our_player: int,
    winner,
    final_scores: list[int],
) -> dict:
    return {
        "grid": {
            "iteration": iteration,
            "checkpoint": f"runs/ez-flywheel/ez-policy-value-4x4-iter{iteration:03d}-sims2000.npz",
            "simulations": simulations,
            "siteThinkTime": think,
            "ourPlayer": our_player,
            "seed": 123 + our_player,
        },
        "ourPlayer": our_player,
        "winner": winner,
        "finalScores": final_scores,
        "moves": ["h:0:0", "v:0:0"],
        "decisions": [{"move": "h:0:0"}],
        "gridCellKey": f"{iteration}-{simulations}-{think}-{our_player}",
    }


def test_game_row_flattens_result_from_our_perspective() -> None:
    row = game_row(
        record(
            iteration=542,
            simulations=5000,
            think=0.25,
            our_player=1,
            winner=1,
            final_scores=[3, 6],
        )
    )

    assert row.iteration == 542
    assert row.our_player == 1
    assert row.win == 1
    assert row.loss == 0
    assert row.score_margin == 3
    assert row.move_count == 2
    assert row.decision_count == 1


def test_cell_rows_combines_first_and_second_player_games() -> None:
    rows = [
        game_row(
            record(
                iteration=542,
                simulations=5000,
                think=0.25,
                our_player=0,
                winner=1,
                final_scores=[4, 5],
            )
        ),
        game_row(
            record(
                iteration=542,
                simulations=5000,
                think=0.25,
                our_player=1,
                winner=1,
                final_scores=[3, 6],
            )
        ),
    ]

    cells = cell_rows(rows)

    assert cells == [
        {
            "iteration": 542,
            "simulations": 5000,
            "site_think_time": 0.25,
            "completed_roles": 2,
            "first_player_win": 0,
            "second_player_win": 1,
            "combined_win_rate": 0.5,
            "first_player_margin": -1,
            "second_player_margin": 3,
        }
    ]


def test_frontier_rows_finds_minimum_simulations_for_each_role() -> None:
    cells = [
        {
            "iteration": 542,
            "simulations": 250,
            "site_think_time": 0.25,
            "first_player_win": 0,
            "second_player_win": 1,
            "combined_win_rate": 0.5,
        },
        {
            "iteration": 542,
            "simulations": 5000,
            "site_think_time": 0.25,
            "first_player_win": 1,
            "second_player_win": 1,
            "combined_win_rate": 1.0,
        },
    ]

    assert frontier_rows(cells) == [
        {
            "iteration": 542,
            "site_think_time": 0.25,
            "first_player_min_win_simulations": 5000,
            "second_player_min_win_simulations": 250,
            "both_roles_min_win_simulations": 5000,
            "any_role_min_win_simulations": 250,
        }
    ]
