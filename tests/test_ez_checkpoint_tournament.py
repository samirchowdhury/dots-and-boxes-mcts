from __future__ import annotations

from pathlib import Path

from dots_boxes_mcts.ez_checkpoint_tournament import (
    CheckpointEntry,
    TournamentGame,
    add_tournament_metadata,
    build_schedule,
    forgetting_diagnosis,
    fit_bradley_terry_elos,
    main,
    parse_checkpoint_iteration,
    pending_games,
    sample_checkpoints,
    standings_rows,
)


def entry(iteration: int) -> CheckpointEntry:
    return CheckpointEntry(
        iteration=iteration,
        path=Path(f"runs/ez-flywheel/ez-policy-value-4x4-iter{iteration:03d}-sims2000.npz"),
    )


def raw_record(
    *,
    candidate: int,
    baseline: int,
    candidate_player: int,
    winner,
    final_scores: list[int] | None = None,
) -> dict:
    return {
        "candidateCheckpoint": entry(candidate).path.as_posix(),
        "baselineCheckpoint": entry(baseline).path.as_posix(),
        "candidatePlayer": candidate_player,
        "finalScores": final_scores or [5, 4],
        "winner": winner,
        "terminal": True,
    }


def tournament_record(
    *,
    candidate: int,
    baseline: int,
    candidate_player: int,
    winner,
    final_scores: list[int] | None = None,
    game_in_pair: int = 0,
) -> dict:
    game = TournamentGame(
        pair_index=0,
        game_in_pair=game_in_pair,
        checkpoint_a=entry(candidate),
        checkpoint_b=entry(baseline),
        candidate_player=candidate_player,
        seed=123,
    )
    return add_tournament_metadata(
        raw_record(
            candidate=candidate,
            baseline=baseline,
            candidate_player=candidate_player,
            winner=winner,
            final_scores=final_scores,
        ),
        game,
    )


def test_parse_checkpoint_iteration() -> None:
    assert parse_checkpoint_iteration(Path("ez-policy-value-4x4-iter542-sims2000.npz")) == 542


def test_sample_checkpoints_includes_earliest_latest_and_anchor() -> None:
    sampled = sample_checkpoints(
        [entry(iteration) for iteration in range(1, 701)],
        sample_size=60,
        include_iters=(542,),
    )

    iterations = [checkpoint.iteration for checkpoint in sampled]
    assert len(sampled) == 60
    assert iterations[0] == 1
    assert iterations[-1] == 700
    assert 542 in iterations


def test_build_schedule_all_pairs_with_two_swapped_seat_games() -> None:
    schedule = build_schedule(
        [entry(iteration) for iteration in range(1, 61)],
        games_per_pair=2,
        seed=900_001,
    )

    assert len(schedule) == 60 * 59 // 2 * 2
    assert schedule[0].checkpoint_a.iteration == 1
    assert schedule[0].checkpoint_b.iteration == 2
    assert schedule[0].candidate_player == 0
    assert schedule[1].candidate_player == 1


def test_pending_games_skips_completed_terminal_records() -> None:
    schedule = build_schedule([entry(1), entry(2)], games_per_pair=2, seed=1)
    completed = add_tournament_metadata(
        raw_record(candidate=1, baseline=2, candidate_player=0, winner=0),
        schedule[0],
    )

    assert pending_games(schedule, [completed]) == [schedule[1]]


def test_standings_assign_results_to_checkpoint_identity_across_seats() -> None:
    records = [
        tournament_record(
            candidate=10,
            baseline=20,
            candidate_player=0,
            winner=0,
            final_scores=[6, 3],
        ),
        tournament_record(
            candidate=10,
            baseline=20,
            candidate_player=1,
            winner=1,
            final_scores=[3, 6],
            game_in_pair=1,
        ),
    ]

    rows = {int(row["iteration"]): row for row in standings_rows(records, [entry(10), entry(20)])}

    assert rows[10]["wins"] == 2
    assert rows[10]["losses"] == 0
    assert rows[10]["averageScoreMargin"] == 3
    assert rows[20]["losses"] == 2


def test_rating_ranks_synthetic_dominant_checkpoint_highest() -> None:
    records = [
        tournament_record(candidate=3, baseline=1, candidate_player=0, winner=0),
        tournament_record(candidate=3, baseline=2, candidate_player=0, winner=0),
        tournament_record(candidate=2, baseline=1, candidate_player=0, winner=0),
    ]

    ratings = fit_bradley_terry_elos(records, [1, 2, 3])

    assert ratings[3] > ratings[2] > ratings[1]


def test_forgetting_diagnosis_detects_regressed_latest() -> None:
    standings = [
        {"iteration": 542, "rating": 1600.0},
        {"iteration": 600, "rating": 1700.0},
        {"iteration": 700, "rating": 1500.0},
    ]
    records = [
        tournament_record(
            candidate=542,
            baseline=700,
            candidate_player=0,
            winner=0,
            final_scores=[6, 3],
        ),
        tournament_record(
            candidate=542,
            baseline=700,
            candidate_player=1,
            winner=1,
            final_scores=[3, 6],
            game_in_pair=1,
        ),
    ]

    diagnosis = forgetting_diagnosis(
        standings=standings,
        records=records,
        anchor_iteration=542,
    )

    assert diagnosis["forgettingFlag"] is True
    assert diagnosis["status"] == "likely_forgetting"


def test_forgetting_diagnosis_marks_improving_latest_as_no_clear_forgetting() -> None:
    diagnosis = forgetting_diagnosis(
        standings=[
            {"iteration": 542, "rating": 1500.0},
            {"iteration": 600, "rating": 1550.0},
            {"iteration": 700, "rating": 1600.0},
        ],
        records=[],
        anchor_iteration=542,
    )

    assert diagnosis["forgettingFlag"] is False
    assert diagnosis["status"] == "no_clear_forgetting"


def test_cli_defaults_to_cpp_backend(monkeypatch, tmp_path: Path) -> None:
    observed = {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "ez_checkpoint_tournament",
            "--checkpoint-dir",
            str(tmp_path),
            "--sample-size",
            "2",
        ],
    )
    monkeypatch.setattr(
        "dots_boxes_mcts.ez_checkpoint_tournament.discover_checkpoints",
        lambda checkpoint_dir, checkpoint_pattern: [entry(1), entry(2)],
    )

    def fake_run_tournament(**kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(
        "dots_boxes_mcts.ez_checkpoint_tournament.run_tournament",
        fake_run_tournament,
    )

    main()

    assert observed["mcts_backend"] == "cpp"
