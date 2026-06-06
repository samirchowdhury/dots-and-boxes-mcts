from pathlib import Path

from dots_boxes_mcts.az_guided_self_play import (
    default_output_path,
    ensure_outputs_do_not_exist,
    format_game_progress,
    format_turn_progress,
    metadata_output_path,
    run_metadata,
)


def test_format_turn_progress_includes_search_timing_and_game_position() -> None:
    message = format_turn_progress(
        game_index=2,
        total_games=10,
        seed=6003,
        turn=7,
        player=1,
        move="v:1:2",
        search_seconds=0.1234,
        simulations=100,
        legal_moves=17,
        scores_before=(1, 0),
        scores_after=(1, 1),
    )

    assert "game=3/10" in message
    assert "seed=6003" in message
    assert "turn=7" in message
    assert "search=0.123s" in message
    assert "simulations=100" in message
    assert "scores=1-1" in message
    assert "scored=true" in message


def test_format_game_progress_includes_final_result() -> None:
    message = format_game_progress(
        game_index=0,
        total_games=5,
        seed=6001,
        moves=24,
        elapsed_seconds=9.876,
        final_scores=(5, 4),
        winner=0,
    )

    assert "game=1/5" in message
    assert "complete" in message
    assert "moves=24" in message
    assert "elapsed=9.876s" in message
    assert "final=5-4" in message
    assert "winner=0" in message


def test_default_output_path_includes_core_self_play_parameters() -> None:
    path = default_output_path(
        rows=4,
        cols=4,
        games=100,
        simulations=250,
    )

    assert path == Path("runs/stage-3.6/guided-self-play-4x4-games100-sims250.jsonl")


def test_default_output_path_can_include_iteration_label() -> None:
    path = default_output_path(
        rows=4,
        cols=4,
        games=100,
        simulations=250,
        iteration=3,
    )

    assert path == Path("runs/stage-3.6/guided-self-play-4x4-iter003-games100-sims250.jsonl")


def test_metadata_output_path_uses_jsonl_stem() -> None:
    assert metadata_output_path(Path("runs/stage-3.6/example.jsonl")) == Path(
        "runs/stage-3.6/example.meta.json"
    )


def test_run_metadata_records_full_parameter_set() -> None:
    metadata = run_metadata(
        out_path=Path("runs/stage-3.6/guided-self-play-4x4-games100-sims250.jsonl"),
        checkpoint=Path("runs/stage-3.3/checkpoint.npz"),
        games=100,
        rows=4,
        cols=4,
        simulations=250,
        iteration=3,
        seed=6001,
        c_puct=1.5,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        device="gpu",
        debug=True,
    )

    assert metadata["output"] == "runs/stage-3.6/guided-self-play-4x4-games100-sims250.jsonl"
    assert metadata["checkpoint"] == "runs/stage-3.3/checkpoint.npz"
    assert metadata["simulations"] == 250
    assert metadata["iteration"] == 3
    assert metadata["seed"] == 6001
    assert metadata["cPuct"] == 1.5
    assert metadata["rootDirichletAlpha"] == 0.3
    assert metadata["rootExplorationFraction"] == 0.25
    assert metadata["mlxDevice"] == "gpu"
    assert metadata["debug"] is True


def test_ensure_outputs_do_not_exist_refuses_existing_jsonl(tmp_path) -> None:
    out_path = tmp_path / "guided-self-play-4x4-iter001-games10-sims25.jsonl"
    out_path.write_text("", encoding="utf8")

    try:
        ensure_outputs_do_not_exist(out_path)
    except FileExistsError as error:
        assert "--overwrite" in str(error)
    else:
        raise AssertionError("Expected FileExistsError")


def test_ensure_outputs_do_not_exist_allows_overwrite(tmp_path) -> None:
    out_path = tmp_path / "guided-self-play-4x4-iter001-games10-sims25.jsonl"
    out_path.write_text("", encoding="utf8")

    ensure_outputs_do_not_exist(out_path, overwrite=True)
