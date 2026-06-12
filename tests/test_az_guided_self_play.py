import random
from pathlib import Path

from dots_boxes_mcts.az_guided_self_play import (
    default_output_path,
    ensure_outputs_do_not_exist,
    format_game_progress,
    format_turn_progress,
    metadata_output_path,
    play_guided_self_play_game,
    run_metadata,
    select_self_play_move,
)
from dots_boxes_mcts.game import legal_moves
from dots_boxes_mcts.mcts import SearchResult, SearchStats


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
        temperature_moves=8,
        sampling_temperature=1.0,
        device="gpu",
        debug=True,
        reuse_tree=True,
        evaluator_cache_entries=50000,
    )

    assert metadata["output"] == "runs/stage-3.6/guided-self-play-4x4-games100-sims250.jsonl"
    assert metadata["checkpoint"] == "runs/stage-3.3/checkpoint.npz"
    assert metadata["simulations"] == 250
    assert metadata["iteration"] == 3
    assert metadata["seed"] == 6001
    assert metadata["cPuct"] == 1.5
    assert metadata["rootDirichletAlpha"] == 0.3
    assert metadata["rootExplorationFraction"] == 0.25
    assert metadata["temperatureMoves"] == 8
    assert metadata["samplingTemperature"] == 1.0
    assert metadata["mlxDevice"] == "gpu"
    assert metadata["debug"] is True
    assert metadata["reuseTree"] is True
    assert metadata["evaluatorCacheEntries"] == 50000


def test_select_self_play_move_samples_from_visits_during_temperature_window() -> None:
    result = SearchResult(
        move="best",
        simulations=10,
        root_player=0,
        stats=[
            SearchStats(move="best", visits=9, mean_value=0.5),
            SearchStats(move="explore", visits=1, mean_value=0.0),
        ],
    )

    move, selection = select_self_play_move(
        result=result,
        turn=0,
        rng=random.Random(2),
        temperature_moves=8,
        sampling_temperature=1.0,
    )

    assert move == "explore"
    assert selection == "sampled_visit_counts"


def test_select_self_play_move_uses_max_visit_after_temperature_window() -> None:
    result = SearchResult(
        move="best",
        simulations=10,
        root_player=0,
        stats=[
            SearchStats(move="best", visits=9, mean_value=0.5),
            SearchStats(move="explore", visits=1, mean_value=0.0),
        ],
    )

    move, selection = select_self_play_move(
        result=result,
        turn=8,
        rng=random.Random(2),
        temperature_moves=8,
        sampling_temperature=1.0,
    )

    assert move == "best"
    assert selection == "max_visit"


def test_guided_self_play_advances_tree_with_sampled_move(monkeypatch) -> None:
    class FakeNetworkGuidedMCTS:
        instances = []

        def __init__(self, **kwargs) -> None:
            self.advanced_moves = []
            FakeNetworkGuidedMCTS.instances.append(self)

        def search_reusing_tree(self, state):
            moves = legal_moves(state)
            stats = [SearchStats(move=moves[0], visits=9, mean_value=0.0)]
            if len(moves) > 1:
                stats.append(SearchStats(move=moves[1], visits=1, mean_value=0.0))
            return SearchResult(
                move=moves[0],
                simulations=10,
                root_player=state.current_player,
                stats=stats,
            )

        def advance_tree(self, move, next_state):
            self.advanced_moves.append(move)
            return True

    monkeypatch.setattr(
        "dots_boxes_mcts.az_guided_self_play.NetworkEvaluator",
        lambda checkpoint, device: object(),
    )
    monkeypatch.setattr(
        "dots_boxes_mcts.az_guided_self_play.NetworkGuidedMCTS",
        FakeNetworkGuidedMCTS,
    )

    record = play_guided_self_play_game(
        checkpoint=Path("checkpoint.npz"),
        rows=2,
        cols=2,
        simulations=10,
        seed=2,
        temperature_moves=1,
    )

    selected = record["decisions"][0]["selectedMove"]
    preferred = record["decisions"][0]["searchPreferredMove"]
    searcher = FakeNetworkGuidedMCTS.instances[0]
    assert selected != preferred
    assert searcher.advanced_moves[0] == selected


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
