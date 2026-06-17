from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dots_boxes_mcts.ez_mcts import game_state_from_snapshot
from dots_boxes_mcts.game import GameState, state_snapshot
from dots_boxes_mcts.mcts import SearchResult, SearchStats

try:
    from dots_boxes_mcts import _fast_ez_mcts_cpp
except ImportError as error:  # pragma: no cover - exercised only without built extension.
    _fast_ez_mcts_cpp = None
    CPP_IMPORT_ERROR = error
else:
    CPP_IMPORT_ERROR = None


def require_cpp_backend() -> None:
    if _fast_ez_mcts_cpp is None:
        raise ImportError(
            "The C++ network-guided MCTS backend is not built. "
            "Run `python setup.py build_ext --inplace` from the repo root."
        ) from CPP_IMPORT_ERROR


@dataclass
class FastNetworkGuidedMCTS:
    evaluator: Any
    simulations: int = 25
    c_puct: float = 1.5
    seed: int | None = None
    root_dirichlet_alpha: float = 0.0
    root_exploration_fraction: float = 0.0
    batch_size: int = 1
    virtual_loss: float = 1.0

    def __post_init__(self) -> None:
        require_cpp_backend()
        if self.simulations < 1:
            raise ValueError("simulations must be at least 1")
        if self.c_puct < 0:
            raise ValueError("c_puct must be non-negative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.virtual_loss < 0:
            raise ValueError("virtual_loss must be non-negative")

    def search(self, state: GameState) -> SearchResult:
        if state.terminal:
            raise ValueError("Cannot search from a terminal state.")
        payload = _fast_ez_mcts_cpp.search(
            snapshot=state_snapshot(state),
            batch_evaluator=self._evaluate_snapshots,
            simulations=self.simulations,
            c_puct=self.c_puct,
            seed=1 if self.seed is None else int(self.seed),
            root_dirichlet_alpha=self.root_dirichlet_alpha,
            root_exploration_fraction=self.root_exploration_fraction,
            batch_size=self.batch_size,
            virtual_loss=self.virtual_loss,
        )
        return result_from_cpp_payload(payload)

    def search_reusing_tree(self, state: GameState) -> SearchResult:
        return self.search(state)

    def choose_move(self, state: GameState) -> str:
        return self.search(state).move

    def advance_tree(self, move: str, next_state: GameState) -> bool:
        return False

    def reset_tree(self) -> None:
        return None

    def _evaluate_snapshots(self, snapshots: list[dict]) -> list[tuple[dict[str, float], float]]:
        if hasattr(self.evaluator, "evaluate_many_snapshots"):
            return self.evaluator.evaluate_many_snapshots(snapshots)
        if hasattr(self.evaluator, "evaluate_snapshot"):
            return [
                self.evaluator.evaluate_snapshot(snapshot)
                for snapshot in snapshots
            ]
        return [
            self.evaluator.evaluate(game_state_from_snapshot(snapshot))
            for snapshot in snapshots
        ]


def result_from_cpp_payload(payload: dict) -> SearchResult:
    return SearchResult(
        move=str(payload["move"]),
        simulations=int(payload["simulations"]),
        root_player=int(payload["rootPlayer"]),
        stats=[
            SearchStats(
                move=str(stat["move"]),
                visits=int(stat["visits"]),
                mean_value=float(stat["meanValue"]),
            )
            for stat in payload["stats"]
        ],
    )
