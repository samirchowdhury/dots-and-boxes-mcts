import pytest

from dots_boxes_mcts.fast_ez_mcts import CPP_IMPORT_ERROR, FastNetworkGuidedMCTS
from dots_boxes_mcts.game import GameState, legal_moves, new_game


pytestmark = pytest.mark.skipif(
    CPP_IMPORT_ERROR is not None,
    reason="C++ network-guided MCTS backend is not built",
)


class UniformSnapshotEvaluator:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def evaluate_many_snapshots(
        self,
        snapshots: list[dict],
    ) -> list[tuple[dict[str, float], float]]:
        self.batch_sizes.append(len(snapshots))
        results = []
        for snapshot in snapshots:
            moves = legal_moves(
                GameState(
                    rows=int(snapshot["rows"]),
                    cols=int(snapshot["cols"]),
                    current_player=int(snapshot["currentPlayer"]),
                    edges=frozenset(str(edge) for edge in snapshot["edges"]),
                    edge_owners=tuple(
                        (str(edge), int(owner))
                        for edge, owner in snapshot.get("edgeOwners", [])
                    ),
                    boxes=tuple(
                        tuple(None if cell is None else int(cell) for cell in row)
                        for row in snapshot["boxes"]
                    ),
                    scores=(int(snapshot["scores"][0]), int(snapshot["scores"][1])),
                    terminal=bool(snapshot.get("terminal", False)),
                    winner=snapshot.get("winner"),
                )
            )
            results.append(({move: 1.0 / len(moves) for move in moves}, 0.0))
        return results


def test_fast_network_guided_mcts_returns_legal_move_and_stats() -> None:
    state = new_game(rows=3, cols=3)
    evaluator = UniformSnapshotEvaluator()
    searcher = FastNetworkGuidedMCTS(evaluator=evaluator, simulations=8, seed=1)

    result = searcher.search(state)

    assert result.move in legal_moves(state)
    assert result.root_player == state.current_player
    assert result.simulations == 8
    assert sum(stat.visits for stat in result.stats) == 8


def test_fast_network_guided_mcts_uses_batched_leaf_evaluation() -> None:
    state = new_game(rows=3, cols=3)
    evaluator = UniformSnapshotEvaluator()
    searcher = FastNetworkGuidedMCTS(
        evaluator=evaluator,
        simulations=9,
        seed=1,
        batch_size=4,
        virtual_loss=1.0,
    )

    searcher.search(state)

    assert evaluator.batch_sizes[0] == 1
    assert any(batch_size > 1 for batch_size in evaluator.batch_sizes[1:])


def test_fast_network_guided_mcts_accepts_zero_virtual_loss() -> None:
    state = new_game(rows=3, cols=3)
    evaluator = UniformSnapshotEvaluator()
    searcher = FastNetworkGuidedMCTS(
        evaluator=evaluator,
        simulations=5,
        seed=1,
        batch_size=3,
        virtual_loss=0.0,
    )

    result = searcher.search(state)

    assert result.move in legal_moves(state)
    assert sum(stat.visits for stat in result.stats) == 5
