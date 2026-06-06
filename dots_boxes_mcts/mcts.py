from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field

from dots_boxes_mcts.game import GameState, apply_move, legal_moves, new_game, state_snapshot


@dataclass
class SearchStats:
    move: str
    visits: int
    mean_value: float


@dataclass
class SearchResult:
    move: str
    simulations: int
    root_player: int
    stats: list[SearchStats]


@dataclass
class Node:
    state: GameState
    parent: Node | None = None
    move: str | None = None
    untried_moves: list[str] = field(default_factory=list)
    children: dict[str, Node] = field(default_factory=dict)
    visits: int = 0
    total_values: list[float] = field(default_factory=lambda: [0.0, 0.0])

    @classmethod
    def from_state(
        cls,
        state: GameState,
        parent: Node | None = None,
        move: str | None = None,
        rng: random.Random | None = None,
    ) -> Node:
        moves = legal_moves(state)
        if rng is not None:
            rng.shuffle(moves)
        return cls(state=state, parent=parent, move=move, untried_moves=moves)

    def fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def mean_value(self, player: int) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_values[player] / self.visits


class UCTMCTS:
    def __init__(
        self,
        simulations: int = 100,
        exploration_constant: float = math.sqrt(2),
        seed: int | None = None,
    ) -> None:
        if simulations < 1:
            raise ValueError("simulations must be at least 1")
        if exploration_constant < 0:
            raise ValueError("exploration_constant must be non-negative")

        self.simulations = simulations
        self.exploration_constant = exploration_constant
        self.rng = random.Random(seed)

    def search(self, state: GameState) -> SearchResult:
        if state.terminal:
            raise ValueError("Cannot search from a terminal state.")

        root = Node.from_state(state, rng=self.rng)
        for _ in range(self.simulations):
            leaf = self._select_or_expand(root)
            terminal_state = self._rollout(leaf.state)
            self._backpropagate(leaf, terminal_state)

        return SearchResult(
            move=self._best_root_move(root),
            simulations=self.simulations,
            root_player=state.current_player,
            stats=self._root_stats(root),
        )

    def choose_move(self, state: GameState) -> str:
        return self.search(state).move

    def _select_or_expand(self, node: Node) -> Node:
        while not node.state.terminal:
            if not node.fully_expanded():
                return self._expand(node)
            node = self._uct_child(node)
        return node

    def _expand(self, node: Node) -> Node:
        move = node.untried_moves.pop()
        child_state = apply_move(node.state, move)
        child = Node.from_state(child_state, parent=node, move=move, rng=self.rng)
        node.children[move] = child
        return child

    def _uct_child(self, node: Node) -> Node:
        player = node.state.current_player
        log_parent_visits = math.log(max(node.visits, 1))

        def score(child: Node) -> float:
            if child.visits == 0:
                return math.inf
            exploitation = child.mean_value(player)
            exploration = self.exploration_constant * math.sqrt(log_parent_visits / child.visits)
            return exploitation + exploration

        return max(node.children.values(), key=score)

    def _rollout(self, state: GameState) -> GameState:
        while not state.terminal:
            move = self.rng.choice(legal_moves(state))
            state = apply_move(state, move)
        return state

    def _backpropagate(self, node: Node, terminal_state: GameState) -> None:
        while node is not None:
            node.visits += 1
            node.total_values[0] += terminal_value(terminal_state, 0)
            node.total_values[1] += terminal_value(terminal_state, 1)
            node = node.parent

    def _best_root_move(self, root: Node) -> str:
        if not root.children:
            raise ValueError("Search did not expand any legal moves.")
        player = root.state.current_player
        best_child = max(
            root.children.values(),
            key=lambda child: (child.visits, child.mean_value(player), child.move or ""),
        )
        assert best_child.move is not None
        return best_child.move

    def _root_stats(self, root: Node) -> list[SearchStats]:
        player = root.state.current_player
        children = sorted(
            root.children.values(),
            key=lambda child: (-child.visits, child.move or ""),
        )
        return [
            SearchStats(
                move=child.move or "",
                visits=child.visits,
                mean_value=child.mean_value(player),
            )
            for child in children
        ]


def terminal_value(state: GameState, player: int) -> float:
    total_boxes = (state.rows - 1) * (state.cols - 1)
    if total_boxes == 0:
        return 0.0
    opponent = 1 if player == 0 else 0
    return (state.scores[player] - state.scores[opponent]) / total_boxes


def result_payload(result: SearchResult) -> dict:
    return {
        "move": result.move,
        "simulations": result.simulations,
        "rootPlayer": result.root_player,
        "stats": [
            {
                "move": stat.move,
                "visits": stat.visits,
                "meanValue": stat.mean_value,
            }
            for stat in result.stats
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run plain UCT MCTS from a fresh position.")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    state = new_game(rows=args.rows, cols=args.cols)
    searcher = UCTMCTS(simulations=args.simulations, seed=args.seed)
    result = searcher.search(state)

    print(
        json.dumps(
            {
                "state": state_snapshot(state),
                "search": result_payload(result),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
