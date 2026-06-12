from __future__ import annotations

import argparse
import json
import math
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from dots_boxes_mcts.encoding import action_ids, encode_snapshot
from dots_boxes_mcts.game import GameState, apply_move, legal_moves, new_game, state_snapshot
from dots_boxes_mcts.mcts import SearchResult, SearchStats, result_payload, terminal_value
from dots_boxes_mcts.train import load_mlx_checkpoint, require_mlx


@dataclass
class AZNode:
    state: GameState
    prior: float = 1.0
    move: str | None = None
    children: dict[str, AZNode] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    def expanded(self) -> bool:
        return bool(self.children)

    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class NetworkEvaluator:
    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        self.model = load_mlx_checkpoint(checkpoint, device=device)
        self.mx = require_mlx(device=device)

    def evaluate(self, state: GameState) -> tuple[dict[str, float], float]:
        encoded = encode_snapshot(state_snapshot(state))
        tensor = np.moveaxis(encoded.tensor, 0, -1)[None, ...].astype(np.float32)
        legal_mask = encoded.legal_mask[None, ...].astype(np.float32)
        policy, value = self.model.forward(self.mx.array(tensor), self.mx.array(legal_mask))
        self.mx.eval(policy, value)
        probabilities = np.array(policy)[0]
        priors = {
            move: float(probabilities[index])
            for index, move in enumerate(encoded.action_ids)
            if legal_mask[0, index] > 0
        }
        return priors, float(np.array(value)[0])


class CachedNetworkEvaluator:
    def __init__(self, evaluator: NetworkEvaluator, max_entries: int = 500_000) -> None:
        self.evaluator = evaluator
        self.max_entries = max_entries
        self.cache: OrderedDict[tuple, tuple[tuple[tuple[str, float], ...], float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def evaluate(self, state: GameState) -> tuple[dict[str, float], float]:
        key = state_key(state)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            self.hits += 1
            priors, value = cached
            return dict(priors), value

        priors, value = self.evaluator.evaluate(state)
        self.misses += 1
        if self.max_entries > 0:
            self.cache[key] = (tuple(sorted(priors.items())), value)
            if len(self.cache) > self.max_entries:
                self.cache.popitem(last=False)
        return priors, value


def state_key(state: GameState) -> tuple:
    return (
        state.rows,
        state.cols,
        state.current_player,
        tuple(sorted(state.edges)),
        state.boxes,
        state.scores,
    )


class NetworkGuidedMCTS:
    def __init__(
        self,
        evaluator: NetworkEvaluator,
        simulations: int = 25,
        c_puct: float = 1.5,
        seed: int | None = None,
        root_dirichlet_alpha: float = 0.0,
        root_exploration_fraction: float = 0.0,
    ) -> None:
        if simulations < 1:
            raise ValueError("simulations must be at least 1")
        if c_puct < 0:
            raise ValueError("c_puct must be non-negative")
        self.evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct
        self.rng = random.Random(seed)
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_fraction = root_exploration_fraction
        self._reuse_root: AZNode | None = None
        self._reuse_root_noise_applied = False

    def search(self, state: GameState) -> SearchResult:
        if state.terminal:
            raise ValueError("Cannot search from a terminal state.")

        root = AZNode(state=state)
        self._expand(root)
        self._add_root_noise(root)
        for _ in range(self.simulations):
            self._run_simulation(root)

        return self._search_result(root, self.simulations)

    def search_many(
        self,
        state: GameState,
        budgets: list[int],
        on_budget: Callable[[int, SearchResult], None] | None = None,
    ) -> list[SearchResult]:
        if state.terminal:
            raise ValueError("Cannot search from a terminal state.")
        if not budgets:
            return []
        normalized_budgets = sorted(set(budgets))
        if normalized_budgets[0] < 1:
            raise ValueError("budgets must be at least 1")

        root = AZNode(state=state)
        self._expand(root)
        self._add_root_noise(root)
        results_by_budget: dict[int, SearchResult] = {}
        next_budget_index = 0
        for simulation in range(1, normalized_budgets[-1] + 1):
            self._run_simulation(root)
            while (
                next_budget_index < len(normalized_budgets)
                and simulation == normalized_budgets[next_budget_index]
            ):
                budget = normalized_budgets[next_budget_index]
                result = self._search_result(root, budget)
                results_by_budget[budget] = result
                if on_budget is not None:
                    on_budget(budget, result)
                next_budget_index += 1

        return [results_by_budget[budget] for budget in budgets]

    def choose_move(self, state: GameState) -> str:
        return self.search(state).move

    def search_reusing_tree(self, state: GameState) -> SearchResult:
        if state.terminal:
            raise ValueError("Cannot search from a terminal state.")
        root = self._reusable_root_for(state)
        self._prepare_root(root)
        simulations_to_run = max(0, self.simulations - root.visits)
        for _ in range(simulations_to_run):
            self._run_simulation(root)
        return self._search_result(root, self.simulations)

    def advance_tree(self, move: str, next_state: GameState) -> bool:
        if self._reuse_root is None:
            self.reset_tree()
            return False

        child = self._reuse_root.children.get(move)
        if child is not None and child.state == next_state:
            self._reuse_root = child
            self._reuse_root_noise_applied = False
            return True

        self.reset_tree()
        return False

    def reset_tree(self) -> None:
        self._reuse_root = None
        self._reuse_root_noise_applied = False

    def _reusable_root_for(self, state: GameState) -> AZNode:
        if self._reuse_root is None or self._reuse_root.state != state:
            self._reuse_root = AZNode(state=state)
            self._reuse_root_noise_applied = False
        return self._reuse_root

    def _prepare_root(self, root: AZNode) -> None:
        if not root.expanded():
            self._expand(root)
        if not self._reuse_root_noise_applied:
            self._add_root_noise(root)
            self._reuse_root_noise_applied = True

    def _run_simulation(self, root: AZNode) -> None:
        node = root
        path: list[tuple[AZNode, AZNode]] = []
        while node.expanded() and not node.state.terminal:
            child = self._select_child(node)
            path.append((node, child))
            node = child

        if node.state.terminal:
            leaf_player = node.state.current_player
            leaf_value = terminal_value(node.state, leaf_player)
        else:
            leaf_player = node.state.current_player
            leaf_value = self._expand(node)

        for parent, child in path:
            child.visits += 1
            child.value_sum += leaf_value if parent.state.current_player == leaf_player else -leaf_value
        root.visits += 1

    def _search_result(self, root: AZNode, simulations: int) -> SearchResult:
        return SearchResult(
            move=self._best_root_move(root),
            simulations=simulations,
            root_player=root.state.current_player,
            stats=self._root_stats(root),
        )

    def _expand(self, node: AZNode) -> float:
        priors, value = self.evaluator.evaluate(node.state)
        for move in legal_moves(node.state):
            child_state = apply_move(node.state, move)
            node.children[move] = AZNode(
                state=child_state,
                prior=max(priors.get(move, 0.0), 0.0),
                move=move,
            )
        if node.children:
            total_prior = sum(child.prior for child in node.children.values())
            if total_prior <= 0:
                uniform = 1.0 / len(node.children)
                for child in node.children.values():
                    child.prior = uniform
            else:
                for child in node.children.values():
                    child.prior /= total_prior
        return value

    def _select_child(self, node: AZNode) -> AZNode:
        sqrt_parent = math.sqrt(max(node.visits, 1))

        def score(child: AZNode) -> float:
            q = child.mean_value()
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visits)
            return q + u

        return max(node.children.values(), key=score)

    def _add_root_noise(self, root: AZNode) -> None:
        if (
            not root.children
            or self.root_dirichlet_alpha <= 0
            or self.root_exploration_fraction <= 0
        ):
            return
        moves = sorted(root.children)
        noise = self._dirichlet(len(moves), self.root_dirichlet_alpha)
        for move, noise_value in zip(moves, noise, strict=True):
            child = root.children[move]
            child.prior = (
                (1.0 - self.root_exploration_fraction) * child.prior
                + self.root_exploration_fraction * noise_value
            )

    def _dirichlet(self, count: int, alpha: float) -> list[float]:
        samples = [self.rng.gammavariate(alpha, 1.0) for _ in range(count)]
        total = sum(samples)
        if total <= 0:
            return [1.0 / count for _ in range(count)]
        return [sample / total for sample in samples]

    def _best_root_move(self, root: AZNode) -> str:
        if not root.children:
            raise ValueError("Search did not expand any legal moves.")
        child = max(
            root.children.values(),
            key=lambda item: (item.visits, item.mean_value(), item.move or ""),
        )
        assert child.move is not None
        return child.move

    def _root_stats(self, root: AZNode) -> list[SearchStats]:
        children = sorted(root.children.values(), key=lambda child: (-child.visits, child.move or ""))
        return [
            SearchStats(
                move=child.move or "",
                visits=child.visits,
                mean_value=child.mean_value(),
            )
            for child in children
        ]


def network_policy_move(checkpoint: Path, state: GameState, device: str = "cpu") -> str:
    evaluator = NetworkEvaluator(checkpoint=checkpoint, device=device)
    priors, _ = evaluator.evaluate(state)
    if not priors:
        raise ValueError("No legal network priors available.")
    legal = set(legal_moves(state))
    return max((move for move in priors if move in legal), key=lambda move: (priors[move], move))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run network-guided MCTS from a fresh position.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=25)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.0)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.0)
    parser.add_argument("--mlx-device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()

    state = new_game(rows=args.rows, cols=args.cols)
    evaluator = NetworkEvaluator(checkpoint=args.checkpoint, device=args.mlx_device)
    searcher = NetworkGuidedMCTS(
        evaluator=evaluator,
        simulations=args.simulations,
        c_puct=args.c_puct,
        seed=args.seed,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_exploration_fraction=args.root_exploration_fraction,
    )
    result = searcher.search(state)
    print(
        json.dumps(
            {
                "state": state_snapshot(state),
                "actionIds": action_ids(args.rows, args.cols),
                "search": result_payload(result),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
