from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np

from dots_boxes_mcts.game import GameState, all_edge_ids, apply_move, legal_moves, state_snapshot
from dots_boxes_mcts.mcts import SearchResult, SearchStats, result_payload

try:
    from numba import njit
except ImportError as error:  # pragma: no cover - exercised only without optional runtime.
    njit = None
    NUMBA_IMPORT_ERROR = error
else:
    NUMBA_IMPORT_ERROR = None


@dataclass(frozen=True)
class FastState:
    rows: int
    cols: int
    current_player: int
    edges: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    edge_count: int


@dataclass(frozen=True)
class FastBoard:
    rows: int
    cols: int
    action_ids: list[str]
    edge_box_a: np.ndarray
    edge_box_b: np.ndarray


def require_numba() -> None:
    if njit is None:
        raise ImportError(
            "Numba is required for dots_boxes_mcts.fast_mcts. "
            "Run `pyenv activate data && python -m pip install numba`."
        ) from NUMBA_IMPORT_ERROR


def fast_board(rows: int, cols: int) -> FastBoard:
    action_ids = all_edge_ids(rows, cols)
    edge_box_a = np.full(len(action_ids), -1, dtype=np.int16)
    edge_box_b = np.full(len(action_ids), -1, dtype=np.int16)
    for index, edge in enumerate(action_ids):
        boxes = edge_adjacent_box_indices(edge, rows=rows, cols=cols)
        if boxes:
            edge_box_a[index] = boxes[0]
        if len(boxes) > 1:
            edge_box_b[index] = boxes[1]
    return FastBoard(
        rows=rows,
        cols=cols,
        action_ids=action_ids,
        edge_box_a=edge_box_a,
        edge_box_b=edge_box_b,
    )


def edge_adjacent_box_indices(edge: str, *, rows: int, cols: int) -> list[int]:
    kind, row_text, col_text = edge.split(":")
    row = int(row_text)
    col = int(col_text)
    boxes: list[int] = []
    if kind == "h":
        if row > 0:
            boxes.append((row - 1) * (cols - 1) + col)
        if row < rows - 1:
            boxes.append(row * (cols - 1) + col)
    else:
        if col > 0:
            boxes.append(row * (cols - 1) + (col - 1))
        if col < cols - 1:
            boxes.append(row * (cols - 1) + col)
    return boxes


def fast_state_from_game(state: GameState) -> FastState:
    board = fast_board(state.rows, state.cols)
    move_to_index = {move: index for index, move in enumerate(board.action_ids)}
    edges = np.zeros(len(board.action_ids), dtype=np.uint8)
    for edge in state.edges:
        edges[move_to_index[edge]] = 1
    boxes = np.full((state.rows - 1) * (state.cols - 1), -1, dtype=np.int8)
    for row_index, row in enumerate(state.boxes):
        for col_index, owner in enumerate(row):
            if owner is not None:
                boxes[row_index * (state.cols - 1) + col_index] = int(owner)
    return FastState(
        rows=state.rows,
        cols=state.cols,
        current_player=state.current_player,
        edges=edges,
        boxes=boxes,
        scores=np.array(state.scores, dtype=np.int16),
        edge_count=len(state.edges),
    )


def fast_state_from_snapshot(snapshot: dict) -> FastState:
    game = GameState(
        rows=int(snapshot["rows"]),
        cols=int(snapshot["cols"]),
        current_player=int(snapshot["currentPlayer"]),
        edges=frozenset(str(edge) for edge in snapshot["edges"]),
        edge_owners=tuple((str(edge), int(owner)) for edge, owner in snapshot["edgeOwners"]),
        boxes=tuple(
            tuple(None if cell is None else int(cell) for cell in row)
            for row in snapshot["boxes"]
        ),
        scores=(int(snapshot["scores"][0]), int(snapshot["scores"][1])),
        terminal=bool(snapshot.get("terminal", False)),
        winner=snapshot.get("winner"),
    )
    return fast_state_from_game(game)


def fast_apply_move(state: FastState, move: str) -> FastState:
    require_numba()
    board = fast_board(state.rows, state.cols)
    move_index = {edge: index for index, edge in enumerate(board.action_ids)}[move]
    edges = state.edges.copy()
    boxes = state.boxes.copy()
    scores = state.scores.copy()
    current_player, edge_count = _apply_move_in_place(
        edges,
        boxes,
        scores,
        state.current_player,
        state.edge_count,
        move_index,
        board.edge_box_a,
        board.edge_box_b,
        state.rows,
        state.cols,
    )
    return FastState(
        rows=state.rows,
        cols=state.cols,
        current_player=int(current_player),
        edges=edges,
        boxes=boxes,
        scores=scores,
        edge_count=int(edge_count),
    )


def game_from_fast_state(state: FastState) -> GameState:
    game = GameState(
        rows=state.rows,
        cols=state.cols,
        current_player=int(state.current_player),
        boxes=tuple(
            tuple(
                None if state.boxes[row * (state.cols - 1) + col] < 0 else int(
                    state.boxes[row * (state.cols - 1) + col]
                )
                for col in range(state.cols - 1)
            )
            for row in range(state.rows - 1)
        ),
    )
    for index, drawn in enumerate(state.edges):
        if drawn:
            game = apply_move(game, all_edge_ids(state.rows, state.cols)[index])
    return game


class FastUCTMCTS:
    def __init__(
        self,
        simulations: int = 50_000,
        exploration_constant: float = math.sqrt(2),
        seed: int = 1,
    ) -> None:
        require_numba()
        if simulations < 1:
            raise ValueError("simulations must be at least 1")
        if exploration_constant < 0:
            raise ValueError("exploration_constant must be non-negative")
        self.simulations = simulations
        self.exploration_constant = exploration_constant
        self.seed = seed

    def search(self, state: GameState) -> SearchResult:
        if state.terminal:
            raise ValueError("Cannot search from a terminal state.")
        board = fast_board(state.rows, state.cols)
        fast_state = fast_state_from_game(state)
        visits, values = _search_numba(
            fast_state.current_player,
            fast_state.edges,
            fast_state.boxes,
            fast_state.scores,
            fast_state.edge_count,
            board.edge_box_a,
            board.edge_box_b,
            state.rows,
            state.cols,
            self.simulations,
            self.exploration_constant,
            self.seed,
        )
        player = state.current_player
        stats = [
            SearchStats(
                move=board.action_ids[index],
                visits=int(visits[index]),
                mean_value=float(values[index] / visits[index]) if visits[index] else 0.0,
            )
            for index in range(len(board.action_ids))
            if visits[index] > 0
        ]
        stats.sort(key=lambda stat: (-stat.visits, stat.move))
        if not stats:
            raise ValueError("Search did not expand any legal moves.")
        best = max(
            stats,
            key=lambda stat: (stat.visits, stat.mean_value, stat.move),
        )
        return SearchResult(
            move=best.move,
            simulations=self.simulations,
            root_player=player,
            stats=stats,
        )

    def choose_move(self, state: GameState) -> str:
        return self.search(state).move


if njit is not None:

    @njit(cache=True)
    def _box_edge_count(edges, rows, cols, box_index):
        box_cols = cols - 1
        row = box_index // box_cols
        col = box_index - row * box_cols
        top = row * (cols - 1) + col
        bottom = (row + 1) * (cols - 1) + col
        vertical_start = rows * (cols - 1)
        left = vertical_start + row * cols + col
        right = vertical_start + row * cols + col + 1
        return edges[top] + edges[bottom] + edges[left] + edges[right]

    @njit(cache=True)
    def _apply_move_in_place(
        edges,
        boxes,
        scores,
        current_player,
        edge_count,
        move,
        edge_box_a,
        edge_box_b,
        rows,
        cols,
    ):
        edges[move] = 1
        edge_count += 1
        scored = 0
        box_a = edge_box_a[move]
        if box_a >= 0 and boxes[box_a] < 0 and _box_edge_count(edges, rows, cols, box_a) == 4:
            boxes[box_a] = current_player
            scores[current_player] += 1
            scored += 1
        box_b = edge_box_b[move]
        if box_b >= 0 and boxes[box_b] < 0 and _box_edge_count(edges, rows, cols, box_b) == 4:
            boxes[box_b] = current_player
            scores[current_player] += 1
            scored += 1
        if scored == 0:
            current_player = 1 - current_player
        return current_player, edge_count

    @njit(cache=True)
    def _terminal_value(scores, player, total_boxes):
        if total_boxes <= 0:
            return 0.0
        opponent = 1 - player
        return (scores[player] - scores[opponent]) / total_boxes

    @njit(cache=True)
    def _rollout(
        start_current_player,
        start_edges,
        start_boxes,
        start_scores,
        start_edge_count,
        edge_box_a,
        edge_box_b,
        rows,
        cols,
    ):
        edges = start_edges.copy()
        boxes = start_boxes.copy()
        scores = start_scores.copy()
        current_player = start_current_player
        edge_count = start_edge_count
        action_count = edges.shape[0]
        while edge_count < action_count:
            legal_count = action_count - edge_count
            choice = np.random.randint(legal_count)
            seen = 0
            move = 0
            for index in range(action_count):
                if edges[index] == 0:
                    if seen == choice:
                        move = index
                        break
                    seen += 1
            current_player, edge_count = _apply_move_in_place(
                edges,
                boxes,
                scores,
                current_player,
                edge_count,
                move,
                edge_box_a,
                edge_box_b,
                rows,
                cols,
            )
        return scores

    @njit(cache=True)
    def _search_numba(
        root_current_player,
        root_edges,
        root_boxes,
        root_scores,
        root_edge_count,
        edge_box_a,
        edge_box_b,
        rows,
        cols,
        simulations,
        exploration_constant,
        seed,
    ):
        np.random.seed(seed)
        action_count = root_edges.shape[0]
        total_boxes = root_boxes.shape[0]
        max_nodes = simulations + 1
        node_edges = np.zeros((max_nodes, action_count), dtype=np.uint8)
        node_boxes = np.full((max_nodes, total_boxes), -1, dtype=np.int8)
        node_scores = np.zeros((max_nodes, 2), dtype=np.int16)
        node_current_player = np.zeros(max_nodes, dtype=np.int8)
        node_edge_count = np.zeros(max_nodes, dtype=np.int16)
        node_parent = np.full(max_nodes, -1, dtype=np.int32)
        node_move = np.full(max_nodes, -1, dtype=np.int16)
        node_visits = np.zeros(max_nodes, dtype=np.int32)
        node_total_values = np.zeros((max_nodes, 2), dtype=np.float64)
        children = np.full((max_nodes, action_count), -1, dtype=np.int32)
        untried = np.full((max_nodes, action_count), -1, dtype=np.int16)
        untried_count = np.zeros(max_nodes, dtype=np.int16)

        node_edges[0, :] = root_edges
        node_boxes[0, :] = root_boxes
        node_scores[0, :] = root_scores
        node_current_player[0] = root_current_player
        node_edge_count[0] = root_edge_count
        for move in range(action_count):
            if root_edges[move] == 0:
                untried[0, untried_count[0]] = move
                untried_count[0] += 1

        node_count = 1
        for _ in range(simulations):
            node = 0
            while node_edge_count[node] < action_count and untried_count[node] == 0:
                player = node_current_player[node]
                log_parent = math.log(max(node_visits[node], 1))
                best_score = -1.0e100
                best_child = -1
                for move in range(action_count):
                    child = children[node, move]
                    if child < 0:
                        continue
                    if node_visits[child] == 0:
                        score = 1.0e100
                    else:
                        exploit = node_total_values[child, player] / node_visits[child]
                        explore = exploration_constant * math.sqrt(log_parent / node_visits[child])
                        score = exploit + explore
                    if score > best_score:
                        best_score = score
                        best_child = child
                node = best_child

            if node_edge_count[node] < action_count and untried_count[node] > 0:
                slot = np.random.randint(untried_count[node])
                move = untried[node, slot]
                untried_count[node] -= 1
                untried[node, slot] = untried[node, untried_count[node]]
                untried[node, untried_count[node]] = -1

                child = node_count
                node_count += 1
                node_edges[child, :] = node_edges[node, :]
                node_boxes[child, :] = node_boxes[node, :]
                node_scores[child, :] = node_scores[node, :]
                node_current_player[child] = node_current_player[node]
                node_edge_count[child] = node_edge_count[node]
                next_player, next_edge_count = _apply_move_in_place(
                    node_edges[child],
                    node_boxes[child],
                    node_scores[child],
                    node_current_player[child],
                    node_edge_count[child],
                    move,
                    edge_box_a,
                    edge_box_b,
                    rows,
                    cols,
                )
                node_current_player[child] = next_player
                node_edge_count[child] = next_edge_count
                node_parent[child] = node
                node_move[child] = move
                children[node, move] = child
                for child_move in range(action_count):
                    if node_edges[child, child_move] == 0:
                        untried[child, untried_count[child]] = child_move
                        untried_count[child] += 1
                node = child

            if node_edge_count[node] >= action_count:
                terminal_scores = node_scores[node]
            else:
                terminal_scores = _rollout(
                    node_current_player[node],
                    node_edges[node],
                    node_boxes[node],
                    node_scores[node],
                    node_edge_count[node],
                    edge_box_a,
                    edge_box_b,
                    rows,
                    cols,
                )

            cursor = node
            while cursor >= 0:
                node_visits[cursor] += 1
                node_total_values[cursor, 0] += _terminal_value(terminal_scores, 0, total_boxes)
                node_total_values[cursor, 1] += _terminal_value(terminal_scores, 1, total_boxes)
                cursor = node_parent[cursor]

        root_child_visits = np.zeros(action_count, dtype=np.int32)
        root_child_values = np.zeros(action_count, dtype=np.float64)
        for move in range(action_count):
            child = children[0, move]
            if child >= 0:
                root_child_visits[move] = node_visits[child]
                root_child_values[move] = node_total_values[child, root_current_player]
        return root_child_visits, root_child_values

else:

    def _search_numba(*args, **kwargs):  # type: ignore[no-untyped-def]
        require_numba()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fast Numba UCT MCTS from a fresh position.")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exploration-constant", type=float, default=math.sqrt(2))
    args = parser.parse_args()

    from dots_boxes_mcts.game import new_game

    state = new_game(rows=args.rows, cols=args.cols)
    result = FastUCTMCTS(
        simulations=args.simulations,
        exploration_constant=args.exploration_constant,
        seed=args.seed,
    ).search(state)
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
