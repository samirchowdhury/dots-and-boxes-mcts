from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from dots_boxes_mcts.az_mcts import NetworkEvaluator, NetworkGuidedMCTS
from dots_boxes_mcts.evaluate import summarize_records
from dots_boxes_mcts.external_games import edge_to_papg_index, external_game_record
from dots_boxes_mcts.game import GameState, apply_move, legal_moves, new_game, state_snapshot
from dots_boxes_mcts.mcts import UCTMCTS, result_payload
from dots_boxes_mcts.self_play import write_jsonl

PAPG_BASE_URL = "http://www.papg.com"
PAPG_NEW_GAME_URL = f"{PAPG_BASE_URL}/dab.html"
PAPG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; dots-boxes-mcts/0.1; paced research replay recorder)",
    "Referer": PAPG_NEW_GAME_URL,
}
PAPG_SIZE_VALUES = {
    (3, 3): "0",
    (4, 4): "1",
    (5, 4): "2",
    (6, 4): "3",
}


def play_mcts_vs_papg_game(
    *,
    rows: int = 4,
    cols: int = 4,
    simulations: int = 100,
    seed: int = 1,
    request_delay: float = 5.0,
    timeout: float = 30.0,
    debug_dir: Path | None = None,
) -> dict:
    mcts = UCTMCTS(simulations=simulations, seed=seed)
    return play_searcher_vs_papg_game(
        searcher=mcts,
        bot=f"uct_mcts_{simulations}",
        rows=rows,
        cols=cols,
        simulations=simulations,
        seed=seed,
        request_delay=request_delay,
        timeout=timeout,
        debug_dir=debug_dir,
        notes=f"Live Papg evaluation with {simulations} UCT simulations per move.",
    )


def play_network_guided_mcts_vs_papg_game(
    *,
    checkpoint: Path,
    rows: int = 4,
    cols: int = 4,
    simulations: int = 100,
    seed: int = 1,
    c_puct: float = 1.5,
    request_delay: float = 5.0,
    timeout: float = 30.0,
    debug_dir: Path | None = None,
    device: str = "cpu",
) -> dict:
    evaluator = NetworkEvaluator(checkpoint=checkpoint, device=device)
    searcher = NetworkGuidedMCTS(
        evaluator=evaluator,
        simulations=simulations,
        c_puct=c_puct,
        seed=seed,
    )
    return play_searcher_vs_papg_game(
        searcher=searcher,
        bot=checkpoint_bot_name(checkpoint=checkpoint, simulations=simulations),
        rows=rows,
        cols=cols,
        simulations=simulations,
        seed=seed,
        request_delay=request_delay,
        timeout=timeout,
        debug_dir=debug_dir,
        notes=(
            f"Live Papg evaluation with {checkpoint} and "
            f"{simulations} network-guided simulations per move."
        ),
        record_fields={
            "checkpoint": str(checkpoint),
            "cPuct": c_puct,
            "mlxDevice": device,
        },
    )


def play_searcher_vs_papg_game(
    *,
    searcher,
    bot: str,
    rows: int,
    cols: int,
    simulations: int,
    seed: int,
    request_delay: float,
    timeout: float,
    debug_dir: Path | None,
    notes: str,
    record_fields: dict | None = None,
) -> dict:
    client = PapgClient(request_delay=request_delay, timeout=timeout, debug_dir=debug_dir)
    page = client.new_game(rows=rows, cols=cols)
    state = new_game(rows=rows, cols=cols)
    moves: list[str] = []
    decisions: list[dict] = []

    while not state.terminal:
        synced_moves = sync_papg_moves(state, page)
        for papg_move in synced_moves:
            moves.append(papg_move)
            state = apply_move(state, papg_move)
        if state.terminal:
            break

        result = searcher.search(state)
        move = result.move
        decisions.append(
            {
                "turn": len(moves),
                "player": state.current_player,
                "state": state_snapshot(state),
                "search": result_payload(result),
            }
        )

        page = client.play_move(page, edge_to_papg_index(move, rows=rows, cols=cols))
        moves.append(move)
        state = apply_move(state, move)

        for papg_move in sync_papg_moves(state, page):
            moves.append(papg_move)
            state = apply_move(state, papg_move)

    record = external_game_record(
        source="papg",
        opponent="papg",
        bot=bot,
        rows=rows,
        cols=cols,
        moves=moves,
        our_player=0,
        notes=notes,
    )
    record["seed"] = seed
    record["simulations"] = simulations
    record["decisions"] = decisions
    if record_fields:
        record.update(record_fields)
    return record


def checkpoint_bot_name(*, checkpoint: Path, simulations: int) -> str:
    return f"network_guided_mcts_{simulations}_{checkpoint.stem}"


def sync_papg_moves(state: GameState, page: PapgPage) -> list[str]:
    missing_papg_moves = [
        edge
        for edge in page.drawn_edges
        if edge not in state.edges
    ]
    return infer_papg_reply(state, missing_papg_moves)


def generate_mcts_vs_papg_games(
    *,
    games: int,
    rows: int = 4,
    cols: int = 4,
    simulations: int = 100,
    seed: int = 1,
    request_delay: float = 5.0,
    timeout: float = 30.0,
    debug_dir: Path | None = None,
) -> list[dict]:
    records: list[dict] = []
    for game_index in range(games):
        record = play_mcts_vs_papg_game(
            rows=rows,
            cols=cols,
            simulations=simulations,
            seed=seed + game_index,
            request_delay=request_delay,
            timeout=timeout,
            debug_dir=debug_dir,
        )
        record["gameIndex"] = game_index
        records.append(record)
    return records


def generate_network_guided_mcts_vs_papg_games(
    *,
    checkpoint: Path,
    games: int,
    rows: int = 4,
    cols: int = 4,
    simulations: int = 100,
    seed: int = 1,
    c_puct: float = 1.5,
    request_delay: float = 5.0,
    timeout: float = 30.0,
    debug_dir: Path | None = None,
    device: str = "cpu",
) -> list[dict]:
    records: list[dict] = []
    evaluator = NetworkEvaluator(checkpoint=checkpoint, device=device)
    for game_index in range(games):
        game_seed = seed + game_index
        searcher = NetworkGuidedMCTS(
            evaluator=evaluator,
            simulations=simulations,
            c_puct=c_puct,
            seed=game_seed,
        )
        record = play_searcher_vs_papg_game(
            searcher=searcher,
            bot=checkpoint_bot_name(checkpoint=checkpoint, simulations=simulations),
            rows=rows,
            cols=cols,
            simulations=simulations,
            seed=game_seed,
            request_delay=request_delay,
            timeout=timeout,
            debug_dir=debug_dir,
            notes=(
                f"Live Papg evaluation with {checkpoint} and "
                f"{simulations} network-guided simulations per move."
            ),
            record_fields={
                "checkpoint": str(checkpoint),
                "cPuct": c_puct,
                "mlxDevice": device,
            },
        )
        record["gameIndex"] = game_index
        records.append(record)
    return records


def infer_papg_reply(
    state: GameState,
    missing_moves: list[str],
    papg_player: int = 1,
) -> list[str]:
    if not missing_moves:
        return []
    if papg_player not in {0, 1}:
        raise ValueError("papg_player must be 0 or 1")

    missing = set(missing_moves)
    solutions: list[list[str]] = []
    our_player = 1 - papg_player

    def search(current_state: GameState, remaining: set[str], sequence: list[str]) -> None:
        if not remaining:
            solutions.append(sequence)
            return
        if current_state.current_player != papg_player:
            return

        for move in sorted(remaining):
            next_state = apply_move(current_state, move)
            if next_state.current_player == our_player and len(remaining) > 1:
                continue
            search(next_state, remaining - {move}, [*sequence, move])

    search(state, missing, [])
    if not solutions:
        raise ValueError(f"Could not infer a legal Papg reply from moves: {sorted(missing)}")

    return solutions[0]


class PapgClient:
    def __init__(
        self,
        *,
        request_delay: float = 5.0,
        timeout: float = 30.0,
        debug_dir: Path | None = None,
    ) -> None:
        if request_delay < 1.0:
            raise ValueError("request_delay must be at least 1 second to avoid hitting Papg too quickly.")
        self.request_delay = request_delay
        self.timeout = timeout
        self.last_request_at = 0.0
        self.debug_dir = debug_dir
        self.request_count = 0

    def new_game(self, *, rows: int, cols: int) -> PapgPage:
        if (rows, cols) not in PAPG_SIZE_VALUES:
            raise ValueError(f"Papg board size is not configured for {rows}x{cols} dots.")

        return initial_papg_page(rows=rows, cols=cols)

    def play_move(self, page: PapgPage, papg_index: int) -> PapgPage:
        try:
            href = page.move_links[papg_index]
        except KeyError as error:
            raise ValueError(
                f"Papg move index {papg_index} is not currently legal. "
                f"Current Papg legal indexes: {sorted(page.move_links)}"
            ) from error

        url = urljoin(PAPG_BASE_URL, href)
        html = self._open(Request(url, headers=PAPG_HEADERS))
        parsed_page = parse_papg_page(html, rows=page.rows, cols=page.cols)

        poll_url = papg_thinking_url(url)
        attempts = 0
        while is_thinking_page(html) and not parsed_page.move_links and attempts < 5:
            html = self._open(Request(poll_url, headers=PAPG_HEADERS))
            parsed_page = parse_papg_page(html, rows=page.rows, cols=page.cols)
            attempts += 1

        return parsed_page

    def _open(self, request: Request | str) -> str:
        elapsed = time.monotonic() - self.last_request_at
        if self.last_request_at > 0 and elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

        with urlopen(request, timeout=self.timeout) as response:
            self.last_request_at = time.monotonic()
            html = response.read().decode("utf8", errors="replace")
            self._write_debug_html(html)
            return html

    def _write_debug_html(self, html: str) -> None:
        if self.debug_dir is None:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.request_count += 1
        (self.debug_dir / f"papg-response-{self.request_count:03}.html").write_text(html, encoding="utf8")


class PapgPage:
    def __init__(
        self,
        *,
        rows: int,
        cols: int,
        move_links: dict[int, str],
        edge_owners: dict[str, int],
        drawn_edges: set[str],
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.move_links = move_links
        self.edge_owners = edge_owners
        self.drawn_edges = drawn_edges


def initial_papg_page(*, rows: int, cols: int) -> PapgPage:
    state_string = initial_papg_state_string(rows=rows, cols=cols)
    move_links = {
        edge_to_papg_index(edge, rows=rows, cols=cols): f"/dab?.+1+1+0+0+{edge_to_papg_index(edge, rows=rows, cols=cols)}+{state_string}"
        for edge in legal_moves(new_game(rows=rows, cols=cols))
    }
    return PapgPage(rows=rows, cols=cols, move_links=move_links, edge_owners={}, drawn_edges=set())


def initial_papg_state_string(*, rows: int, cols: int) -> str:
    chars: list[str] = []
    for table_row in range(rows * 2 - 1):
        for table_col in range(cols * 2 - 1):
            if table_row % 2 == 0 and table_col % 2 == 0:
                chars.append("6")
            elif table_row % 2 == 1 and table_col % 2 == 1:
                chars.append("3")
            else:
                chars.append("0")
    return "".join(chars)


def parse_papg_page(html: str, *, rows: int, cols: int) -> PapgPage:
    move_links = parse_move_links(html)
    edge_owners = parse_edge_owners(html, rows=rows, cols=cols)
    return PapgPage(
        rows=rows,
        cols=cols,
        move_links=move_links,
        edge_owners=edge_owners,
        drawn_edges=parse_drawn_edges(move_links=move_links, edge_owners=edge_owners, rows=rows, cols=cols),
    )


def is_thinking_page(html: str) -> bool:
    return "Thinking..." in html


def papg_thinking_url(move_url: str) -> str:
    return move_url.replace("/dab?.+1+", "/dab?.+2+", 1)


def parse_drawn_edges(
    *,
    move_links: dict[int, str],
    edge_owners: dict[str, int],
    rows: int,
    cols: int,
) -> set[str]:
    if not move_links:
        return set(edge_owners)

    legal_indexes = set(move_links)
    return {
        edge
        for edge in legal_moves(new_game(rows=rows, cols=cols))
        if edge_to_papg_index(edge, rows=rows, cols=cols) not in legal_indexes
    }


def parse_move_links(html: str) -> dict[int, str]:
    links: dict[int, str] = {}
    for href, raw_index in re.findall(r'href="?([^"\s>]*/dab\?\.\+1\+1\+0\+0\+(\d+)\+[^"\s>]*)"?', html):
        links[int(raw_index)] = href
    return links


def parse_edge_owners(html: str, *, rows: int, cols: int) -> dict[str, int]:
    board_match = re.search(r"<table[^>]*>(.*?)</table>", html, flags=re.DOTALL | re.IGNORECASE)
    if not board_match:
        return {}

    edge_owners: dict[str, int] = {}
    table_html = board_match.group(1)
    row_htmls = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.DOTALL | re.IGNORECASE)
    for table_row, row_html in enumerate(row_htmls[: rows * 2 - 1]):
        cell_htmls = re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.DOTALL | re.IGNORECASE)
        for table_col, cell_html in enumerate(cell_htmls[: cols * 2 - 1]):
            owner = edge_owner_from_cell(cell_html)
            if owner is None:
                continue
            if table_row % 2 == 0 and table_col % 2 == 1:
                edge_owners[f"h:{table_row // 2}:{(table_col - 1) // 2}"] = owner
            elif table_row % 2 == 1 and table_col % 2 == 0:
                edge_owners[f"v:{(table_row - 1) // 2}:{table_col // 2}"] = owner
    return edge_owners


def edge_owner_from_cell(cell_html: str) -> int | None:
    image_match = re.search(r"/assets/dab_[HV]([BR])\.gif", cell_html)
    if image_match is None:
        return None
    return 0 if image_match.group(1) == "B" else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an MCTS player against the live Papg bot.")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--simulations", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Optional MLX checkpoint for network-guided MCTS.")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--mlx-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--request-delay", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/papg/stage-2.5/papg-eval.jsonl"))
    args = parser.parse_args()

    if args.games < 1:
        raise SystemExit("--games must be at least 1")

    if args.checkpoint is None:
        records = generate_mcts_vs_papg_games(
            games=args.games,
            rows=args.rows,
            cols=args.cols,
            simulations=args.simulations,
            seed=args.seed,
            request_delay=args.request_delay,
            timeout=args.timeout,
            debug_dir=args.debug_dir,
        )
    else:
        records = generate_network_guided_mcts_vs_papg_games(
            checkpoint=args.checkpoint,
            games=args.games,
            rows=args.rows,
            cols=args.cols,
            simulations=args.simulations,
            seed=args.seed,
            c_puct=args.c_puct,
            request_delay=args.request_delay,
            timeout=args.timeout,
            debug_dir=args.debug_dir,
            device=args.mlx_device,
        )
    write_jsonl(records, args.out)
    print(json.dumps(summarize_records(records, mcts_player=0), sort_keys=True))
    print(f"Wrote {len(records)} games to {args.out}")


if __name__ == "__main__":
    main()
