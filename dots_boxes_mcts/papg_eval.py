from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from dots_boxes_mcts.az_mcts import CachedNetworkEvaluator, NetworkEvaluator, NetworkGuidedMCTS
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
DEFAULT_EVALUATOR_CACHE_ENTRIES = 500_000


def play_mcts_vs_papg_game(
    *,
    rows: int = 4,
    cols: int = 4,
    simulations: int = 100,
    seed: int = 1,
    request_delay: float = 5.0,
    timeout: float = 30.0,
    debug_dir: Path | None = None,
    our_player: int = 0,
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
        our_player=our_player,
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
    reuse_tree: bool = True,
    evaluator_cache_entries: int = DEFAULT_EVALUATOR_CACHE_ENTRIES,
    our_player: int = 0,
) -> dict:
    network_evaluator = NetworkEvaluator(checkpoint=checkpoint, device=device)
    evaluator = (
        CachedNetworkEvaluator(network_evaluator, max_entries=evaluator_cache_entries)
        if evaluator_cache_entries > 0
        else network_evaluator
    )
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
            "reuseTree": reuse_tree,
            "evaluatorCacheEntries": evaluator_cache_entries,
        },
        reuse_tree=reuse_tree,
        our_player=our_player,
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
    reuse_tree: bool = True,
    our_player: int = 0,
) -> dict:
    if our_player not in {0, 1}:
        raise ValueError("our_player must be 0 or 1")
    papg_player = 1 - our_player
    client = PapgClient(request_delay=request_delay, timeout=timeout, debug_dir=debug_dir)
    page = client.new_game(rows=rows, cols=cols, our_player=our_player)
    state = new_game(rows=rows, cols=cols)
    moves: list[str] = []
    decisions: list[dict] = []

    while not state.terminal:
        papg_polls = 0
        while state.current_player == papg_player and not state.terminal:
            synced_moves = sync_papg_moves(state, page, papg_player=papg_player)
            if synced_moves:
                for papg_move in synced_moves:
                    moves.append(papg_move)
                    next_state = apply_move(state, papg_move)
                    advance_searcher_tree(searcher, papg_move, next_state, reuse_tree=reuse_tree)
                    state = next_state
                continue

            if papg_polls >= 10:
                raise RuntimeError(
                    "Papg did not produce a reply while it was Papg's turn. "
                    "Refusing to let the local searcher play for Papg."
                )
            page = client.poll_reply(page)
            papg_polls += 1

        if state.terminal:
            break
        if state.current_player != our_player:
            raise RuntimeError(
                f"Expected local bot player {our_player} to move, "
                f"but current player is {state.current_player}."
            )

        result = (
            searcher.search_reusing_tree(state)
            if reuse_tree and hasattr(searcher, "search_reusing_tree")
            else searcher.search(state)
        )
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
        next_state = apply_move(state, move)
        advance_searcher_tree(searcher, move, next_state, reuse_tree=reuse_tree)
        state = next_state

    record = external_game_record(
        source="papg",
        opponent="papg",
        bot=bot,
        rows=rows,
        cols=cols,
        moves=moves,
        our_player=our_player,
        notes=notes,
    )
    record["seed"] = seed
    record["simulations"] = simulations
    record["decisions"] = decisions
    if record_fields:
        record.update(record_fields)
    return record


def advance_searcher_tree(searcher, move: str, next_state: GameState, *, reuse_tree: bool) -> None:
    if reuse_tree and hasattr(searcher, "advance_tree"):
        searcher.advance_tree(move, next_state)


def checkpoint_bot_name(*, checkpoint: Path, simulations: int) -> str:
    return f"network_guided_mcts_{simulations}_{checkpoint.stem}"


def sync_papg_moves(state: GameState, page: PapgPage, *, papg_player: int = 1) -> list[str]:
    missing_papg_moves = [
        edge
        for edge in page.drawn_edges
        if edge not in state.edges
    ]
    return infer_papg_reply(state, missing_papg_moves, papg_player=papg_player)


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
    our_player: int = 0,
    alternate_players: bool = False,
) -> list[dict]:
    records: list[dict] = []
    for game_index in range(games):
        game_our_player = alternating_our_player(game_index) if alternate_players else our_player
        record = play_mcts_vs_papg_game(
            rows=rows,
            cols=cols,
            simulations=simulations,
            seed=seed + game_index,
            request_delay=request_delay,
            timeout=timeout,
            debug_dir=debug_dir,
            our_player=game_our_player,
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
    reuse_tree: bool = True,
    evaluator_cache_entries: int = DEFAULT_EVALUATOR_CACHE_ENTRIES,
    our_player: int = 0,
    alternate_players: bool = False,
) -> list[dict]:
    records: list[dict] = []
    network_evaluator = NetworkEvaluator(checkpoint=checkpoint, device=device)
    for game_index in range(games):
        game_seed = seed + game_index
        game_our_player = alternating_our_player(game_index) if alternate_players else our_player
        evaluator = (
            CachedNetworkEvaluator(network_evaluator, max_entries=evaluator_cache_entries)
            if evaluator_cache_entries > 0
            else network_evaluator
        )
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
                "reuseTree": reuse_tree,
                "evaluatorCacheEntries": evaluator_cache_entries,
            },
            reuse_tree=reuse_tree,
            our_player=game_our_player,
        )
        record["gameIndex"] = game_index
        records.append(record)
    return records


def alternating_our_player(game_index: int) -> int:
    return game_index % 2


def summarize_papg_records(records: list[dict]) -> dict:
    summary = summarize_papg_record_subset(records)
    summary["byOurPlayer"] = {}
    for player in (0, 1):
        subset = [record for record in records if int(record.get("ourPlayer", 0)) == player]
        if subset:
            summary["byOurPlayer"][str(player)] = summarize_papg_record_subset(subset)
    return summary


def summarize_papg_record_subset(records: list[dict]) -> dict:
    if not records:
        return {
            "games": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "winRate": 0.0,
            "averageScoreMargin": 0.0,
            "strategic": summarize_records_by_perspective([]),
        }
    margins = [
        record["finalScores"][int(record.get("ourPlayer", 0))]
        - record["finalScores"][1 - int(record.get("ourPlayer", 0))]
        for record in records
    ]
    wins = sum(
        1
        for record in records
        if record["winner"] == int(record.get("ourPlayer", 0))
    )
    draws = sum(1 for record in records if record["winner"] == "draw")
    losses = len(records) - wins - draws
    return {
        "games": len(records),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "winRate": wins / len(records),
        "averageScoreMargin": sum(margins) / len(margins),
        "strategic": summarize_records_by_perspective(records),
    }


def summarize_records_by_perspective(records: list[dict]) -> dict:
    from dots_boxes_mcts.strategic_eval import summarize_strategic_records

    return summarize_strategic_records(
        records,
        perspective_player=lambda record: int(record.get("ourPlayer", 0)),
    )


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

    def new_game(self, *, rows: int, cols: int, our_player: int = 0) -> PapgPage:
        if (rows, cols) not in PAPG_SIZE_VALUES:
            raise ValueError(f"Papg board size is not configured for {rows}x{cols} dots.")
        if our_player not in {0, 1}:
            raise ValueError("our_player must be 0 or 1")
        if our_player == 1:
            data = urlencode(
                {
                    "YOUR-NAME": "MCTS",
                    "SIZE": PAPG_SIZE_VALUES[(rows, cols)],
                    "HUMAN": "2",
                    "PLAY": "New Game",
                }
            ).encode()
            html = self._open(Request(PAPG_NEW_GAME_URL, data=data, headers=PAPG_HEADERS))
            return parse_papg_page(html, rows=rows, cols=cols)

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
        poll_url = papg_thinking_url(url)
        html = self._open(Request(url, headers=PAPG_HEADERS))
        parsed_page = parse_papg_page(html, rows=page.rows, cols=page.cols, poll_url=poll_url)

        attempts = 0
        while is_thinking_page(html) and not parsed_page.move_links and attempts < 5:
            html = self._open(Request(poll_url, headers=PAPG_HEADERS))
            parsed_page = parse_papg_page(html, rows=page.rows, cols=page.cols, poll_url=poll_url)
            attempts += 1

        return parsed_page

    def poll_reply(self, page: PapgPage) -> PapgPage:
        if page.poll_url is None:
            raise RuntimeError("Cannot poll Papg reply before submitting a local move.")
        html = self._open(Request(page.poll_url, headers=PAPG_HEADERS))
        return parse_papg_page(html, rows=page.rows, cols=page.cols, poll_url=page.poll_url)

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
        poll_url: str | None = None,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.move_links = move_links
        self.edge_owners = edge_owners
        self.drawn_edges = drawn_edges
        self.poll_url = poll_url


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


def parse_papg_page(html: str, *, rows: int, cols: int, poll_url: str | None = None) -> PapgPage:
    move_links = parse_move_links(html)
    edge_owners = parse_edge_owners(html, rows=rows, cols=cols)
    return PapgPage(
        rows=rows,
        cols=cols,
        move_links=move_links,
        edge_owners=edge_owners,
        drawn_edges=parse_drawn_edges(move_links=move_links, edge_owners=edge_owners, rows=rows, cols=cols),
        poll_url=poll_url,
    )


def is_thinking_page(html: str) -> bool:
    return "Thinking..." in html


def papg_thinking_url(move_url: str) -> str:
    parts = urlsplit(move_url)
    query_parts = parts.query.split("+")
    if len(query_parts) > 1:
        query_parts[1] = "2"
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "+".join(query_parts), parts.fragment))
    return move_url.replace("/dab?.+1+", "/dab?.+2+", 1)


def parse_drawn_edges(
    *,
    move_links: dict[int, str],
    edge_owners: dict[str, int],
    rows: int,
    cols: int,
) -> set[str]:
    if edge_owners:
        return set(edge_owners)

    if not move_links:
        return set()

    legal_indexes = set(move_links)
    return {
        edge
        for edge in legal_moves(new_game(rows=rows, cols=cols))
        if edge_to_papg_index(edge, rows=rows, cols=cols) not in legal_indexes
    }


def parse_move_links(html: str) -> dict[int, str]:
    links: dict[int, str] = {}
    pattern = r'href="?([^"\s>]*/dab\?[^"\s>]*?\+\d+\+\d+\+\d+\+\d+\+(\d+)\+[^"\s>]*)"?'
    for href, raw_index in re.findall(pattern, html):
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
    parser.add_argument("--disable-tree-reuse", action="store_true")
    parser.add_argument("--evaluator-cache-entries", type=int, default=DEFAULT_EVALUATOR_CACHE_ENTRIES)
    parser.add_argument("--our-player", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--alternate-players",
        action="store_true",
        help="Alternate local bot between player 0 and player 1. Requires an even --games value.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--request-delay", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/papg/stage-2.5/papg-eval.jsonl"))
    args = parser.parse_args()

    if args.games < 1:
        raise SystemExit("--games must be at least 1")
    if args.alternate_players and args.games % 2 != 0:
        raise SystemExit("--alternate-players requires an even --games value")
    if args.evaluator_cache_entries < 0:
        raise SystemExit("--evaluator-cache-entries must be non-negative")

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
            our_player=args.our_player,
            alternate_players=args.alternate_players,
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
            reuse_tree=not args.disable_tree_reuse,
            evaluator_cache_entries=args.evaluator_cache_entries,
            our_player=args.our_player,
            alternate_players=args.alternate_players,
        )
    write_jsonl(records, args.out)
    print(json.dumps(summarize_papg_records(records), sort_keys=True))
    print(f"Wrote {len(records)} games to {args.out}")


if __name__ == "__main__":
    main()
