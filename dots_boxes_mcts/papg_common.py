from __future__ import annotations

import re
from pathlib import Path

from dots_boxes_mcts.game import GameState, apply_move, legal_moves, new_game, state_snapshot

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


def checkpoint_bot_name(*, checkpoint: Path, simulations: int) -> str:
    return f"network_guided_mcts_{simulations}_{checkpoint.stem}"


def papg_index_to_edge(index: int, rows: int, cols: int) -> str:
    if rows < 2 or cols < 2:
        raise ValueError("Papg index conversion needs at least a 2x2 dot grid.")
    if index < 0:
        raise ValueError(f"Unknown Papg move index: {index}")

    table_width = cols * 2 - 1
    table_height = rows * 2 - 1
    table_row, table_col = divmod(index, table_width)

    if table_row >= table_height:
        raise ValueError(f"Unknown Papg {rows}x{cols} move index: {index}")

    if table_row % 2 == 0 and table_col % 2 == 1:
        return f"h:{table_row // 2}:{(table_col - 1) // 2}"

    if table_row % 2 == 1 and table_col % 2 == 0:
        return f"v:{(table_row - 1) // 2}:{table_col // 2}"

    raise ValueError(f"Papg index {index} is a dot or box cell, not an edge.")


def papg_indexes_to_edges(indexes: list[int], rows: int = 3, cols: int = 3) -> list[str]:
    return [papg_index_to_edge(index, rows=rows, cols=cols) for index in indexes]


def edge_to_papg_index(edge_id: str, rows: int, cols: int) -> int:
    kind, raw_row, raw_col = edge_id.split(":")
    row = int(raw_row)
    col = int(raw_col)
    table_width = cols * 2 - 1

    if kind == "h":
        return (row * 2) * table_width + col * 2 + 1
    if kind == "v":
        return (row * 2 + 1) * table_width + col * 2

    raise ValueError(f"Invalid edge id: {edge_id}")


def papg_game_record(
    *,
    opponent: str,
    bot: str,
    rows: int,
    cols: int,
    moves: list[str],
    our_player: int = 0,
    source: str = "papg",
    notes: str | None = None,
) -> dict:
    if our_player not in {0, 1}:
        raise ValueError("our_player must be 0 or 1.")

    state = replay_moves(rows=rows, cols=cols, moves=moves)
    opponent_player = 1 if our_player == 0 else 0
    record = {
        "source": source,
        "opponent": opponent,
        "bot": bot,
        "rows": state.rows,
        "cols": state.cols,
        "players": {
            str(our_player): bot,
            str(opponent_player): opponent,
        },
        "ourPlayer": our_player,
        "moves": moves,
        "finalScores": [state.scores[0], state.scores[1]],
        "winner": state.winner,
        "terminal": state.terminal,
        "state": state_snapshot(state),
    }
    if notes:
        record["notes"] = notes
    return record


def replay_moves(*, rows: int, cols: int, moves: list[str]) -> GameState:
    state = new_game(rows=rows, cols=cols)
    for move in moves:
        state = apply_move(state, move)
    return state


def sync_papg_moves(state: GameState, page: PapgPage, *, papg_player: int = 1) -> list[str]:
    missing_papg_moves = [
        edge
        for edge in page.drawn_edges
        if edge not in state.edges
    ]
    return infer_papg_reply(state, missing_papg_moves, papg_player=papg_player)


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
