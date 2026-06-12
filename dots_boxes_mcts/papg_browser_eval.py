from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from dots_boxes_mcts.external_games import external_game_record
from dots_boxes_mcts.papg_decision_server import bot_name, decision_response
from dots_boxes_mcts.papg_common import PAPG_NEW_GAME_URL, summarize_papg_records
from dots_boxes_mcts.self_play import write_jsonl

DEFAULT_CHROME_PATH = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
PAPG_SIZE_VALUES = {
    (3, 3): "0",
    (4, 4): "1",
    (5, 4): "2",
    (6, 4): "3",
}


def generate_browser_papg_games(
    *,
    checkpoint: Path | None,
    games: int,
    rows: int = 4,
    cols: int = 4,
    simulations: int = 100,
    seed: int = 1,
    c_puct: float = 1.5,
    mlx_device: str = "cpu",
    request_delay: float = 5.0,
    our_player: int = 0,
    alternate_players: bool = False,
    headless: bool = False,
    browser_executable: Path | None = DEFAULT_CHROME_PATH,
    slow_mo_ms: int = 0,
) -> list[dict]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise RuntimeError(
            "The browser PAPG runner needs Playwright. Install it with: "
            "pyenv activate data && pip install playwright"
        ) from error

    records: list[dict] = []
    with sync_playwright() as playwright:
        launch_kwargs: dict[str, Any] = {
            "headless": headless,
            "slow_mo": slow_mo_ms,
        }
        if browser_executable is not None:
            launch_kwargs["executable_path"] = str(browser_executable)
        browser = playwright.chromium.launch(**launch_kwargs)
        page = browser.new_page()
        try:
            for game_index in range(games):
                game_our_player = game_index % 2 if alternate_players else our_player
                record = play_browser_papg_game(
                    page=page,
                    checkpoint=checkpoint,
                    rows=rows,
                    cols=cols,
                    simulations=simulations,
                    seed=seed + game_index,
                    c_puct=c_puct,
                    mlx_device=mlx_device,
                    request_delay=request_delay,
                    our_player=game_our_player,
                )
                record["gameIndex"] = game_index
                records.append(record)
        finally:
            browser.close()
    return records


def play_browser_papg_game(
    *,
    page,
    checkpoint: Path | None,
    rows: int,
    cols: int,
    simulations: int,
    seed: int,
    c_puct: float,
    mlx_device: str,
    request_delay: float,
    our_player: int,
) -> dict:
    if our_player not in {0, 1}:
        raise ValueError("our_player must be 0 or 1")
    start_papg_game(page=page, rows=rows, cols=cols, our_player=our_player)

    moves: list[str] = []
    decisions: list[dict] = []
    checkpoint_value = str(checkpoint) if checkpoint is not None else None
    base_payload = {
        "rows": rows,
        "cols": cols,
        "simulations": simulations,
        "seed": seed,
        "checkpoint": checkpoint_value,
        "cPuct": c_puct,
        "mlxDevice": mlx_device,
        "our_player": our_player,
    }

    for _turn in range(120):
        board = read_board_info(page)
        response = decision_response(
            {
                **base_payload,
                "moves": moves,
                "drawn_edges": board["drawn"],
            }
        )
        moves = list(response["moves"])
        if response.get("terminal"):
            break

        decisions.append(response["decision"])
        fresh_board = read_board_info(page)
        move = response["move"]
        href = fresh_board["links"].get(move)
        if href is None or move in fresh_board["drawn"]:
            raise RuntimeError(
                f"Cannot click {move}; drawn={fresh_board['drawn']} "
                f"legal={sorted(fresh_board['links'])}"
            )

        time.sleep(request_delay)
        click_papg_move(page, href)
        moves.append(move)
        wait_for_papg_ready(page)
    else:
        raise RuntimeError("Browser PAPG game exceeded 120 decision turns.")

    board = read_board_info(page)
    final_response = decision_response(
        {
            **base_payload,
            "moves": moves,
            "drawn_edges": board["drawn"],
        }
    )
    moves = list(final_response["moves"])
    bot = bot_name(checkpoint=checkpoint_value, simulations=simulations)
    record = external_game_record(
        source="papg",
        opponent="papg",
        bot=bot,
        rows=rows,
        cols=cols,
        moves=moves,
        our_player=our_player,
        notes=(
            "Browser-backed Papg game with network-guided MCTS."
            if checkpoint is not None
            else "Browser-backed Papg game with UCT MCTS."
        ),
    )
    record["seed"] = seed
    record["simulations"] = simulations
    record["decisions"] = decisions
    if checkpoint is not None:
        record["checkpoint"] = checkpoint_value
        record["cPuct"] = c_puct
        record["mlxDevice"] = mlx_device
    return record


def start_papg_game(*, page, rows: int, cols: int, our_player: int) -> None:
    try:
        size_value = PAPG_SIZE_VALUES[(rows, cols)]
    except KeyError as error:
        raise ValueError(f"Papg board size is not configured for {rows}x{cols} dots.") from error

    page.goto(PAPG_NEW_GAME_URL, wait_until="load")
    page.locator(f'input[name="SIZE"][value="{size_value}"]').click()
    page.locator(f'input[name="HUMAN"][value="{our_player + 1}"]').click()
    page.get_by_role("button", name="New Game", exact=True).click()
    page.wait_for_load_state("load")
    wait_for_papg_ready(page)

    board = read_board_info(page)
    if our_player == 0 and board["drawn"]:
        raise RuntimeError(f"Expected an empty board for player 0, saw {board['drawn']}")
    if our_player == 1 and not board["drawn"]:
        raise RuntimeError("Expected Papg to open the game for player 1, saw an empty board.")


def click_papg_move(page, href: str) -> None:
    escaped_href = href.replace('"', '\\"')
    selector = f'a[href="{escaped_href}"]'
    link = page.locator(selector)
    count = link.count()
    if count != 1:
        raise RuntimeError(f"Expected one Papg move link for {href}, saw {count}.")
    link.click(timeout=5000)
    page.wait_for_load_state("load", timeout=15000)


def wait_for_papg_ready(page) -> str:
    last_text = ""
    for _attempt in range(45):
        try:
            status = page.evaluate(
                """
                () => {
                  const text = document.body.innerText;
                  const moveLinks = document.querySelectorAll('table a[href^="/dab?"]').length;
                  const edgeImages = [...document.querySelectorAll('table img[src*="dab_"]')].filter((image) => {
                    const src = image.getAttribute("src") || "";
                    return /dab_[HV][BR]\\.gif/.test(src);
                  }).length;
                  return { text, moveLinks, edgeImages };
                }
                """
            )
        except Exception:
            time.sleep(1)
            continue
        last_text = status["text"]
        if (
            "Thinking..." not in last_text
            and (
                status["moveLinks"] > 0
                or status["edgeImages"] == 24
                or "Game finished" in last_text
            )
        ):
            return last_text
        time.sleep(1)
    return last_text


def read_board_info(page) -> dict:
    return page.evaluate(
        """
        () => {
          const table = document.querySelector("table");
          const info = { drawn: [], links: {}, text: document.body.innerText };
          if (!table) return info;

          [...table.querySelectorAll("tr")].forEach((tr, tableRow) => {
            [...tr.querySelectorAll("td")].forEach((td, tableCol) => {
              let edge = null;
              if (tableRow % 2 === 0 && tableCol % 2 === 1) {
                edge = `h:${Math.floor(tableRow / 2)}:${Math.floor((tableCol - 1) / 2)}`;
              }
              if (tableRow % 2 === 1 && tableCol % 2 === 0) {
                edge = `v:${Math.floor((tableRow - 1) / 2)}:${Math.floor(tableCol / 2)}`;
              }
              if (!edge) return;

              const image = td.querySelector("img");
              const src = image ? image.getAttribute("src") || "" : "";
              if (/dab_[HV][BR]\\.gif/.test(src)) {
                info.drawn.push(edge);
              }

              const link = td.querySelector('a[href^="/dab?"]');
              if (link) {
                info.links[edge] = link.getAttribute("href");
              }
            });
          });
          return info;
        }
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a local searcher against live PAPG through Chrome.")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--simulations", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Optional MLX checkpoint for network-guided MCTS.")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--mlx-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--our-player", type=int, choices=[0, 1], default=0)
    parser.add_argument("--alternate-players", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--request-delay", type=float, default=5.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--browser-executable", type=Path, default=DEFAULT_CHROME_PATH)
    parser.add_argument("--slow-mo-ms", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/papg/stage-4/papg-browser-eval.jsonl"))
    args = parser.parse_args()

    if args.games < 1:
        raise SystemExit("--games must be at least 1")
    if args.alternate_players and args.games % 2 != 0:
        raise SystemExit("--alternate-players requires an even --games value")
    if args.request_delay < 1.0:
        raise SystemExit("--request-delay must be at least 1 second")
    if args.browser_executable is not None and not args.browser_executable.exists():
        raise SystemExit(f"Browser executable does not exist: {args.browser_executable}")

    records = generate_browser_papg_games(
        checkpoint=args.checkpoint,
        games=args.games,
        rows=args.rows,
        cols=args.cols,
        simulations=args.simulations,
        seed=args.seed,
        c_puct=args.c_puct,
        mlx_device=args.mlx_device,
        request_delay=args.request_delay,
        our_player=args.our_player,
        alternate_players=args.alternate_players,
        headless=args.headless,
        browser_executable=args.browser_executable,
        slow_mo_ms=args.slow_mo_ms,
    )
    write_jsonl(records, args.out)
    print(json.dumps(summarize_papg_records(records), sort_keys=True))
    print(f"Wrote {len(records)} games to {args.out}")


if __name__ == "__main__":
    main()
