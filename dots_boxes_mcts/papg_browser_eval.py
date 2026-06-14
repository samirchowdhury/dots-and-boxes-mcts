from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dots_boxes_mcts.external_games import external_game_record
from dots_boxes_mcts.game import apply_move, new_game
from dots_boxes_mcts.papg_decision_server import bot_name, decision_response
from dots_boxes_mcts.papg_common import PAPG_NEW_GAME_URL, summarize_papg_records
from dots_boxes_mcts.self_play import write_jsonl

DEFAULT_CHROME_PATH = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
DEFAULT_BOT_DISPLAY_NAME = "EpsilonZero"
PAPG_SIZE_VALUES = {
    (3, 3): "0",
    (4, 4): "1",
    (5, 4): "2",
    (6, 4): "3",
}


@dataclass(frozen=True)
class RecordingOptions:
    video_path: Path | None = None
    width: int = 1280
    height: int = 900
    bot_display_name: str = DEFAULT_BOT_DISPLAY_NAME
    captions: bool = False
    scoring_pause: float = 1.25
    final_pause: float = 3.0


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
    our_player: int = 0,
    alternate_players: bool = False,
    headless: bool = False,
    browser_executable: Path | None = DEFAULT_CHROME_PATH,
    slow_mo_ms: int = 0,
    recording: RecordingOptions | None = None,
) -> list[dict]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise RuntimeError(
            "The browser PAPG runner needs Playwright. Install it with: "
            "`uv sync`."
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
        context_kwargs: dict[str, Any] = {}
        if recording and recording.video_path is not None:
            recording.video_path.parent.mkdir(parents=True, exist_ok=True)
            context_kwargs["record_video_dir"] = str(recording.video_path.parent)
            context_kwargs["record_video_size"] = {
                "width": recording.width,
                "height": recording.height,
            }
        if recording:
            context_kwargs["viewport"] = {"width": recording.width, "height": recording.height}
        context = browser.new_context(**context_kwargs)
        page = context.new_page()
        video = page.video if recording and recording.video_path is not None else None
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
                    our_player=game_our_player,
                    recording=recording,
                )
                record["gameIndex"] = game_index
                records.append(record)
        finally:
            context.close()
            if video is not None and recording is not None and recording.video_path is not None:
                raw_video_path = Path(video.path())
                video.save_as(recording.video_path)
                if raw_video_path != recording.video_path and raw_video_path.exists():
                    raw_video_path.unlink()
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
    our_player: int,
    recording: RecordingOptions | None = None,
) -> dict:
    if our_player not in {0, 1}:
        raise ValueError("our_player must be 0 or 1")
    start_papg_game(page=page, rows=rows, cols=cols, our_player=our_player)
    if recording and recording.captions:
        set_recording_caption(
            page,
            title=f"{recording.bot_display_name} vs PAPG",
            detail="Live browser game, network-guided MCTS",
            kind="ready",
        )

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
        if recording and recording.captions:
            set_recording_caption(
                page,
                title=f"{recording.bot_display_name} is thinking...",
                detail=f"{simulations} simulations on {mlx_device.upper()} | move {len(moves) + 1}",
                kind="thinking",
            )
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

        if recording and recording.captions:
            set_recording_caption(
                page,
                title=f"{recording.bot_display_name} chooses {move}",
                detail=recording_move_detail(response["decision"]),
                kind="move",
            )

        click_papg_move(page, href)
        if recording and recording.captions:
            maybe_pause_after_scoring_move(
                rows=rows,
                cols=cols,
                moves=moves,
                move=move,
                recording=recording,
            )
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
    if recording:
        record["botDisplayName"] = recording.bot_display_name
    if recording and recording.captions:
        final_scores = record["finalScores"]
        set_recording_caption(
            page,
            title=f"Final: {recording.bot_display_name} {final_scores[our_player]} - PAPG {final_scores[1 - our_player]}",
            detail=f"{recording.bot_display_name} wins" if record["winner"] == our_player else "Game finished",
            kind="final",
        )
        time.sleep(recording.final_pause)
    return record


def recording_move_detail(decision: dict) -> str:
    search = decision.get("search", {})
    stats = search.get("stats", [])
    if not stats:
        return "Network-guided MCTS move selected"
    top = stats[0]
    visits = top.get("visits", "?")
    mean_value = top.get("meanValue")
    if isinstance(mean_value, int | float):
        return f"Top visit count: {visits} | value {mean_value:+.3f}"
    return f"Top visit count: {visits}"


def maybe_pause_after_scoring_move(
    *,
    rows: int,
    cols: int,
    moves: list[str],
    move: str,
    recording: RecordingOptions,
) -> None:
    if move_scores_box(rows=rows, cols=cols, moves=moves, move=move):
        time.sleep(recording.scoring_pause)


def move_scores_box(*, rows: int, cols: int, moves: list[str], move: str) -> bool:
    state = new_game(rows=rows, cols=cols)
    for previous_move in moves:
        state = apply_move(state, previous_move)
    next_state = apply_move(state, move)
    return bool(next_state.history[-1]["scored"])


def set_recording_caption(page, *, title: str, detail: str, kind: str) -> None:
    page.evaluate(
        """
        ({ title, detail, kind }) => {
          const styleId = "ez-recording-caption-style";
          if (!document.getElementById(styleId)) {
            const style = document.createElement("style");
            style.id = styleId;
            style.textContent = `
              #ez-recording-caption {
                position: fixed;
                left: 24px;
                right: 24px;
                bottom: 24px;
                z-index: 2147483647;
                display: grid;
                gap: 6px;
                max-width: 780px;
                padding: 18px 20px;
                border: 1px solid rgba(255,255,255,0.28);
                border-radius: 14px;
                background: rgba(16, 24, 30, 0.88);
                color: white;
                box-shadow: 0 18px 45px rgba(0,0,0,0.26);
                font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                pointer-events: none;
              }
              #ez-recording-caption::before {
                content: "";
                position: absolute;
                width: 12px;
                height: 12px;
                left: 20px;
                top: 24px;
                border-radius: 999px;
                background: #2f9bff;
              }
              #ez-recording-caption[data-kind="thinking"]::before {
                background: #f2c94c;
                animation: ezPulse 1.1s ease-in-out infinite;
              }
              #ez-recording-caption[data-kind="final"]::before {
                background: #2dd4bf;
              }
              #ez-recording-caption strong {
                display: block;
                padding-left: 24px;
                font-size: 26px;
                line-height: 1.18;
                letter-spacing: 0;
              }
              #ez-recording-caption span {
                display: block;
                padding-left: 24px;
                color: rgba(255,255,255,0.78);
                font-size: 17px;
                line-height: 1.35;
                letter-spacing: 0;
              }
              @keyframes ezPulse {
                0%, 100% { transform: scale(0.78); opacity: 0.55; }
                50% { transform: scale(1.2); opacity: 1; }
              }
            `;
            document.head.append(style);
          }

          let caption = document.getElementById("ez-recording-caption");
          if (!caption) {
            caption = document.createElement("div");
            caption.id = "ez-recording-caption";
            caption.innerHTML = "<strong></strong><span></span>";
            document.body.append(caption);
          }
          caption.dataset.kind = kind;
          caption.querySelector("strong").textContent = title;
          caption.querySelector("span").textContent = detail;
        }
        """,
        {"title": title, "detail": detail, "kind": kind},
    )


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
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--browser-executable", type=Path, default=DEFAULT_CHROME_PATH)
    parser.add_argument("--slow-mo-ms", type=int, default=0)
    parser.add_argument(
        "--record-video",
        type=Path,
        help="Record the browser page to this WebM path using Playwright video capture.",
    )
    parser.add_argument("--record-video-width", type=int, default=1280)
    parser.add_argument("--record-video-height", type=int, default=900)
    parser.add_argument(
        "--record-captions",
        action="store_true",
        help="Show in-page recording captions even without --record-video.",
    )
    parser.add_argument(
        "--bot-display-name",
        default=DEFAULT_BOT_DISPLAY_NAME,
        help="Human-friendly bot name for recording captions.",
    )
    parser.add_argument("--record-scoring-pause", type=float, default=1.25)
    parser.add_argument("--record-final-pause", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=Path("runs/papg/stage-4/papg-browser-eval.jsonl"))
    args = parser.parse_args()

    if args.games < 1:
        raise SystemExit("--games must be at least 1")
    if args.alternate_players and args.games % 2 != 0:
        raise SystemExit("--alternate-players requires an even --games value")
    if args.browser_executable is not None and not args.browser_executable.exists():
        raise SystemExit(f"Browser executable does not exist: {args.browser_executable}")
    if args.record_video is not None and args.record_video.suffix.lower() != ".webm":
        raise SystemExit("--record-video currently writes Playwright WebM files; use a .webm path")
    if args.record_video_width < 320 or args.record_video_height < 240:
        raise SystemExit("--record-video-width/height are too small")
    if args.record_scoring_pause < 0 or args.record_final_pause < 0:
        raise SystemExit("--record-scoring-pause and --record-final-pause must be non-negative")

    recording = None
    if args.record_video is not None or args.record_captions:
        recording = RecordingOptions(
            video_path=args.record_video,
            width=args.record_video_width,
            height=args.record_video_height,
            bot_display_name=args.bot_display_name,
            captions=True,
            scoring_pause=args.record_scoring_pause,
            final_pause=args.record_final_pause,
        )

    records = generate_browser_papg_games(
        checkpoint=args.checkpoint,
        games=args.games,
        rows=args.rows,
        cols=args.cols,
        simulations=args.simulations,
        seed=args.seed,
        c_puct=args.c_puct,
        mlx_device=args.mlx_device,
        our_player=args.our_player,
        alternate_players=args.alternate_players,
        headless=args.headless,
        browser_executable=args.browser_executable,
        slow_mo_ms=args.slow_mo_ms,
        recording=recording,
    )
    write_jsonl(records, args.out)
    print(json.dumps(summarize_papg_records(records), sort_keys=True))
    print(f"Wrote {len(records)} games to {args.out}")
    if args.record_video is not None:
        print(f"Recorded browser video to {args.record_video}")


if __name__ == "__main__":
    main()
