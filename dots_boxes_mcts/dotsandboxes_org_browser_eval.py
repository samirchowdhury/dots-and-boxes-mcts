from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dots_boxes_mcts.browser_decision_server import bot_name, decision_response
from dots_boxes_mcts.external_bot_common import (
    external_game_record,
    replay_moves,
    summarize_external_records,
)
from dots_boxes_mcts.game import apply_move, new_game
from dots_boxes_mcts.self_play import write_jsonl

DEFAULT_CHROME_PATH = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
DEFAULT_BOT_DISPLAY_NAME = "EpsilonZero"
DOTSANDBOXES_ORG_URL = "https://dotsandboxes.org/"
DOTSANDBOXES_ORG_OPPONENT = "dotsandboxes.org"
DEFAULT_SITE_THINK_TIME = 0.25
BLOCKED_DOTSANDBOXES_ORG_PATTERNS = (
    "**/dotsAndBoxesLog.php",
    "**/dotsAndBoxesHighScores.php",
    "**://www.googletagmanager.com/**",
    "**://www.google-analytics.com/**",
    "**://platform.twitter.com/**",
    "**://connect.facebook.net/**",
)


@dataclass(frozen=True)
class RecordingOptions:
    video_path: Path | None = None
    width: int = 1280
    height: int = 900
    bot_display_name: str = DEFAULT_BOT_DISPLAY_NAME
    captions: bool = False
    scoring_pause: float = 1.25
    final_pause: float = 3.0


def generate_dotsandboxes_org_games(
    *,
    checkpoint: Path | None,
    games: int,
    rows: int = 4,
    cols: int = 4,
    simulations: int = 100,
    seed: int = 1,
    c_puct: float = 1.5,
    mlx_device: str = "cpu",
    backend: str = "numba",
    our_player: int = 0,
    alternate_players: bool = False,
    headless: bool = False,
    browser_executable: Path | None = DEFAULT_CHROME_PATH,
    slow_mo_ms: int = 0,
    site_url: str = DOTSANDBOXES_ORG_URL,
    site_think_time: float = DEFAULT_SITE_THINK_TIME,
    block_telemetry: bool = True,
    recording: RecordingOptions | None = None,
) -> list[dict]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise RuntimeError(
            "The dotsandboxes.org browser runner needs Playwright. Install it with: "
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
        if block_telemetry:
            block_nonessential_site_traffic(context)
        page = context.new_page()
        video = page.video if recording and recording.video_path is not None else None
        try:
            for game_index in range(games):
                game_our_player = game_index % 2 if alternate_players else our_player
                record = play_dotsandboxes_org_game(
                    page=page,
                    checkpoint=checkpoint,
                    rows=rows,
                    cols=cols,
                    simulations=simulations,
                    seed=seed + game_index,
                    c_puct=c_puct,
                    mlx_device=mlx_device,
                    backend=backend,
                    our_player=game_our_player,
                    site_url=site_url,
                    site_think_time=site_think_time,
                    block_telemetry=block_telemetry,
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


def block_nonessential_site_traffic(context) -> None:
    for pattern in BLOCKED_DOTSANDBOXES_ORG_PATTERNS:
        context.route(pattern, lambda route: route.abort())


def play_dotsandboxes_org_game(
    *,
    page,
    checkpoint: Path | None,
    rows: int,
    cols: int,
    simulations: int,
    seed: int,
    c_puct: float,
    mlx_device: str,
    backend: str,
    our_player: int,
    site_url: str = DOTSANDBOXES_ORG_URL,
    site_think_time: float = DEFAULT_SITE_THINK_TIME,
    block_telemetry: bool = True,
    recording: RecordingOptions | None = None,
) -> dict:
    if our_player not in {0, 1}:
        raise ValueError("our_player must be 0 or 1")
    start_dotsandboxes_org_game(
        page=page,
        rows=rows,
        cols=cols,
        our_player=our_player,
        site_url=site_url,
        site_think_time=site_think_time,
        bot_display_name=recording.bot_display_name if recording else DEFAULT_BOT_DISPLAY_NAME,
    )
    if recording and recording.captions:
        set_recording_caption(
            page,
            title=f"{recording.bot_display_name} vs dotsandboxes.org",
            detail="Browser game, client-side opponent engine",
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
        "backend": backend,
        "our_player": our_player,
        "opponent": DOTSANDBOXES_ORG_OPPONENT,
        "source": DOTSANDBOXES_ORG_OPPONENT,
    }

    max_turns = rows * (cols - 1) + cols * (rows - 1)
    for _turn in range(max_turns + 1):
        browser_state = wait_for_dotsandboxes_org_turn(
            page=page,
            rows=rows,
            cols=cols,
            our_player=our_player,
            timeout_seconds=max(30.0, site_think_time + 10.0),
        )
        moves = list(browser_state["moves"])
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
                "drawn_edges": browser_state["drawn"],
            }
        )
        moves = list(response["moves"])
        if response.get("terminal"):
            break

        decisions.append(response["decision"])
        move = response["move"]
        if move in browser_state["drawn"]:
            raise RuntimeError(f"Cannot play already drawn edge {move}.")

        if recording and recording.captions:
            set_recording_caption(
                page,
                title=f"{recording.bot_display_name} chooses {move}",
                detail=recording_move_detail(response["decision"]),
                kind="move",
            )

        play_dotsandboxes_org_move(page, move)
        if recording and recording.captions:
            maybe_pause_after_scoring_move(
                rows=rows,
                cols=cols,
                moves=moves,
                move=move,
                recording=recording,
            )
    else:
        raise RuntimeError("dotsandboxes.org browser game exceeded its legal move count.")

    browser_state = read_dotsandboxes_org_state(page=page, rows=rows, cols=cols)
    moves = list(browser_state["moves"])
    bot = bot_name(checkpoint=checkpoint_value, simulations=simulations)
    record = external_game_record(
        opponent=DOTSANDBOXES_ORG_OPPONENT,
        bot=bot,
        rows=rows,
        cols=cols,
        moves=moves,
        our_player=our_player,
        source=DOTSANDBOXES_ORG_OPPONENT,
        notes=(
            "Browser-backed dotsandboxes.org game with network-guided MCTS."
            if checkpoint is not None
            else "Browser-backed dotsandboxes.org game with UCT MCTS."
        ),
    )
    record["seed"] = seed
    record["simulations"] = simulations
    record["decisions"] = decisions
    record["siteUrl"] = site_url
    record["siteThinkTime"] = site_think_time
    record["siteTelemetryBlocked"] = block_telemetry
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
            title=(
                f"Final: {recording.bot_display_name} {final_scores[our_player]} - "
                f"dotsandboxes.org {final_scores[1 - our_player]}"
            ),
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


def start_dotsandboxes_org_game(
    *,
    page,
    rows: int,
    cols: int,
    our_player: int,
    site_url: str,
    site_think_time: float,
    bot_display_name: str,
) -> dict:
    validate_dotsandboxes_org_board_size(rows=rows, cols=cols)
    page.goto(site_url, wait_until="domcontentloaded")
    page.wait_for_function(
        "() => window.manager && window.graphics && document.getElementById('gameCode')",
        timeout=15000,
    )
    page.evaluate(
        """
        ({ rows, cols, ourPlayer, siteThinkTime, botDisplayName }) => {
          document.getElementById("numDotsX").value = String(cols);
          document.getElementById("numDotsY").value = String(rows);
          document.getElementById("thinkTime").value = String(siteThinkTime);
          document.getElementById("showAnimations").checked = false;
          document.getElementById("freeTurns").checked = false;
          document.getElementById("player1Name").value =
            ourPlayer === 0 ? botDisplayName : "dotsandboxes.org";
          document.getElementById("player2Name").value =
            ourPlayer === 1 ? botDisplayName : "dotsandboxes.org";
          document.getElementById("player1IsComputer").checked = ourPlayer !== 0;
          document.getElementById("player2IsComputer").checked = ourPlayer !== 1;
          document.getElementById("player1PlaysFirst").checked = true;
          document.getElementById("player2PlaysFirst").checked = false;
          window.manager.newGame("browser-eval");
        }
        """,
        {
            "rows": rows,
            "cols": cols,
            "ourPlayer": our_player,
            "siteThinkTime": site_think_time,
            "botDisplayName": bot_display_name,
        },
    )
    return wait_for_dotsandboxes_org_turn(
        page=page,
        rows=rows,
        cols=cols,
        our_player=our_player,
        timeout_seconds=max(30.0, site_think_time + 10.0),
    )


def validate_dotsandboxes_org_board_size(*, rows: int, cols: int) -> None:
    if not 2 <= rows <= 10 or not 2 <= cols <= 10:
        raise ValueError("dotsandboxes.org supports 2 to 10 dots in each dimension.")


def play_dotsandboxes_org_move(page, edge_id: str) -> None:
    x1, y1, x2, y2 = edge_to_dotsandboxes_org_coords(edge_id)
    page.evaluate(
        """
        ({ x1, y1, x2, y2 }) => {
          if (!window.manager.isLegal(x1, y1, x2, y2)) {
            throw new Error(`Illegal dotsandboxes.org move: ${x1},${y1},${x2},${y2}`);
          }
          window.manager.doMove(x1, y1, x2, y2, false);
        }
        """,
        {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    )


def wait_for_dotsandboxes_org_turn(
    *,
    page,
    rows: int,
    cols: int,
    our_player: int,
    timeout_seconds: float = 30.0,
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last_state = read_dotsandboxes_org_state(page=page, rows=rows, cols=cols)
        if not last_state.get("ready"):
            time.sleep(0.05)
            continue
        try:
            state = replay_moves(rows=rows, cols=cols, moves=last_state["moves"])
        except Exception as error:
            last_state["error"] = str(error)
            time.sleep(0.05)
            continue
        last_state["currentPlayer"] = state.current_player
        last_state["scores"] = list(state.scores)
        last_state["terminal"] = state.terminal
        if state.terminal or state.current_player == our_player:
            return last_state
        time.sleep(0.05)
    raise RuntimeError(f"Timed out waiting for dotsandboxes.org turn; last state: {last_state}")


def read_dotsandboxes_org_state(*, page, rows: int, cols: int) -> dict:
    raw_state = page.evaluate(
        """
        () => {
          const gameCodeNode = document.getElementById("gameCode");
          const gameCode = gameCodeNode ? gameCodeNode.value : "";
          let parsed = null;
          let parseError = null;
          try {
            parsed = gameCode ? JSON.parse(gameCode) : null;
          } catch (error) {
            parseError = String(error);
          }
          return {
            ready: Boolean(parsed),
            gameCode,
            parseError,
            parsed,
            hLines: window.manager && window.manager.getHLines ? window.manager.getHLines() : [],
            vLines: window.manager && window.manager.getVLines ? window.manager.getVLines() : [],
            text: document.body ? document.body.innerText : "",
          };
        }
        """
    )
    if not raw_state.get("ready"):
        raw_state["moves"] = []
        raw_state["drawn"] = []
        return raw_state

    parsed = raw_state["parsed"]
    parsed_cols = int(parsed.get("NX", cols))
    parsed_rows = int(parsed.get("NY", rows))
    if parsed_rows != rows or parsed_cols != cols:
        raise RuntimeError(
            f"dotsandboxes.org board is {parsed_rows}x{parsed_cols}, expected {rows}x{cols}."
        )
    moves = [
        dotsandboxes_org_edge_number_to_edge(int(turn["E"]), rows=rows, cols=cols)
        for turn in parsed.get("M", [])
    ]
    drawn = drawn_edges_from_dotsandboxes_org_lines(
        h_lines=raw_state["hLines"],
        v_lines=raw_state["vLines"],
        rows=rows,
        cols=cols,
    )
    raw_state["moves"] = moves
    raw_state["drawn"] = sorted(drawn)
    raw_state["players"] = list(parsed.get("P", []))
    return raw_state


def drawn_edges_from_dotsandboxes_org_lines(
    *,
    h_lines: list[list[int]],
    v_lines: list[list[int]],
    rows: int,
    cols: int,
) -> set[str]:
    drawn: set[str] = set()
    for row, line_row in enumerate(h_lines[:rows]):
        for col, value in enumerate(line_row[: cols - 1]):
            if value:
                drawn.add(f"h:{row}:{col}")
    for col, line_col in enumerate(v_lines[:cols]):
        for row, value in enumerate(line_col[: rows - 1]):
            if value:
                drawn.add(f"v:{row}:{col}")
    return drawn


def dotsandboxes_org_edge_number_to_edge(edge_number: int, *, rows: int, cols: int) -> str:
    horizontal_count = (cols - 1) * rows
    total_edges = horizontal_count + cols * (rows - 1)
    if edge_number < 0 or edge_number >= total_edges:
        raise ValueError(f"dotsandboxes.org edge number is out of range: {edge_number}")
    if edge_number >= horizontal_count:
        offset = edge_number - horizontal_count
        row = offset % (rows - 1)
        col = (offset - row) // (rows - 1)
        return f"v:{row}:{col}"
    col = edge_number % (cols - 1)
    row = (edge_number - col) // (cols - 1)
    return f"h:{row}:{col}"


def edge_to_dotsandboxes_org_edge_number(edge_id: str, *, rows: int, cols: int) -> int:
    kind, raw_row, raw_col = edge_id.split(":")
    row = int(raw_row)
    col = int(raw_col)
    if kind == "h":
        if not 0 <= row < rows or not 0 <= col < cols - 1:
            raise ValueError(f"Invalid horizontal edge for {rows}x{cols}: {edge_id}")
        return col + row * (cols - 1)
    if kind == "v":
        if not 0 <= row < rows - 1 or not 0 <= col < cols:
            raise ValueError(f"Invalid vertical edge for {rows}x{cols}: {edge_id}")
        return (cols - 1) * rows + row + col * (rows - 1)
    raise ValueError(f"Invalid edge id: {edge_id}")


def edge_to_dotsandboxes_org_coords(edge_id: str) -> tuple[int, int, int, int]:
    kind, raw_row, raw_col = edge_id.split(":")
    row = int(raw_row)
    col = int(raw_col)
    if kind == "h":
        return col, row, col + 1, row
    if kind == "v":
        return col, row, col, row + 1
    raise ValueError(f"Invalid edge id: {edge_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a local searcher against dotsandboxes.org through Chrome."
    )
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--simulations", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Optional MLX checkpoint for network-guided MCTS.")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--mlx-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--backend", choices=["python", "numba"], default="numba")
    parser.add_argument("--our-player", type=int, choices=[0, 1], default=0)
    parser.add_argument("--alternate-players", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--browser-executable", type=Path, default=DEFAULT_CHROME_PATH)
    parser.add_argument("--slow-mo-ms", type=int, default=0)
    parser.add_argument("--site-url", default=DOTSANDBOXES_ORG_URL)
    parser.add_argument("--site-think-time", type=float, default=DEFAULT_SITE_THINK_TIME)
    parser.add_argument(
        "--allow-site-telemetry",
        action="store_true",
        help="Allow dotsandboxes.org log/high-score/analytics requests instead of blocking them.",
    )
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
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/dotsandboxes-org/stage-2/dotsandboxes-org-browser-eval.jsonl"),
    )
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
    if args.site_think_time < 0:
        raise SystemExit("--site-think-time must be non-negative")
    validate_dotsandboxes_org_board_size(rows=args.rows, cols=args.cols)

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

    records = generate_dotsandboxes_org_games(
        checkpoint=args.checkpoint,
        games=args.games,
        rows=args.rows,
        cols=args.cols,
        simulations=args.simulations,
        seed=args.seed,
        c_puct=args.c_puct,
        mlx_device=args.mlx_device,
        backend=args.backend,
        our_player=args.our_player,
        alternate_players=args.alternate_players,
        headless=args.headless,
        browser_executable=args.browser_executable,
        slow_mo_ms=args.slow_mo_ms,
        site_url=args.site_url,
        site_think_time=args.site_think_time,
        block_telemetry=not args.allow_site_telemetry,
        recording=recording,
    )
    write_jsonl(records, args.out)
    print(json.dumps(summarize_external_records(records), sort_keys=True))
    print(f"Wrote {len(records)} games to {args.out}")
    if args.record_video is not None:
        print(f"Recorded browser video to {args.record_video}")


if __name__ == "__main__":
    main()
