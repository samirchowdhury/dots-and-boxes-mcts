import { spawnSync } from "node:child_process";

const REPO_ROOT = "/Users/samirchowdhury/dots-and-boxes-mcts";
const DEFAULT_OUT_DIR = `${REPO_ROOT}/runs/papg/stage-2.5`;

export async function runPapgBrowserBatch({
  browser,
  rows = 4,
  cols = 4,
  simulationsList = [10, 50, 100],
  games = 1,
  seed = 1,
  requestDelayMs = 5000,
  outDir = DEFAULT_OUT_DIR,
  onProgress = console.log,
} = {}) {
  if (!browser) {
    throw new Error("Pass the Codex Browser object as { browser }.");
  }
  if (requestDelayMs < 1000) {
    throw new Error("requestDelayMs must be at least 1000 to avoid hitting Papg too quickly.");
  }

  const summaries = [];
  const tab = await browser.tabs.new();
  for (const simulations of simulationsList) {
    const records = [];
    for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
      const gameSeed = seed + gameIndex;
      const out = `${outDir}/mcts-${simulations}-vs-papg-${rows}x${cols}.jsonl`;
      const record = await playOnePapgBrowserGame({
        browser,
        tab,
        rows,
        cols,
        simulations,
        seed: gameSeed,
        requestDelayMs,
        out,
        onProgress: (message) => onProgress(`[${simulations} sim game ${gameIndex + 1}/${games}] ${message}`),
      });
      records.push(record);
    }
    summaries.push(summarize(records, simulations));
  }
  return summaries;
}

export async function playOnePapgBrowserGame({
  browser,
  tab = null,
  rows = 4,
  cols = 4,
  simulations = 100,
  seed = 1,
  requestDelayMs = 5000,
  out = `${DEFAULT_OUT_DIR}/mcts-${simulations}-vs-papg-${rows}x${cols}.jsonl`,
  onProgress = console.log,
} = {}) {
  const gameTab = tab ?? await browser.tabs.new();
  await startGame(gameTab, rows, cols);

  let moves = [];
  let finalRecord = null;
  const basePayload = { rows, cols, simulations, seed, out };

  for (let turn = 0; turn < 120; turn += 1) {
    const info = await readBoardInfo(gameTab);
    const decision = runPythonDecision({
      ...basePayload,
      moves,
      drawn_edges: info.drawn,
    });
    moves = decision.moves;

    if (decision.terminal) {
      finalRecord = runPythonDecision({
        ...basePayload,
        moves,
        drawn_edges: info.drawn,
        write_record: true,
      });
      break;
    }

    const freshInfo = await readBoardInfo(gameTab);
    const href = freshInfo.links[decision.move];
    if (!href || freshInfo.drawn.includes(decision.move)) {
      throw new Error(
        `Cannot click ${decision.move}; drawn=${JSON.stringify(freshInfo.drawn)} legal=${JSON.stringify(Object.keys(freshInfo.links))}`,
      );
    }

    await sleep(requestDelayMs);
    await clickPapgMoveLink(gameTab, href);
    moves.push(decision.move);
    const readyText = await waitForPapgReady(gameTab);
    onProgress(`${decision.move}; ${pageStatus(readyText)}`);
  }

  if (!finalRecord) {
    const info = await readBoardInfo(gameTab);
    finalRecord = runPythonDecision({
      ...basePayload,
      moves,
      drawn_edges: info.drawn,
      write_record: true,
    });
  }

  if (!tab) {
    await gameTab.close();
  }
  return finalRecord;
}

async function clickPapgMoveLink(tab, href) {
  const selector = `a[href="${href.replaceAll('"', '\\"')}"]`;
  const link = tab.playwright.locator(selector);
  const count = await link.count();
  if (count !== 1) {
    throw new Error(`Expected one Papg move link for ${href}, saw ${count}.`);
  }
  await link.click({ timeoutMs: 5000 });
  await tab.playwright.waitForLoadState({ state: "load", timeoutMs: 15000 });
}

async function startGame(tab, rows, cols) {
  if (rows !== 4 || cols !== 4) {
    throw new Error("The browser runner currently starts Papg's 4x4-dot form option.");
  }

  await tab.goto("http://www.papg.com/dab.html");
  await tab.playwright.waitForLoadState({ state: "load", timeoutMs: 15000 });

  const size4 = tab.playwright.locator('input[name="SIZE"][value="1"]');
  const firstYes = tab.playwright.locator('input[name="HUMAN"][value="1"]');
  if ((await size4.count()) !== 1 || (await firstYes.count()) !== 1) {
    throw new Error("Could not find Papg 4x4/play-first controls.");
  }
  await size4.click({ timeoutMs: 5000 });
  await firstYes.click({ timeoutMs: 5000 });

  const checked = await tab.playwright.evaluate(() => ({
    size: document.querySelector('input[name="SIZE"]:checked')?.getAttribute("value"),
    human: document.querySelector('input[name="HUMAN"]:checked')?.getAttribute("value"),
  }));
  if (checked.size !== "1" || checked.human !== "1") {
    throw new Error(`Papg form did not select 4x4/play-first: ${JSON.stringify(checked)}`);
  }

  await tab.playwright.getByRole("button", { name: "New Game", exact: true }).click({ timeoutMs: 5000 });
  await tab.playwright.waitForLoadState({ state: "load", timeoutMs: 15000 });
  await waitForPapgReady(tab);

  const info = await readBoardInfo(tab);
  if (info.drawn.length > 0) {
    throw new Error(`Expected an empty Papg board after starting play-first, saw ${JSON.stringify(info.drawn)}`);
  }
}

async function readBoardInfo(tab) {
  return await tab.playwright.evaluate(() => {
    const table = document.querySelector("table");
    const info = {
      drawn: [],
      links: {},
      rects: {},
      text: document.body.innerText,
    };
    if (!table) {
      return info;
    }

    [...table.querySelectorAll("tr")].forEach((tr, tableRow) => {
      [...tr.querySelectorAll("td")].forEach((td, tableCol) => {
        let edge = null;
        if (tableRow % 2 === 0 && tableCol % 2 === 1) {
          edge = `h:${Math.floor(tableRow / 2)}:${Math.floor((tableCol - 1) / 2)}`;
        }
        if (tableRow % 2 === 1 && tableCol % 2 === 0) {
          edge = `v:${Math.floor((tableRow - 1) / 2)}:${Math.floor(tableCol / 2)}`;
        }
        if (!edge) {
          return;
        }

        const image = td.querySelector("img");
        const src = image ? image.getAttribute("src") || "" : "";
        if (/dab_[HV][BR]\.gif/.test(src)) {
          info.drawn.push(edge);
        }

        const link = td.querySelector('a[href^="/dab?"]');
        if (link) {
          info.links[edge] = link.getAttribute("href");
        }

        const rect = td.getBoundingClientRect();
        info.rects[edge] = {
          x: Math.round(rect.left + rect.width / 2),
          y: Math.round(rect.top + rect.height / 2),
        };
      });
    });
    return info;
  });
}

async function waitForPapgReady(tab) {
  let lastText = "";
  for (let attempt = 0; attempt < 45; attempt += 1) {
    const status = await tab.playwright.evaluate(() => {
      const text = document.body.innerText;
      const moveLinks = document.querySelectorAll('table a[href^="/dab?"]').length;
      const edgeImages = [...document.querySelectorAll('table img[src*="dab_"]')].filter((image) => {
        const src = image.getAttribute("src") || "";
        return /dab_[HV][BR]\.gif/.test(src);
      }).length;
      return { text, moveLinks, edgeImages };
    });
    lastText = status.text;
    if (
      !lastText.includes("Thinking...") &&
      (status.moveLinks > 0 || status.edgeImages === 24 || lastText.includes("Game finished"))
    ) {
      return lastText;
    }
    await sleep(1000);
  }
  return lastText;
}

function runPythonDecision(payload) {
  const encoded = Buffer.from(JSON.stringify(payload), "utf8").toString("base64");
  const code = `
import base64
import json
from pathlib import Path
from dots_boxes_mcts.game import apply_move, new_game
from dots_boxes_mcts.mcts import UCTMCTS, result_payload
from dots_boxes_mcts.external_games import edge_to_papg_index, external_game_record, append_jsonl
from dots_boxes_mcts.papg_eval import infer_papg_reply

payload = json.loads(base64.b64decode("${encoded}"))
rows = payload["rows"]
cols = payload["cols"]
state = new_game(rows=rows, cols=cols)
moves = list(payload["moves"])
for move in moves:
    state = apply_move(state, move)

missing = [edge for edge in payload["drawn_edges"] if edge not in state.edges]
if missing:
    for move in infer_papg_reply(state, missing):
        moves.append(move)
        state = apply_move(state, move)

if payload.get("write_record"):
    record = external_game_record(
        source="papg",
        opponent="papg",
        bot=f"uct_mcts_{payload['simulations']}",
        rows=rows,
        cols=cols,
        moves=moves,
        our_player=0,
        notes="Browser-backed Stage 2.5 Papg game.",
    )
    record["seed"] = payload["seed"]
    record["simulations"] = payload["simulations"]
    append_jsonl(record, Path(payload["out"]))
    print(json.dumps({
        "moves": moves,
        "terminal": record["terminal"],
        "scores": record["finalScores"],
        "winner": record["winner"],
    }))
elif state.terminal:
    print(json.dumps({
        "moves": moves,
        "terminal": True,
        "scores": list(state.scores),
        "winner": state.winner,
    }))
else:
    result = UCTMCTS(
        simulations=payload["simulations"],
        seed=payload["seed"] + len(moves),
    ).search(state)
    print(json.dumps({
        "moves": moves,
        "terminal": False,
        "move": result.move,
        "index": edge_to_papg_index(result.move, rows=rows, cols=cols),
        "search": result_payload(result),
    }))
`;

  const result = spawnSync(
    "/bin/zsh",
    ["-lc", 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate data && python -'],
    {
      input: code,
      cwd: REPO_ROOT,
      encoding: "utf8",
      maxBuffer: 1024 * 1024 * 20,
    },
  );
  if (result.status !== 0) {
    throw new Error(result.stderr || result.stdout);
  }
  return JSON.parse(result.stdout.trim().split("\\n").at(-1));
}

function summarize(records, simulations) {
  const wins = records.filter((record) => record.winner === 0).length;
  const draws = records.filter((record) => record.winner === "draw").length;
  const losses = records.length - wins - draws;
  const margins = records.map((record) => record.scores[0] - record.scores[1]);
  return {
    simulations,
    games: records.length,
    wins,
    draws,
    losses,
    winRate: records.length === 0 ? 0 : wins / records.length,
    averageScoreMargin: margins.length === 0 ? 0 : margins.reduce((sum, value) => sum + value, 0) / margins.length,
  };
}

function pageStatus(text) {
  return text.match(/(Your move again\.|Your move\.|Click on the board to move\.|Game finished[^\n]*)/)?.[1] || "ready";
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
