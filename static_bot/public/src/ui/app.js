import {
  applyMove,
  legalMoves,
  newGame,
} from "../game/rules.js";

const boardEl = document.querySelector("#board");
const statusEl = document.querySelector("#status");
const botSummaryEl = document.querySelector("#botSummary");
const humanScoreEl = document.querySelector("#humanScore");
const botScoreEl = document.querySelector("#botScore");
const lastMoveEl = document.querySelector("#lastMove");
const resetButton = document.querySelector("#resetButton");
const playerSelect = document.querySelector("#playerSelect");
const simulationsInput = document.querySelector("#simulationsInput");

const botUrl = new URL("./assets/bots/latest.json", window.location.href).toString();
const worker = new Worker(new URL("../worker/bot-worker.js", import.meta.url), { type: "module" });

let state = newGame(4, 4);
let humanPlayer = 0;
let botPlayer = 1;
let thinking = false;
let botLoaded = false;
let lastMoveEdge = null;
let statusOverride = "";

const colors = {
  0: {
    edge: getCssVariable("--player-0"),
    box: getCssVariable("--player-0-soft"),
  },
  1: {
    edge: getCssVariable("--player-1"),
    box: getCssVariable("--player-1-soft"),
  },
};

worker.addEventListener("message", (event) => {
  const message = event.data;
  if (message.type === "botLoaded") {
    botLoaded = true;
    updateBotSummary(message.bot);
    render();
    if (state.currentPlayer === botPlayer) {
      requestBotMove();
    }
    return;
  }
  if (message.type === "thinking") {
    thinking = true;
    statusOverride = "";
    statusEl.textContent = `Bot thinking with ${message.simulations} simulations...`;
    render();
    return;
  }
  if (message.type === "move") {
    thinking = false;
    statusOverride = "";
    state = applyMove(state, message.move);
    lastMoveEdge = message.move;
    const top = message.search.stats[0];
    lastMoveEl.title = `Bot played in ${message.elapsedMs.toFixed(0)} ms. Top line: ${
      top.move
    }, ${top.visits} visits.`;
    render();
    if (!state.terminal && state.currentPlayer === botPlayer) {
      window.setTimeout(requestBotMove, 80);
    }
    return;
  }
  if (message.type === "error") {
    thinking = false;
    statusOverride = message.message;
    render();
  }
});

resetButton.addEventListener("click", () => resetGame());
playerSelect.addEventListener("change", () => resetGame());
simulationsInput.addEventListener("change", () => {
  simulationsInput.value = String(clampedSimulations());
});

worker.postMessage({ type: "loadBot", botUrl });
render();

function resetGame() {
  humanPlayer = Number(playerSelect.value);
  botPlayer = humanPlayer === 0 ? 1 : 0;
  state = newGame(4, 4);
  thinking = false;
  statusOverride = "";
  lastMoveEdge = null;
  lastMoveEl.title = "";
  render();
  if (botLoaded && state.currentPlayer === botPlayer) {
    requestBotMove();
  }
}

function requestBotMove() {
  if (thinking || state.terminal || state.currentPlayer !== botPlayer) {
    return;
  }
  thinking = true;
  statusOverride = "";
  render();
  worker.postMessage({
    type: "chooseMove",
    botUrl,
    state,
    simulations: clampedSimulations(),
    backend: "wasm",
    cPuct: 1.5,
  });
}

function render() {
  renderBoard();
  humanScoreEl.textContent = String(state.scores[humanPlayer]);
  botScoreEl.textContent = String(state.scores[botPlayer]);
  lastMoveEl.textContent = lastMoveText();
  if (statusOverride) {
    statusEl.textContent = statusOverride;
  } else if (!thinking) {
    statusEl.textContent = statusText();
  }
}

function renderBoard() {
  const rows = state.rows;
  const cols = state.cols;
  const spacing = 88;
  const margin = 42;
  const width = (cols - 1) * spacing + margin * 2;
  const height = (rows - 1) * spacing + margin * 2;
  const ownerByEdge = new Map(state.edgeOwners);
  const drawnEdges = new Set(state.edges);
  const legal = new Set(legalMoves(state));

  boardEl.setAttribute("viewBox", `0 0 ${width} ${height}`);
  boardEl.replaceChildren();

  for (let row = 0; row < rows - 1; row += 1) {
    for (let col = 0; col < cols - 1; col += 1) {
      const owner = state.boxes[row][col];
      if (owner !== null) {
        addRect(
          margin + col * spacing + 7,
          margin + row * spacing + 7,
          spacing - 14,
          spacing - 14,
          colors[owner].box,
        );
        addText(
          margin + col * spacing + spacing / 2,
          margin + row * spacing + spacing / 2,
          `P${owner}`,
        );
      }
    }
  }

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols - 1; col += 1) {
      const edge = `h:${row}:${col}`;
      addVisibleEdge(
        edge,
        margin + col * spacing,
        margin + row * spacing,
        margin + (col + 1) * spacing,
        margin + row * spacing,
        ownerByEdge,
        drawnEdges,
        legal,
      );
    }
  }

  for (let row = 0; row < rows - 1; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const edge = `v:${row}:${col}`;
      addVisibleEdge(
        edge,
        margin + col * spacing,
        margin + row * spacing,
        margin + col * spacing,
        margin + (row + 1) * spacing,
        ownerByEdge,
        drawnEdges,
        legal,
      );
    }
  }

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols - 1; col += 1) {
      const edge = `h:${row}:${col}`;
      addHitEdge(
        edge,
        margin + col * spacing,
        margin + row * spacing,
        margin + (col + 1) * spacing,
        margin + row * spacing,
        legal,
      );
    }
  }

  for (let row = 0; row < rows - 1; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const edge = `v:${row}:${col}`;
      addHitEdge(
        edge,
        margin + col * spacing,
        margin + row * spacing,
        margin + col * spacing,
        margin + (row + 1) * spacing,
        legal,
      );
    }
  }

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      addDot(margin + col * spacing, margin + row * spacing);
    }
  }
}

function playHumanMove(edge) {
  if (thinking || state.terminal || state.currentPlayer !== humanPlayer) {
    return;
  }
  state = applyMove(state, edge);
  lastMoveEdge = edge;
  lastMoveEl.title = "";
  render();
  if (!state.terminal && state.currentPlayer === botPlayer) {
    requestBotMove();
  }
}

function statusText() {
  if (!botLoaded) {
    return "Loading bot...";
  }
  if (state.terminal) {
    if (state.winner === "draw") {
      return "Draw.";
    }
    return state.winner === humanPlayer ? "You win." : "Bot wins.";
  }
  if (state.currentPlayer === humanPlayer) {
    return "Your move.";
  }
  return "Bot to move.";
}

function updateBotSummary(bot) {
  botSummaryEl.textContent = `ITER ${String(bot.iteration).padStart(3, "0")}`;
}

function clampedSimulations() {
  const parsed = Number(simulationsInput.value);
  if (!Number.isFinite(parsed)) {
    return 300;
  }
  return Math.min(5000, Math.max(1, Math.floor(parsed)));
}

function addVisibleEdge(edge, x1, y1, x2, y2, ownerByEdge, drawnEdges, legal) {
  const line = createSvgElement("line");
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);
  line.setAttribute("stroke-width", "7");

  if (drawnEdges.has(edge)) {
    const owner = ownerByEdge.get(edge);
    line.setAttribute("class", "drawn-edge");
    line.setAttribute("stroke", colors[owner].edge);
  } else {
    line.setAttribute("class", "available-edge");
  }

  boardEl.append(line);
}

function addHitEdge(edge, x1, y1, x2, y2, legal) {
  const disabled =
    state.terminal || thinking || state.currentPlayer !== humanPlayer || !legal.has(edge);
  const pad = 16;
  const rect = createSvgElement("rect");
  rect.setAttribute("x", Math.min(x1, x2) - pad);
  rect.setAttribute("y", Math.min(y1, y2) - pad);
  rect.setAttribute("width", Math.abs(x2 - x1) + pad * 2);
  rect.setAttribute("height", Math.abs(y2 - y1) + pad * 2);
  rect.setAttribute("rx", "10");
  rect.setAttribute("class", `edge-hit${disabled ? " disabled" : ""}`);
  if (!disabled) {
    rect.setAttribute("aria-label", edge);
    rect.setAttribute("role", "button");
    rect.setAttribute("tabindex", "0");
    rect.addEventListener("click", () => playHumanMove(edge));
    rect.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        playHumanMove(edge);
      }
    });
  } else {
    rect.setAttribute("aria-hidden", "true");
  }
  boardEl.append(rect);
}

function addDot(cx, cy) {
  const circle = createSvgElement("circle");
  circle.setAttribute("class", "dot");
  circle.setAttribute("cx", cx);
  circle.setAttribute("cy", cy);
  circle.setAttribute("r", "6");
  boardEl.append(circle);
}

function addRect(x, y, width, height, fill) {
  const rect = createSvgElement("rect");
  rect.setAttribute("x", x);
  rect.setAttribute("y", y);
  rect.setAttribute("width", width);
  rect.setAttribute("height", height);
  rect.setAttribute("rx", "5");
  rect.setAttribute("fill", fill);
  boardEl.append(rect);
}

function addText(x, y, value) {
  const text = createSvgElement("text");
  text.setAttribute("class", "box-label");
  text.setAttribute("x", x);
  text.setAttribute("y", y);
  text.textContent = value;
  boardEl.append(text);
}

function lastMoveText() {
  const latest = state.history[state.history.length - 1];
  if (!latest) {
    return "Start";
  }
  const scoredText = latest.scored ? ` scored ${latest.boxes.length}` : "";
  return `P${latest.player}: ${latest.edgeId}${scoredText}`;
}

function createSvgElement(tagName) {
  return document.createElementNS("http://www.w3.org/2000/svg", tagName);
}

function getCssVariable(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
