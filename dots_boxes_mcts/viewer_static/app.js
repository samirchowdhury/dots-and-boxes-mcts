const state = {
  payload: null,
  frameIndex: 0,
};

const elements = {
  form: document.querySelector("#load-form"),
  fileSelect: document.querySelector("#file-select"),
  lineInput: document.querySelector("#line-input"),
  board: document.querySelector("#board"),
  moveCounter: document.querySelector("#move-counter"),
  score: document.querySelector("#score"),
  turn: document.querySelector("#turn"),
  lastMove: document.querySelector("#last-move"),
  summary: document.querySelector("#game-summary"),
  moveList: document.querySelector("#move-list"),
  slider: document.querySelector("#move-slider"),
  firstButton: document.querySelector("#first-button"),
  prevButton: document.querySelector("#prev-button"),
  nextButton: document.querySelector("#next-button"),
  lastButton: document.querySelector("#last-button"),
  message: document.querySelector("#message"),
};

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

function getCssVariable(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

async function fetchJson(url) {
  const response = await fetch(url);
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.error || "Request failed.");
  }
  return body;
}

async function loadFiles() {
  const payload = await fetchJson("/api/files");
  elements.fileSelect.innerHTML = "";

  if (payload.files.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No JSONL files in runs/";
    elements.fileSelect.append(option);
    elements.fileSelect.disabled = true;
    return;
  }

  for (const fileName of payload.files) {
    const option = document.createElement("option");
    option.value = fileName;
    option.textContent = fileName;
    elements.fileSelect.append(option);
  }
}

async function loadGame() {
  const fileName = elements.fileSelect.value;
  const lineNumber = elements.lineInput.value || "1";
  const url = `/api/game?file=${encodeURIComponent(fileName)}&line=${encodeURIComponent(lineNumber)}`;
  state.payload = await fetchJson(url);
  state.frameIndex = 0;
  elements.slider.max = String(state.payload.frames.length - 1);
  elements.slider.value = "0";
  render();
}

function currentFrame() {
  return state.payload?.frames[state.frameIndex] || null;
}

function render() {
  if (!state.payload) {
    renderEmptyBoard();
    return;
  }

  const frame = currentFrame();
  const snapshot = frame.state;
  const totalMoves = state.payload.frames.length - 1;
  const latestHistory = frame.historyEntry;

  elements.moveCounter.textContent = `${state.frameIndex} / ${totalMoves}`;
  elements.score.textContent = `${snapshot.scores[0]} - ${snapshot.scores[1]}`;
  elements.turn.textContent = snapshot.terminal
    ? winnerText(snapshot.winner)
    : `Player ${snapshot.currentPlayer}`;
  elements.lastMove.textContent = latestHistory
    ? moveText(latestHistory)
    : "Start";
  elements.summary.innerHTML = [
    `File: ${state.payload.file}`,
    `Line: ${state.payload.line}`,
    `Board: ${state.payload.record.rows} x ${state.payload.record.cols} dots`,
    `Seed: ${state.payload.record.seed ?? "n/a"}`,
    `Winner: ${winnerText(state.payload.record.winner)}`,
  ]
    .map((line) => `<span>${escapeHtml(line)}</span>`)
    .join("");

  renderBoard(snapshot, frame.move);
  renderMoveList();
  updateTransport();
}

function winnerText(winner) {
  if (winner === "draw") {
    return "Draw";
  }
  if (winner === 0 || winner === 1) {
    return `Player ${winner} wins`;
  }
  return "In progress";
}

function moveText(historyEntry) {
  const scoredText = historyEntry.scored
    ? ` scored ${historyEntry.boxes.length}`
    : "";
  return `P${historyEntry.player}: ${historyEntry.edgeId}${scoredText}`;
}

function renderBoard(snapshot, lastMove) {
  const rows = snapshot.rows;
  const cols = snapshot.cols;
  const spacing = 88;
  const margin = 42;
  const width = (cols - 1) * spacing + margin * 2;
  const height = (rows - 1) * spacing + margin * 2;

  elements.board.setAttribute("viewBox", `0 0 ${width} ${height}`);
  elements.board.innerHTML = "";

  const edgeOwners = new Map(snapshot.edgeOwners);
  const drawnEdges = new Set(snapshot.edges);

  for (let row = 0; row < rows - 1; row += 1) {
    for (let col = 0; col < cols - 1; col += 1) {
      const owner = snapshot.boxes[row][col];
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
      const edgeId = `h:${row}:${col}`;
      addEdge(edgeId, margin + col * spacing, margin + row * spacing, margin + (col + 1) * spacing, margin + row * spacing);
    }
  }

  for (let row = 0; row < rows - 1; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const edgeId = `v:${row}:${col}`;
      addEdge(edgeId, margin + col * spacing, margin + row * spacing, margin + col * spacing, margin + (row + 1) * spacing);
    }
  }

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      addDot(margin + col * spacing, margin + row * spacing);
    }
  }

  function addEdge(edgeId, x1, y1, x2, y2) {
    const line = createSvgElement("line");
    line.setAttribute("x1", x1);
    line.setAttribute("y1", y1);
    line.setAttribute("x2", x2);
    line.setAttribute("y2", y2);
    line.setAttribute("stroke-width", "7");

    if (drawnEdges.has(edgeId)) {
      const owner = edgeOwners.get(edgeId);
      line.setAttribute("class", "drawn-edge");
      line.setAttribute("stroke", colors[owner].edge);
    } else {
      line.setAttribute("class", "available-edge");
    }

    elements.board.append(line);
  }

  function addDot(cx, cy) {
    const circle = createSvgElement("circle");
    circle.setAttribute("class", "dot");
    circle.setAttribute("cx", cx);
    circle.setAttribute("cy", cy);
    circle.setAttribute("r", "6");
    elements.board.append(circle);
  }

  function addRect(x, y, boxWidth, boxHeight, fill) {
    const rect = createSvgElement("rect");
    rect.setAttribute("x", x);
    rect.setAttribute("y", y);
    rect.setAttribute("width", boxWidth);
    rect.setAttribute("height", boxHeight);
    rect.setAttribute("rx", "5");
    rect.setAttribute("fill", fill);
    elements.board.append(rect);
  }

  function addText(x, y, textValue) {
    const text = createSvgElement("text");
    text.setAttribute("class", "box-label");
    text.setAttribute("x", x);
    text.setAttribute("y", y);
    text.textContent = textValue;
    elements.board.append(text);
  }
}

function renderMoveList() {
  elements.moveList.innerHTML = "";

  for (const frame of state.payload.frames.slice(1)) {
    const item = document.createElement("li");
    item.textContent = moveText(frame.historyEntry);
    if (frame.moveNumber === state.frameIndex) {
      item.classList.add("active");
    }
    item.addEventListener("click", () => {
      state.frameIndex = frame.moveNumber;
      elements.slider.value = String(state.frameIndex);
      render();
    });
    elements.moveList.append(item);
  }
}

function updateTransport() {
  const maxIndex = state.payload.frames.length - 1;
  elements.slider.value = String(state.frameIndex);
  elements.firstButton.disabled = state.frameIndex === 0;
  elements.prevButton.disabled = state.frameIndex === 0;
  elements.nextButton.disabled = state.frameIndex === maxIndex;
  elements.lastButton.disabled = state.frameIndex === maxIndex;
}

function renderEmptyBoard() {
  elements.board.setAttribute("viewBox", "0 0 420 420");
  elements.board.innerHTML = "";
}

function createSvgElement(tagName) {
  return document.createElementNS("http://www.w3.org/2000/svg", tagName);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setMessage(text, isError = false) {
  elements.message.textContent = text;
  elements.message.style.color = isError ? "#a53321" : "";
}

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    setMessage("Loading game...");
    await loadGame();
    setMessage("Game loaded.");
  } catch (error) {
    setMessage(error.message, true);
  }
});

elements.slider.addEventListener("input", () => {
  state.frameIndex = Number(elements.slider.value);
  render();
});

elements.firstButton.addEventListener("click", () => {
  state.frameIndex = 0;
  render();
});

elements.prevButton.addEventListener("click", () => {
  state.frameIndex = Math.max(0, state.frameIndex - 1);
  render();
});

elements.nextButton.addEventListener("click", () => {
  state.frameIndex = Math.min(state.payload.frames.length - 1, state.frameIndex + 1);
  render();
});

elements.lastButton.addEventListener("click", () => {
  state.frameIndex = state.payload.frames.length - 1;
  render();
});

loadFiles()
  .then(() => {
    renderEmptyBoard();
    if (!elements.fileSelect.disabled) {
      return loadGame();
    }
    setMessage("Generate a JSONL file under runs/ to begin.");
  })
  .then(() => {
    if (state.payload) {
      setMessage("Game loaded.");
    }
  })
  .catch((error) => setMessage(error.message, true));
