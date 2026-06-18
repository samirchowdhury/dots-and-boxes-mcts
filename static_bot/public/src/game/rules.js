export function newGame(rows = 4, cols = 4) {
  if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 2 || cols < 2) {
    throw new Error("Dots and Boxes needs at least a 2x2 dot grid.");
  }
  return {
    rows,
    cols,
    currentPlayer: 0,
    edges: [],
    edgeOwners: [],
    boxes: Array.from({ length: rows - 1 }, () => Array(cols - 1).fill(null)),
    scores: [0, 0],
    history: [],
    terminal: false,
    winner: null,
  };
}

export function allEdgeIds(rows, cols) {
  const edges = [];
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols - 1; col += 1) {
      edges.push(edgeId("h", row, col));
    }
  }
  for (let row = 0; row < rows - 1; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      edges.push(edgeId("v", row, col));
    }
  }
  return edges;
}

export function edgeId(kind, row, col) {
  return `${kind}:${row}:${col}`;
}

export function parseEdgeId(value) {
  const [kind, rowText, colText] = String(value).split(":");
  const row = Number(rowText);
  const col = Number(colText);
  if ((kind !== "h" && kind !== "v") || !Number.isInteger(row) || !Number.isInteger(col)) {
    throw new Error(`Invalid edge id: ${value}`);
  }
  return { kind, row, col };
}

export function legalMoves(state) {
  if (state.terminal) {
    return [];
  }
  const drawn = new Set(state.edges);
  return allEdgeIds(state.rows, state.cols).filter((edge) => !drawn.has(edge));
}

export function applyMove(state, edge) {
  if (state.terminal) {
    throw new Error("Cannot move after the game is over.");
  }
  const parsed = parseEdgeId(edge);
  if (!isEdgeInBounds(state, parsed)) {
    throw new Error(`Edge is out of bounds: ${edge}`);
  }
  if (state.edges.includes(edge)) {
    throw new Error(`Edge is already drawn: ${edge}`);
  }

  const player = state.currentPlayer;
  const edges = [...state.edges, edge].sort();
  const edgeSet = new Set(edges);
  const edgeOwners = [...state.edgeOwners, [edge, player]].sort((a, b) => a[0].localeCompare(b[0]));
  const boxes = state.boxes.map((row) => row.slice());
  const scores = state.scores.slice();

  const scoredBoxes = [];
  for (const box of boxesForEdge(state.rows, state.cols, edge)) {
    if (boxes[box.row][box.col] === null && isBoxComplete(edgeSet, box.row, box.col)) {
      boxes[box.row][box.col] = player;
      scores[player] += 1;
      scoredBoxes.push({ row: box.row, col: box.col });
    }
  }

  const next = {
    rows: state.rows,
    cols: state.cols,
    currentPlayer: scoredBoxes.length > 0 ? player : otherPlayer(player),
    edges,
    edgeOwners,
    boxes,
    scores,
    history: [
      ...(state.history || []),
      {
        player,
        edgeId: edge,
        scored: scoredBoxes.length > 0,
        boxes: scoredBoxes,
      },
    ],
    terminal: false,
    winner: null,
  };

  if (legalMoves(next).length === 0) {
    next.terminal = true;
    next.winner = winnerFor(next);
  }
  return next;
}

export function boxEdgeIds(row, col) {
  return [
    edgeId("h", row, col),
    edgeId("h", row + 1, col),
    edgeId("v", row, col),
    edgeId("v", row, col + 1),
  ];
}

export function boxesForEdge(rows, cols, edge) {
  const { kind, row, col } = parseEdgeId(edge);
  const boxes = [];
  if (kind === "h") {
    if (row > 0) {
      boxes.push({ row: row - 1, col });
    }
    if (row < rows - 1) {
      boxes.push({ row, col });
    }
  } else {
    if (col > 0) {
      boxes.push({ row, col: col - 1 });
    }
    if (col < cols - 1) {
      boxes.push({ row, col });
    }
  }
  return boxes.filter(
    (box) => box.row >= 0 && box.row < rows - 1 && box.col >= 0 && box.col < cols - 1,
  );
}

export function stateSnapshot(state) {
  return {
    rows: state.rows,
    cols: state.cols,
    currentPlayer: state.currentPlayer,
    edges: [...state.edges].sort(),
    edgeOwners: [...state.edgeOwners].sort((a, b) => a[0].localeCompare(b[0])),
    boxes: state.boxes.map((row) => row.slice()),
    scores: state.scores.slice(),
    terminal: state.terminal,
    winner: state.winner,
  };
}

export function winnerFor(state) {
  if (state.scores[0] > state.scores[1]) {
    return 0;
  }
  if (state.scores[1] > state.scores[0]) {
    return 1;
  }
  return "draw";
}

function isBoxComplete(edgeSet, row, col) {
  return boxEdgeIds(row, col).every((edge) => edgeSet.has(edge));
}

function isEdgeInBounds(state, { kind, row, col }) {
  if (kind === "h") {
    return row >= 0 && row < state.rows && col >= 0 && col < state.cols - 1;
  }
  return row >= 0 && row < state.rows - 1 && col >= 0 && col < state.cols;
}

export function otherPlayer(player) {
  return player === 0 ? 1 : 0;
}
