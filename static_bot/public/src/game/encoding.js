import { allEdgeIds, parseEdgeId } from "./rules.js";

export const CHANNEL_NAMES = [
  "drawn_edge",
  "current_player_edge",
  "opponent_edge",
  "current_player_box",
  "opponent_box",
  "legal_move",
  "current_player",
  "score_margin",
];

export function boardShape(rows, cols) {
  return { height: 2 * rows - 1, width: 2 * cols - 1 };
}

export function edgeCoordinate(edge) {
  const { kind, row, col } = parseEdgeId(edge);
  if (kind === "h") {
    return { row: 2 * row, col: 2 * col + 1 };
  }
  return { row: 2 * row + 1, col: 2 * col };
}

export function boxCoordinate(row, col) {
  return { row: 2 * row + 1, col: 2 * col + 1 };
}

export function encodeSnapshot(snapshot) {
  const rows = Number(snapshot.rows);
  const cols = Number(snapshot.cols);
  const player = Number(snapshot.currentPlayer);
  const opponent = player === 0 ? 1 : 0;
  const { height, width } = boardShape(rows, cols);
  const channels = CHANNEL_NAMES.length;
  const tensor = new Float32Array(height * width * channels);
  const edgeOwners = new Map((snapshot.edgeOwners || []).map(([edge, owner]) => [edge, owner]));

  for (const edge of snapshot.edges || []) {
    const coord = edgeCoordinate(edge);
    const owner = edgeOwners.get(edge);
    tensor[indexOf(coord.row, coord.col, 0, width, channels)] = 1;
    if (owner === player) {
      tensor[indexOf(coord.row, coord.col, 1, width, channels)] = 1;
    } else if (owner === opponent) {
      tensor[indexOf(coord.row, coord.col, 2, width, channels)] = 1;
    }
  }

  for (let row = 0; row < (snapshot.boxes || []).length; row += 1) {
    for (let col = 0; col < snapshot.boxes[row].length; col += 1) {
      const owner = snapshot.boxes[row][col];
      if (owner === null || owner === undefined) {
        continue;
      }
      const coord = boxCoordinate(row, col);
      if (owner === player) {
        tensor[indexOf(coord.row, coord.col, 3, width, channels)] = 1;
      } else if (owner === opponent) {
        tensor[indexOf(coord.row, coord.col, 4, width, channels)] = 1;
      }
    }
  }

  const edgeSet = new Set(snapshot.edges || []);
  const ids = allEdgeIds(rows, cols);
  const legalMask = new Float32Array(ids.length);
  for (let action = 0; action < ids.length; action += 1) {
    const edge = ids[action];
    if (!edgeSet.has(edge)) {
      const coord = edgeCoordinate(edge);
      legalMask[action] = 1;
      tensor[indexOf(coord.row, coord.col, 5, width, channels)] = 1;
    }
  }

  if (player === 1) {
    fillChannel(tensor, height, width, channels, 6, 1);
  }

  const scores = snapshot.scores || [0, 0];
  const totalBoxes = Math.max((rows - 1) * (cols - 1), 1);
  fillChannel(tensor, height, width, channels, 7, (scores[player] - scores[opponent]) / totalBoxes);

  return { tensor, legalMask, actionIds: ids, height, width, channels };
}

export function indexOf(row, col, channel, width, channels) {
  return (row * width + col) * channels + channel;
}

function fillChannel(tensor, height, width, channels, channel, value) {
  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < width; col += 1) {
      tensor[indexOf(row, col, channel, width, channels)] = value;
    }
  }
}
