import { applyMove, legalMoves, stateSnapshot } from "../game/rules.js";
import { evaluate } from "../nn/inference.js";

class Node {
  constructor(state, prior = 1, move = null) {
    this.state = state;
    this.prior = prior;
    this.move = move;
    this.children = [];
    this.visits = 0;
    this.valueSum = 0;
  }

  expanded() {
    return this.children.length > 0;
  }

  meanValue() {
    return this.visits === 0 ? 0 : this.valueSum / this.visits;
  }
}

export function searchNetworkGuided(model, state, options = {}) {
  if (state.terminal) {
    throw new Error("Cannot search from a terminal state.");
  }
  const simulations = Math.max(1, Number(options.simulations || 100));
  const cPuct = Math.max(0, Number(options.cPuct || 1.5));
  const selector = options.selector || null;
  const root = new Node(state);
  expand(model, root);

  for (let simulation = 0; simulation < simulations; simulation += 1) {
    runSimulation(model, root, cPuct, selector);
  }

  return {
    move: bestRootMove(root),
    simulations,
    rootPlayer: root.state.currentPlayer,
    stats: rootStats(root),
  };
}

function runSimulation(model, root, cPuct, selector) {
  let node = root;
  const path = [];

  while (node.expanded() && !node.state.terminal) {
    const child = selectChild(node, cPuct, selector);
    path.push([node, child]);
    node = child;
  }

  let leafPlayer;
  let leafValue;
  if (node.state.terminal) {
    leafPlayer = node.state.currentPlayer;
    leafValue = terminalValue(node.state, leafPlayer);
  } else {
    leafPlayer = node.state.currentPlayer;
    leafValue = expand(model, node);
  }

  for (const [parent, child] of path) {
    child.visits += 1;
    child.valueSum += parent.state.currentPlayer === leafPlayer ? leafValue : -leafValue;
  }
  root.visits += 1;
}

function expand(model, node) {
  const { priors, value } = evaluate(model, stateSnapshot(node.state));
  const moves = legalMoves(node.state);
  node.children = moves.map(
    (move) => new Node(nodeAfterMove(node.state, move), priors[move] || 0, move),
  );

  const totalPrior = node.children.reduce((total, child) => total + Math.max(0, child.prior), 0);
  if (node.children.length > 0 && totalPrior <= 0) {
    const uniform = 1 / node.children.length;
    for (const child of node.children) {
      child.prior = uniform;
    }
  } else if (totalPrior > 0) {
    for (const child of node.children) {
      child.prior = Math.max(0, child.prior) / totalPrior;
    }
  }
  return value;
}

function nodeAfterMove(state, move) {
  return applyMove(state, move);
}

function selectChild(node, cPuct, selector) {
  if (selector && selector.available) {
    return selector.selectChild(node.children, node.visits, cPuct);
  }
  const sqrtParent = Math.sqrt(Math.max(node.visits, 1));
  let best = null;
  let bestScore = -Infinity;
  for (const child of node.children) {
    const q = child.meanValue();
    const u = (cPuct * child.prior * sqrtParent) / (1 + child.visits);
    const score = q + u;
    if (score > bestScore) {
      best = child;
      bestScore = score;
    }
  }
  return best;
}

function bestRootMove(root) {
  if (root.children.length === 0) {
    throw new Error("Search did not expand any legal moves.");
  }
  return [...root.children].sort((a, b) => {
    if (b.visits !== a.visits) {
      return b.visits - a.visits;
    }
    if (b.meanValue() !== a.meanValue()) {
      return b.meanValue() - a.meanValue();
    }
    return String(b.move).localeCompare(String(a.move));
  })[0].move;
}

function rootStats(root) {
  return [...root.children]
    .sort((a, b) => {
      if (b.visits !== a.visits) {
        return b.visits - a.visits;
      }
      return String(a.move).localeCompare(String(b.move));
    })
    .map((child) => ({
      move: child.move,
      visits: child.visits,
      meanValue: child.meanValue(),
    }));
}

function terminalValue(state, player) {
  const totalBoxes = (state.rows - 1) * (state.cols - 1);
  if (totalBoxes === 0) {
    return 0;
  }
  const opponent = player === 0 ? 1 : 0;
  return (state.scores[player] - state.scores[opponent]) / totalBoxes;
}
