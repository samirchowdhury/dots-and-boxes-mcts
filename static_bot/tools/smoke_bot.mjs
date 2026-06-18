import { readFile } from "node:fs/promises";
import path from "node:path";

import { newGame } from "../public/src/game/rules.js";
import { searchNetworkGuided } from "../public/src/mcts/search.js";
import { evaluate } from "../public/src/nn/inference.js";

const manifestPath = await resolveManifestPath(
  process.argv[2] || "static_bot/public/assets/bots/latest.json",
);
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
const weightsPath = path.join(path.dirname(manifestPath), manifest.weightsFile);
const weightsBytes = await readFile(weightsPath);
const weightsBuffer = weightsBytes.buffer.slice(
  weightsBytes.byteOffset,
  weightsBytes.byteOffset + weightsBytes.byteLength,
);
const weights = {};

for (const [name, spec] of Object.entries(manifest.weights)) {
  weights[name] = new Float32Array(weightsBuffer, spec.byteOffset, spec.byteLength / 4);
}

const model = { manifest, weights };
const state = newGame(manifest.board.rows, manifest.board.cols);
const evaluation = evaluate(model, state);
const search = searchNetworkGuided(model, state, { simulations: 5 });
const priorTotal = Object.values(evaluation.priors).reduce((total, value) => total + value, 0);

console.log(
  JSON.stringify(
    {
      iteration: manifest.iteration,
      priorCount: Object.keys(evaluation.priors).length,
      priorTotal,
      value: evaluation.value,
      move: search.move,
      topPolicy: Object.entries(evaluation.priors)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5),
      topSearch: search.stats[0],
    },
    null,
    2,
  ),
);

async function resolveManifestPath(inputPath) {
  if (path.basename(inputPath) !== "latest.json") {
    return inputPath;
  }
  const latest = JSON.parse(await readFile(inputPath, "utf8"));
  return path.join(path.dirname(inputPath), latest.manifest);
}
