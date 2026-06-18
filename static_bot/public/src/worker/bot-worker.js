import { loadBot } from "../nn/loader.js";
import { searchNetworkGuided } from "../mcts/search.js";
import { createWasmSearchBackend } from "../mcts/wasm-search.js";

let model = null;
let modelUrl = null;
let wasmBackend = null;

self.addEventListener("message", (event) => {
  handleMessage(event.data).catch((error) => {
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  });
});

async function handleMessage(message) {
  if (message.type === "loadBot") {
    const loaded = await ensureModel(message.botUrl);
    self.postMessage({ type: "botLoaded", bot: botSummary(loaded) });
    return;
  }

  if (message.type === "chooseMove") {
    const loaded = await ensureModel(message.botUrl);
    const simulations = Math.max(1, Number(message.simulations || 100));
    let backendUsed = "js";
    if (message.backend === "wasm") {
      wasmBackend = wasmBackend || (await createWasmSearchBackend());
      backendUsed = wasmBackend.available ? "wasm" : "js";
    }

    self.postMessage({ type: "thinking", simulations, backendUsed });
    const started = performance.now();
    const search = searchNetworkGuided(loaded, message.state, {
      simulations,
      cPuct: Number(message.cPuct || 1.5),
      selector: wasmBackend?.available ? wasmBackend : null,
    });
    const elapsedMs = performance.now() - started;
    self.postMessage({
      type: "move",
      move: search.move,
      search,
      elapsedMs,
      backendUsed,
      bot: botSummary(loaded),
    });
  }
}

async function ensureModel(botUrl) {
  if (!model || modelUrl !== botUrl) {
    modelUrl = botUrl;
    model = await loadBot(botUrl);
  }
  return model;
}

function botSummary(loaded) {
  return {
    iteration: loaded.manifest.iteration,
    board: loaded.manifest.board,
    actionCount: loaded.manifest.model.actionCount,
    hiddenSize: loaded.manifest.model.hiddenSize,
    residualBlocks: loaded.manifest.model.residualBlocks,
    weightsByteLength: loaded.manifest.weightsByteLength,
  };
}
