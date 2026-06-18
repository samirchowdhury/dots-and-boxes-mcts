# WASM Backend Slot

The app currently runs the network evaluator and MCTS in a Web Worker using
portable JavaScript. The worker already accepts `backend: "wasm"` and falls back
to JavaScript through `public/src/mcts/wasm-search.js`.

When the JavaScript path is validated, the next step is to compile the hot MCTS
tree/search loop here and keep the browser neural evaluator unchanged. The WASM
boundary should use batched leaf evaluations so the compiled search core does
not cross into JavaScript once per simulation.
