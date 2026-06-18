export async function createWasmSearchBackend() {
  return {
    available: false,
    reason: "The static app is wired for a WASM search backend, but no module is built yet.",
  };
}
