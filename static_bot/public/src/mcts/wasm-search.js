export async function createWasmSearchBackend() {
  const moduleUrl = new URL("../../../wasm/search-kernel.wasm", import.meta.url);
  const instance = await WebAssembly.instantiateStreaming(fetch(moduleUrl));
  const exports = instance.instance.exports;
  const memory = exports.memory;
  const maxChildren = 64;
  const visitsPtr = 0;
  const valueSumsPtr = 256;
  const priorsPtr = 1024;
  const visits = new Int32Array(memory.buffer, visitsPtr, maxChildren);
  const valueSums = new Float64Array(memory.buffer, valueSumsPtr, maxChildren);
  const priors = new Float64Array(memory.buffer, priorsPtr, maxChildren);

  return {
    available: true,
    moduleUrl: moduleUrl.toString(),
    selectChild(children, parentVisits, cPuct) {
      if (children.length > maxChildren) {
        throw new Error(`WASM selector supports at most ${maxChildren} children.`);
      }
      for (let index = 0; index < children.length; index += 1) {
        const child = children[index];
        visits[index] = child.visits;
        valueSums[index] = child.valueSum;
        priors[index] = child.prior;
      }
      const selected = exports.select_child(
        visitsPtr,
        valueSumsPtr,
        priorsPtr,
        children.length,
        parentVisits,
        cPuct,
      );
      return children[selected];
    },
  };
}
