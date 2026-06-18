import { readFile } from "node:fs/promises";

const wasm = await readFile("static_bot/public/wasm/search-kernel.wasm");
const { instance } = await WebAssembly.instantiate(wasm);
const { memory, select_child: selectChild } = instance.exports;

const visitsPtr = 0;
const valueSumsPtr = 256;
const priorsPtr = 1024;
const visits = new Int32Array(memory.buffer, visitsPtr, 4);
const valueSums = new Float64Array(memory.buffer, valueSumsPtr, 4);
const priors = new Float64Array(memory.buffer, priorsPtr, 4);

visits.set([10, 1, 0, 3]);
valueSums.set([2.5, 0.8, 0, -0.2]);
priors.set([0.2, 0.3, 0.4, 0.1]);

const selected = selectChild(visitsPtr, valueSumsPtr, priorsPtr, 4, 14, 1.5);
console.log(JSON.stringify({ selected }));

if (selected !== 2) {
  throw new Error(`Expected child 2, got ${selected}`);
}
