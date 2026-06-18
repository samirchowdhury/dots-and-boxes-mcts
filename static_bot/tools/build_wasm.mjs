import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import wabtFactory from "wabt";

const projectRoot = new URL("../", import.meta.url);
const sourcePath = new URL("wasm/search-kernel.wat", projectRoot);
const outputPath = new URL("public/wasm/search-kernel.wasm", projectRoot);

const wabt = await wabtFactory();
const wat = await readFile(sourcePath, "utf8");
const module = wabt.parseWat(path.basename(sourcePath.pathname), wat);
module.resolveNames();
module.validate();
const { buffer } = module.toBinary({
  log: false,
  write_debug_names: true,
});
await mkdir(new URL(".", outputPath), { recursive: true });
await writeFile(outputPath, Buffer.from(buffer));
console.log(`Wrote ${outputPath.pathname}`);
