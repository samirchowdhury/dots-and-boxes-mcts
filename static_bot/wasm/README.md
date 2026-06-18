# WASM Backend

The browser bot keeps neural-network inference in JavaScript and uses a small
standalone WASM search kernel for PUCT child selection when the user chooses the
WASM backend.

Build the module with:

```bash
npm --prefix static_bot run build:wasm
```

That compiles `search-kernel.wat` into
`public/wasm/search-kernel.wasm`.

This is intentionally a narrow first WASM boundary. It moves the repeated child
selection math into WASM without forcing the model evaluator or tree data
structure across the JS/WASM boundary. The next deeper port would move tree
storage and batched leaf reservation into WASM while still calling the
JavaScript neural evaluator in batches.
