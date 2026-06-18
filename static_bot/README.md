# Static Bot

This is the browser-playable EpsilonZero deployment path. Training continues to
write checkpoints under `runs/ez-flywheel/`; this app only consumes exported
browser artifacts under `static_bot/public/assets/bots/`.

## Promote a Checkpoint

```bash
uv run python static_bot/tools/export_checkpoint.py \
  runs/ez-flywheel/ez-policy-value-4x4-iter542-sims2000.npz \
  --set-latest
```

That writes:

- `public/assets/bots/iterNNN/manifest.json`
- `public/assets/bots/iterNNN/weights.bin`
- `public/assets/bots/latest.json`

The app fetches `latest.json` on load, so redeploying a new checkpoint is a data
update as long as the model architecture stays stable.

## Local Preview

```bash
cd static_bot/public
python -m http.server 8787
```

Then open `http://localhost:8787`.

## Runtime Split

- UI thread: board rendering and controls.
- Web Worker: model loading, neural inference, and MCTS search.
- WASM slot: `public/src/mcts/wasm-search.js` is the swap point for a future
  compiled search backend.
