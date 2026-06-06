# PAPG External Games

Use this note to keep games against `http://www.papg.com/dab.html` replayable as
the learning checklist grows new bots.

## Folder Convention

Captured games live under `runs/papg/`, grouped by checklist stage:

- `runs/papg/stage-1/` for random/manual baselines.
- `runs/papg/stage-2/` for plain UCT MCTS games.
- `runs/papg/stage-3/` for stronger MCTS variants.
- `runs/papg/stage-4/` for AlphaZero-style experiments.

The folders are tracked, but captured `.jsonl` games stay ignored by git.

## Recording A Papg Game

Use Papg's 4x4-dot board as the default external benchmark. Record games as
JSONL with the same shape used by the local replay viewer. After a game, pass
the full move sequence in order. This example is runnable, but replace the
board-order moves with the real game:

```bash
pyenv activate data
python -m dots_boxes_mcts.external_games \
  --bot uct_mcts_10k \
  --out runs/papg/stage-2/uct-mcts-vs-papg-4x4.jsonl \
  h:0:0 h:0:1 h:0:2 v:0:0 v:0:1 v:0:2 v:0:3 \
  h:1:0 h:1:1 h:1:2 v:1:0 v:1:1 v:1:2 v:1:3 \
  h:2:0 h:2:1 h:2:2 v:2:0 v:2:1 v:2:2 v:2:3 \
  h:3:0 h:3:1 h:3:2
```

You can also record Papg URL move numbers directly:

```bash
pyenv activate data
python -m dots_boxes_mcts.external_games \
  --bot uct_mcts_10k \
  --papg-indexes \
  --out runs/papg/stage-2/uct-mcts-vs-papg-4x4.jsonl \
  1 3 5 7 9 11 13 15 17 19 21 23 \
  25 27 29 31 33 35 37 39 41 43 45 47
```

Then replay it with:

```bash
pyenv activate data
python -m dots_boxes_mcts.viewer
```

Open `http://localhost:8000` and choose the `runs/papg/...` file.

## Dedicated PAPG Runs

Prefer the Python runner for repeatable Stage 2.5 batches:

```bash
pyenv activate data
python -m dots_boxes_mcts.papg_eval \
  --games 10 \
  --simulations 50 \
  --seed 1001 \
  --request-delay 5 \
  --out runs/papg/stage-2.5/mcts-50-vs-papg-4x4.jsonl
```

This runner talks to PAPG directly, keeps requests single-threaded, waits at
least 5 seconds between requests, and writes replayable JSONL records under
`runs/papg/stage-2.5/`.

PAPG has a two-step response after a move. The initial human-move URL uses a
`/dab?.+1+...` form and may return a `Thinking...` board with no legal move
links. The browser then polls the corresponding `/dab?.+2+...` URL until PAPG's
reply is available. `dots_boxes_mcts.papg_eval` mirrors that behavior, so it is
now suitable for longer batches without relying on Codex Browser staying open.

For a cautious comparison batch:

```bash
pyenv activate data
for spec in "10 1" "57 1001" "100 2001"; do
  set -- $spec
  python -m dots_boxes_mcts.papg_eval \
    --games 10 \
    --simulations "$1" \
    --seed "$2" \
    --request-delay 5 \
    --out "runs/papg/stage-2.5/mcts-$1-vs-papg-4x4.jsonl"
done
```

For the larger 50-game version, change `--games 10` to `--games 50`. Expect it
to take hours, not minutes, because every live PAPG request is deliberately
paced. Keep it single-threaded.

Use `--debug-dir runs/papg/stage-2.5/debug-<name>` only when diagnosing a failed
run; it stores PAPG HTML responses and can grow quickly.

## Browser-Backed PAPG Runs

The browser-backed runner remains useful for smoke testing the visible PAPG
board from a Codex Browser session:

```js
const { runPapgBrowserBatch } = await import("./tools/papg_browser_runner.mjs");
await runPapgBrowserBatch({
  browser,
  games: 1,
  simulationsList: [10, 50, 100],
  requestDelayMs: 5000,
});
```

The runner uses the visible board as the source of truth, clicks exact PAPG move
links, waits between moves, and appends completed games under
`runs/papg/stage-2.5/`. Keep this for small smoke runs; use the Python runner
for larger batches so the experiment is not tied to the Codex Browser pane
lifecycle.

## Papg Move Indexes

Papg indexes edge cells in its board table from left to right, top to bottom.
The recorder can convert indexes for rectangular dot grids, including 4x4,
5x4, and 6x4.

The 4x4-dot edge indexes are:

| Papg index | Edge id |
| --- | --- |
| 1 | `h:0:0` |
| 3 | `h:0:1` |
| 5 | `h:0:2` |
| 7 | `v:0:0` |
| 9 | `v:0:1` |
| 11 | `v:0:2` |
| 13 | `v:0:3` |
| 15 | `h:1:0` |
| 17 | `h:1:1` |
| 19 | `h:1:2` |
| 21 | `v:1:0` |
| 23 | `v:1:1` |
| 25 | `v:1:2` |
| 27 | `v:1:3` |
| 29 | `h:2:0` |
| 31 | `h:2:1` |
| 33 | `h:2:2` |
| 35 | `v:2:0` |
| 37 | `v:2:1` |
| 39 | `v:2:2` |
| 41 | `v:2:3` |
| 43 | `h:3:0` |
| 45 | `h:3:1` |
| 47 | `h:3:2` |
