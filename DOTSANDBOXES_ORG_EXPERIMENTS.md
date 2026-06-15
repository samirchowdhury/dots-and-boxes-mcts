# dotsandboxes.org External Games

Use this note to keep games against `https://dotsandboxes.org/` replayable as
the learning checklist grows new bots.

The browser runner loads the public page, configures one player as the local bot
and the other as the page's built-in JavaScript engine, and records ordered moves
from the page's `gameCode` JSON. By default it blocks the site's log,
high-score, and analytics requests; the game engine itself runs client-side
after the page assets load.

## Folder Convention

Captured games live under `runs/dotsandboxes-org/`, grouped by checklist stage:

- `runs/dotsandboxes-org/stage-1/` for random/manual baselines.
- `runs/dotsandboxes-org/stage-2/` for plain UCT MCTS games.
- `runs/dotsandboxes-org/stage-3/` for stronger MCTS variants.
- `runs/dotsandboxes-org/stage-4/` for AlphaZero-style experiments.

The folders are tracked, but captured `.jsonl` games stay ignored by git.

## Replaying A Batch

Replay captured batches with:

```bash
uv run python -m dots_boxes_mcts.viewer
```

Open `http://localhost:8000` and choose the `runs/dotsandboxes-org/...` file.

## Dedicated Browser Runs

Use the Chrome-backed Python runner for repeatable live batches:

```bash
uv run python -m dots_boxes_mcts.dotsandboxes_org_browser_eval \
  --games 10 \
  --simulations 50 \
  --seed 1001 \
  --out runs/dotsandboxes-org/stage-2/mcts-50-vs-dotsandboxes-org-4x4.jsonl
```

For a cautious comparison batch:

```bash
for spec in "10 1" "57 1001" "100 2001"; do
  set -- $spec
  uv run python -m dots_boxes_mcts.dotsandboxes_org_browser_eval \
    --games 10 \
    --simulations "$1" \
    --seed "$2" \
    --out "runs/dotsandboxes-org/stage-2/mcts-$1-vs-dotsandboxes-org-4x4.jsonl"
done
```

For checkpoint evaluation with an equal player split:

```bash
uv run python -m dots_boxes_mcts.dotsandboxes_org_browser_eval \
  --checkpoint runs/stage-4/mlx-resconv-policy-value-4x4-iter016-pure-restart-sims2000.npz \
  --games 10 \
  --simulations 2000 \
  --mlx-device gpu \
  --alternate-players \
  --out runs/dotsandboxes-org/stage-4/iter016-network-guided-sims2000-vs-dotsandboxes-org-4x4.jsonl
```

The page's built-in engine uses the site's "Thinking Time" setting. The runner
defaults to `--site-think-time 0.25`; raise it when you want the page opponent
to search longer.

Use `--allow-site-telemetry` only when you explicitly want to allow the page's
normal log, high-score, and analytics requests.

## Move Indexes

dotsandboxes.org stores moves as edge numbers in `gameCode`. It lists all
horizontal edges first, row by row, then all vertical edges, column by column.

The 4x4-dot edge numbers are:

| dotsandboxes.org edge | Edge id |
| --- | --- |
| 0 | `h:0:0` |
| 1 | `h:0:1` |
| 2 | `h:0:2` |
| 3 | `h:1:0` |
| 4 | `h:1:1` |
| 5 | `h:1:2` |
| 6 | `h:2:0` |
| 7 | `h:2:1` |
| 8 | `h:2:2` |
| 9 | `h:3:0` |
| 10 | `h:3:1` |
| 11 | `h:3:2` |
| 12 | `v:0:0` |
| 13 | `v:1:0` |
| 14 | `v:2:0` |
| 15 | `v:0:1` |
| 16 | `v:1:1` |
| 17 | `v:2:1` |
| 18 | `v:0:2` |
| 19 | `v:1:2` |
| 20 | `v:2:2` |
| 21 | `v:0:3` |
| 22 | `v:1:3` |
| 23 | `v:2:3` |
