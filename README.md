# dots-and-boxes-mcts

Dots and Boxes is a nostalgic children's game; possibly the second strategy game that children learn after tic-tac-toe. In this repo, we build a bot for Dots and Boxes using AlphaZero-style self-play with Monte Carlo Tree Search (MCTS).

This README is meant to be a pedagogical guide. Follow this pattern:

1. Run or inspect a small experiment.
2. Look at the evidence: games, stats, visual replays, and failure cases.
3. Read only the one or two files that explain the mechanism you are studying.

## Environment

This project uses `uv` for Python environment and dependency management. Install
`uv`, then let it create the project-local `.venv` from `pyproject.toml` and
`uv.lock`:

```bash
uv sync
uv run python -m pytest -q
```

`uv run` executes commands inside the managed project environment, so no manual
virtualenv activation is required.

## Stage 1: Random Self-Play

Goal: understand the game simulator, where the recorded game is saved, and how to view recorded games later.

- [ ] Generate a tiny batch of random games.

```bash
uv run python -m dots_boxes_mcts.self_play \
  --games 10 \
  --seed 1 \
  --out runs/random-self-play.jsonl
```

Each output line is one complete game with the board size, seed, move list, final
scores, winner, and terminal snapshot.

### Replay Viewer

- [ ] Use the local HTML replay viewer to inspect one JSONL game line visually.

```bash
uv run python -m dots_boxes_mcts.viewer
```

Then open `http://localhost:8000`, choose a file from `runs/`, enter a line
number, and step through the game.

## Stage 2: Plain MCTS

Goal: see improvement emerge from search.

Now we can let one of the bots play with MCTS. At each turn, the MCTS bot simulates a number of moves from the current board position and selects the best one. It does not learn from one game to the next.

- [ ] Run a few MCTS-vs-random batches.

The reference search code lives in `mcts.py`; `fast_mcts.py` is the Numba backend; `mcts_vs_random.py` is the batch runner that measures MCTS against random play.

```bash
uv run python -m dots_boxes_mcts.mcts_vs_random --backend numba --simulations 10 --out runs/mcts-10-vs-random-4x4.jsonl
uv run python -m dots_boxes_mcts.mcts_vs_random --backend numba --simulations 50 --out runs/mcts-50-vs-random-4x4.jsonl
uv run python -m dots_boxes_mcts.mcts_vs_random --backend numba --simulations 100 --out runs/mcts-100-vs-random-4x4.jsonl
uv run python -m dots_boxes_mcts.mcts_vs_random --backend numba --simulations 500 --out runs/mcts-500-vs-random-4x4.jsonl
```

Using `--backend numba` points the code to `fast_mcts.py`; the default `mcts.py` is the more readable alternative.
Each command prints win rate and average score margin. Observe that deeper search leads to more wins, as expected.

### Playing Against an External Bot

Now that we have a bot with a non-random strategy, we can meaningfully try to play against other bots. For dotsandboxes.org evaluation, use the dedicated Chrome-backed runner:

```bash
uv run python -m dots_boxes_mcts.dotsandboxes_org_browser_eval \
  --games 2 \
  --alternate-players \
  --simulations 2000 \
  --out runs/dotsandboxes-org/stage-2/mcts-2000-vs-dotsandboxes-org-4x4.jsonl
```

It drives a real Chrome page, plays through dotsandboxes.org's client-side
engine, blocks the site's log/high-score/analytics requests by default, and
records ordered moves from the page's game code. Without `--checkpoint`, it uses
fast Numba UCT MCTS by default; pass `--backend python` to use the readable
reference implementation.

See `DOTSANDBOXES_ORG_EXPERIMENTS.md` for the folder convention and extra
dotsandboxes.org notes.

For a broader inspection routine, see `LEARNING_CHECKLIST.md`.

## Stage 3 Training Examples

Build a tiny replayable MCTS batch, then convert its MCTS decisions into
AlphaZero-style policy/value examples:

```bash
uv run python -m dots_boxes_mcts.mcts_vs_random \
  --backend numba \
  --games 2 \
  --rows 3 \
  --cols 3 \
  --simulations 8 \
  --seed 30 \
  --out runs/stage-3-tiny-mcts.jsonl

uv run python -m dots_boxes_mcts.train \
  runs/stage-3-tiny-mcts.jsonl \
  --limit 3 \
  --preview \
  --out runs/stage-3-tiny-examples.jsonl
```

Each example includes the decision state, the MCTS visit-count policy target,
the final score-margin value target from the decision player's perspective, and
a small encoding summary for inspection.

To verify that the learning pipeline can fit a tiny batch, run the MLX
residual-conv overfit scaffold:

```bash
uv run python -m dots_boxes_mcts.train \
  runs/stage-3.1/debug-mcts-vs-random-10.jsonl \
  --limit 20 \
  --overfit-epochs 1000 \
  --learning-rate 0.001 \
  --hidden-size 64 \
  --residual-blocks 2 \
  --diagnostics-every 250 \
  --mlx-device gpu \
  --checkpoint-out runs/stage-3.1/tiny-overfit-mlx.npz
```

On Apple Silicon, `--mlx-device gpu` uses Metal. Use `--mlx-device cpu` when you
want the smallest deterministic smoke test.

## Stage 3.2 MCTS Self-Play

Generate true 4x4-dot MCTS-vs-MCTS data for both players:

```bash
uv run python -m dots_boxes_mcts.az_self_play \
  --games 100 \
  --rows 4 \
  --cols 4 \
  --simulations 25 \
  --seed 1001 \
  --out runs/stage-3.2/self-play-4x4-100.jsonl

uv run python -m dots_boxes_mcts.train \
  runs/stage-3.2/self-play-4x4-100.jsonl \
  --out runs/stage-3.2/examples-4x4-100.jsonl
```

For 4x4-dot boards, each game has 24 moves, so the example count should be
`games * 24`.

## Stage 3.3 MLX Checkpoint Training

Train a policy/value checkpoint from serialized Stage 3.2 examples:

```bash
uv run python -m dots_boxes_mcts.train \
  runs/stage-3.2/examples-4x4-1000.jsonl \
  --train-epochs 20 \
  --batch-size 256 \
  --learning-rate 0.001 \
  --hidden-size 64 \
  --residual-blocks 4 \
  --validation-fraction 0.1 \
  --diagnostics-every 5 \
  --mlx-device gpu \
  --diagnostics-out runs/stage-3.3/mlx-resconv-policy-value-4x4-1000-diagnostics.jsonl \
  --checkpoint-out runs/stage-3.3/mlx-resconv-policy-value-4x4-1000.npz
```

## Stage 3.4+ Network-Guided Search

Run PUCT-style network-guided MCTS from the initial position:

```bash
uv run python -m dots_boxes_mcts.az_mcts \
  --checkpoint runs/stage-3.3/mlx-resconv-policy-value-4x4-1000.npz \
  --rows 4 \
  --cols 4 \
  --simulations 25 \
  --mlx-device gpu
```

Generate guided self-play for the next flywheel iteration:

```bash
uv run python -m dots_boxes_mcts.az_guided_self_play \
  --checkpoint runs/stage-3.3/mlx-resconv-policy-value-4x4-1000.npz \
  --iteration 1 \
  --games 100 \
  --rows 4 \
  --cols 4 \
  --simulations 25 \
  --seed 5001 \
  --root-dirichlet-alpha 0.3 \
  --root-exploration-fraction 0.25 \
  --mlx-device gpu \
  --debug
```

By default this writes a parameter-derived JSONL path such as
`runs/stage-3.6/guided-self-play-4x4-iter001-games100-sims25.jsonl`, plus a
`.meta.json` sidecar with the full run settings. Pass `--out` only when you
want a custom filename.
