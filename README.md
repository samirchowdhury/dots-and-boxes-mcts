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

## Plain UCT MCTS

Run a single UCT search from the initial position:

```bash
uv run python -m dots_boxes_mcts.mcts --rows 3 --cols 3 --simulations 100 --seed 1
```

Evaluate MCTS against a random player and save replayable games:

```bash
uv run python -m dots_boxes_mcts.evaluate \
  --games 10 \
  --rows 3 \
  --cols 3 \
  --simulations 100 \
  --seed 1 \
  --out runs/mcts-vs-random-3x3.jsonl
```

Each MCTS game record includes the normal replay fields plus `decisions`, a list
of MCTS turns with the root state, selected move, child visit counts, and mean
values from the player-to-move perspective.


## External Bot Games

Games played against external bots, including PAPG, can be recorded in the same
JSONL replay format:

```bash
uv run python -m dots_boxes_mcts.external_games \
  --bot uct_mcts_10k \
  --papg-indexes \
  --out runs/papg/stage-2/uct-mcts-vs-papg-4x4.jsonl \
  1 3 5 7 9 11 13 15 17 19 21 23 \
  25 27 29 31 33 35 37 39 41 43 45 47
```

See `PAPG_EXPERIMENTS.md` for the folder convention and Papg move-index map.

For live PAPG evaluation, use the dedicated Chrome-backed runner:

```bash
uv run python -m dots_boxes_mcts.papg_browser_eval \
  --games 10 \
  --simulations 50 \
  --out runs/papg/stage-2.5/mcts-50-vs-papg-4x4.jsonl
```

It drives a real Chrome page, clicks the local bot's moves, and records PAPG's
actual board replies.

For a broader inspection routine, see `LEARNING_CHECKLIST.md`.

## Stage 3 Training Examples

Build a tiny replayable MCTS batch, then convert its MCTS decisions into
AlphaZero-style policy/value examples:

```bash
uv run python -m dots_boxes_mcts.evaluate \
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
