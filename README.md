# dots-and-boxes-mcts

Dots and Boxes is a nostalgic children's game; possibly the second strategy game that children learn after tic-tac-toe. In this repo, we build a bot for Dots and Boxes using AlphaZero-style self-play with Monte Carlo Tree Search (MCTS).

This README is meant to be a pedagogical guide. Follow this pattern:

1. Run or inspect a small experiment.
2. Look at the evidence: games, stats, visual replays, and failure cases.
3. Read only the one or two files that explain the mechanism you are studying.

**Notation:** We will use AZ and AGZ as shorthand for AlphaZero and AlphaGo Zero, respectively. Note that these are different works with slightly different implementations. The ELF OpenGo[^1] paper explains these differences well.

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

Now that we have a bot with a non-random strategy, we can meaningfully try to play against other bots. The bot at [dotsandboxes.org](https://dotsandboxes.org/) is a powerful opponent, and its game engine runs client side so we can use it for evaluation without spamming the server:

```bash
uv run python -m dots_boxes_mcts.dotsandboxes_org_browser_eval \
  --games 2 \
  --alternate-players \
  --simulations 2000 \
  --out runs/dotsandboxes-org/stage-2/mcts-2000-vs-dotsandboxes-org-4x4.jsonl
```

The runner drives a real Chrome page, plays through dotsandboxes.org's client-side
engine, blocks the site's log/high-score/analytics requests by default, and
records ordered moves from the page's game code.

See `DOTSANDBOXES_ORG_EXPERIMENTS.md` for the folder convention and extra
dotsandboxes.org notes.

## Stage 3: EpsilonZero

### Algorithm

EpsilonZero is the name for our tiny AlphaZero-inspired bot. Let's recall the AlphaZero algorithm:

1. Initialize an untrained network.
2. Start a game.
3. At each time step `t` of the game, perform MCTS simulations.
    - For each simulation, perform network-guided search.
        - In the selection phase, we start at the root `s_t` of the game tree and select children (using the statistics of the tree) until a leaf node is reached.
        - The network runs an "evaluate and expand" step by taking the leaf node `s_L` as input, generating a policy vector `p` and a value scalar `v`, and initializing child nodes with prior probabilities from the policy vector.
        - The visit counts and values are updated along the traversed path using the value `v`.
        - The next simulation starts again from the root and may now traverse into the newly initialized children.
    - After all simulations are complete, AlphaZero selects a move based on the updated tree statistics. The normalized visit counts from the root `s_t` are saved as a policy target `π_t` for later.

4. At the end of each game, we get a score `z_t` for each `s_t` (final outcome from the perspective of the player to move at that state). Here `s_t` is the state we had at time step `t`. The network parameters `θ` are updated so that `p_θ(s_t)->π_t` and `v_θ(s_t)->z_t`.
5. Repeat steps 2-4.

### Training

- [ ] Initialize a random policy/value network:

```bash
uv run python -m dots_boxes_mcts.ez_flywheel init-state \
  --random-checkpoint \
  --random-seed 1
```

- [ ] Run a few EpsilonZero flywheel iterations.

```bash
uv run python -m dots_boxes_mcts.ez_flywheel loop --iterations 3
```

Or run whole iterations until a wall-clock budget is reached:

```bash
uv run python -m dots_boxes_mcts.ez_flywheel loop --duration 8h
```

The flywheel performs self-play games, trains the network, and advances to the latest checkpoint after each completed iteration.
It tracks results and training state in a small ledger under `runs/ez-flywheel/`.
Rerunning the command will automatically resume self-play and training from the latest checkpoint.

### Evaluation

Evaluate a trained EpsilonZero checkpoint against the browser bot at
[dotsandboxes.org](https://dotsandboxes.org/):

```bash
ITER=001
THINK=0.25
THINK_TAG=${THINK/./p}
uv run python -m dots_boxes_mcts.dotsandboxes_org_browser_eval \
  --checkpoint runs/ez-flywheel/ez-policy-value-4x4-iter${ITER}-sims2000.npz \
  --games 2 \
  --simulations 2000 \
  --mlx-device gpu \
  --alternate-players \
  --site-think-time "$THINK" \
  --out runs/dotsandboxes-org/ez-flywheel/iter${ITER}-vs-dotsandboxes-org-4x4-think${THINK_TAG}.jsonl
```

Use the latest checkpoint from `runs/ez-flywheel/`. `--alternate-players`
splits games across first and second player, which matters a lot in Dots and
Boxes.

The eval runner writes per-game records to `--out` and prints the aggregate
summary to the terminal. To recompute the summary later from the JSONL file:

```bash
uv run python -m dots_boxes_mcts.summarize_external_eval \
  runs/dotsandboxes-org/ez-flywheel/iter${ITER}-vs-dotsandboxes-org-4x4-think${THINK_TAG}.jsonl
```

### Port to C++

In my experience, training to even 90+ iterations is not sufficient to overcome the dotsandboxes.org bot.
Porting to C++ allows us to get a 30x speedup, which is crucial for getting sufficient self-play.

Build the optional C++ network-guided MCTS backend for the active Python
environment:

```bash
uv run python -m dots_boxes_mcts.build_fast_ez_mcts
```

If this is a fresh flywheel run, initialize the first training checkpoint once:

```bash
uv run python -m dots_boxes_mcts.ez_flywheel init-state \
  --random-checkpoint \
  --random-seed 1
```

Run the C++ flywheel for a set duration. Tree reuse is off by default; this uses fresh per-move C++ searches with batched leaf evaluation and virtual loss.

```bash
uv run python -m dots_boxes_mcts.ez_flywheel loop \
  --duration 8h \
  --mcts-backend cpp \
  --mcts-batch-size 8 \
  --virtual-loss 1.0 \
  --mlx-device gpu \
  --simulations 2000
```

Check progress or compare the Python and C++ search paths:

```bash
uv run python -m dots_boxes_mcts.ez_flywheel status

uv run python -m dots_boxes_mcts.profile_ez_mcts \
  --backend both \
  --rows 4 \
  --cols 4 \
  --simulations 200 \
  --repeat 3 \
  --batch-size 8 \
  --virtual-loss 1.0
```

Evaluate a trained checkpoint against dotsandboxes.org with the C++ search path:

```bash
ITER=001
THINK=0.25
THINK_TAG=${THINK/./p}
uv run python -m dots_boxes_mcts.dotsandboxes_org_browser_eval \
  --checkpoint runs/ez-flywheel/ez-policy-value-4x4-iter${ITER}-sims2000.npz \
  --games 2 \
  --simulations 2000 \
  --mcts-backend cpp \
  --mcts-batch-size 8 \
  --virtual-loss 1.0 \
  --mlx-device gpu \
  --alternate-players \
  --site-think-time "$THINK" \
  --out runs/dotsandboxes-org/ez-flywheel/iter${ITER}-cpp-vs-dotsandboxes-org-4x4-think${THINK_TAG}.jsonl
```

## Resources

[^1]: Tian, Yuandong, et al. "Elf opengo: An analysis and open reimplementation of alphazero." International conference on machine learning. PMLR, 2019.
