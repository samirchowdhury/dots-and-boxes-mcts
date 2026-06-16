# Dots and Boxes Learning Checklist

Use this checklist when you come back to the repo and want to understand how the
bot is improving without reading every line of code. The recurring pattern is:

1. Run or inspect a small experiment.
2. Look at the evidence: games, stats, visual replays, and failure cases.
3. Read only the one or two files that explain the mechanism you are studying.

## Setup

- [ ] Enter the repo.

```bash
cd dots-and-boxes-mcts
```

- [ ] Create or refresh the Python environment.

```bash
uv sync
```

- [ ] Run the test suite before trusting experiment output.

```bash
uv run python -m pytest -q
```

## Stage 1: Random Self-Play

Goal: understand the game simulator. The question is: can we generate lots of
valid Dots and Boxes games?

- [ ] Generate a tiny batch of random games.

```bash
uv run python -m dots_boxes_mcts.self_play \
  --games 5 \
  --rows 3 \
  --cols 3 \
  --seed 1 \
  --out runs/random-3x3.jsonl
```

- [ ] Inspect the first JSONL record directly.

```bash
sed -n '1p' runs/random-3x3.jsonl
```

- [ ] Open the HTML replay tool and visually inspect a game.

```bash
uv run python -m dots_boxes_mcts.viewer
```

Then open:

```text
http://localhost:8000
```

Use the file dropdown, enter line `1`, and press **Load Game**.

- [ ] While replaying, check these mechanics:
  - A player gets another turn after completing a box.
  - Scores increase exactly when boxes are completed.
  - The game ends after all edges are drawn.
  - The final score equals the number of boxes: `(rows - 1) * (cols - 1)`.

- [ ] Read the core rule implementation.

```bash
sed -n '1,240p' dots_boxes_mcts/game.py
```

- [ ] Read the random self-play driver.

```bash
sed -n '1,220p' dots_boxes_mcts/self_play.py
```

- [ ] Read the tests that protect the current behavior.

```bash
sed -n '1,240p' tests/test_game.py
sed -n '1,220p' tests/test_self_play.py
```

## Stage 2: Plain UCT MCTS

Goal: see improvement emerge from search, not from learning.

- [ ] Ask Codex to implement the smallest MCTS player that can play against
      random.
- [ ] Run a few MCTS-vs-random batches.

```bash
uv run python -m dots_boxes_mcts.mcts_vs_random --backend numba --games 50 --rows 4 --cols 4 --simulations 10 --seed 1 --out runs/mcts-10-vs-random-4x4.jsonl
uv run python -m dots_boxes_mcts.mcts_vs_random --backend numba --games 50 --rows 4 --cols 4 --simulations 50 --seed 1 --out runs/mcts-50-vs-random-4x4.jsonl
uv run python -m dots_boxes_mcts.mcts_vs_random --backend numba --games 50 --rows 4 --cols 4 --simulations 100 --seed 1 --out runs/mcts-100-vs-random-4x4.jsonl
uv run python -m dots_boxes_mcts.mcts_vs_random --backend numba --games 50 --rows 4 --cols 4 --simulations 500 --seed 1 --out runs/mcts-500-vs-random-4x4.jsonl
```

Each command prints win rate and average score margin. Look for the curve:
does 500 simulations beat 100, does 100 beat 10, and where does runtime start
feeling annoying?

Important concept: `--simulations 10` does **not** mean "train an MCTS bot on
10 self-play games." Plain UCT MCTS has no training, no saved weights, and no
memory across games. It means "spend 10 temporary search simulations every time
the MCTS player needs to choose one real move." The imagined simulations are
discarded after the move is chosen.

So this command:

```bash
uv run python -m dots_boxes_mcts.mcts_vs_random \
  --backend numba \
  --games 50 \
  --rows 3 \
  --cols 3 \
  --simulations 10 \
  --seed 1 \
  --out runs/mcts-10-vs-random-3x3.jsonl
```

means: play 50 fresh games, and on each MCTS turn run 10 new imagined playouts
from the current position before picking the next move.

- [ ] Open the HTML viewer and replay a few MCTS games.

```bash
uv run python -m dots_boxes_mcts.viewer
```

Then replay a few games from `runs/mcts-*.jsonl`. Look for:

- Does MCTS take obvious boxes?
- Does it avoid handing the random player easy boxes?
- Do higher simulation counts choose visibly different moves?
- Are there games where MCTS still makes a silly sacrifice?

- [ ] Inspect the key MCTS files once they exist.

```bash
sed -n '1,260p' dots_boxes_mcts/mcts.py
sed -n '1,260p' dots_boxes_mcts/mcts_vs_random.py
```

- [ ] Ask for a move-choice explanation from one position.

```text
Pick one MCTS game where search clearly changed the move choice. Show me the
position, the selected move, visit counts, and why the move makes sense.
```

## Stage 2.5: Play The dotsandboxes.org Bot

Goal: compare search against a different hand-built bot, not just random.

Important constraint: dotsandboxes.org is a public website. Keep these batches
small and single-threaded. The runner blocks log, high-score, and analytics
requests by default; the page's opponent engine runs client-side after the page
assets load.

- [ ] Run small live dotsandboxes.org batches on the 4x4-dot board.

Use the dedicated Chrome-backed Python runner for real batches. It opens the
live page in Chrome, plays through dotsandboxes.org's client-side engine, reads
ordered moves from the page's `gameCode`, and writes replayable JSONL files.
Without `--checkpoint`, it uses fast Numba UCT MCTS by default; pass
`--backend python` to use the readable reference implementation.

```bash
uv run python -m dots_boxes_mcts.dotsandboxes_org_browser_eval \
  --games 10 \
  --simulations 10 \
  --seed 1 \
  --out runs/dotsandboxes-org/stage-2.5/mcts-10-vs-dotsandboxes-org-4x4.jsonl

uv run python -m dots_boxes_mcts.dotsandboxes_org_browser_eval \
  --games 10 \
  --simulations 57 \
  --seed 1001 \
  --out runs/dotsandboxes-org/stage-2.5/mcts-57-vs-dotsandboxes-org-4x4.jsonl

uv run python -m dots_boxes_mcts.dotsandboxes_org_browser_eval \
  --games 10 \
  --simulations 100 \
  --seed 2001 \
  --out runs/dotsandboxes-org/stage-2.5/mcts-100-vs-dotsandboxes-org-4x4.jsonl
```

For 50-game batches, change `--games 10` to `--games 50`. Keep the runs
single-threaded.

For checkpoint runs, use the same runner with `--checkpoint`, `--mlx-device`,
and `--alternate-players` when you want an equal split between local player 0
and local player 1:

```bash
uv run python -m dots_boxes_mcts.dotsandboxes_org_browser_eval \
  --checkpoint runs/stage-4/mlx-resconv-policy-value-4x4-iter016-pure-restart-sims2000.npz \
  --games 10 \
  --simulations 2000 \
  --mlx-device gpu \
  --alternate-players \
  --out runs/dotsandboxes-org/stage-4/iter016-network-guided-sims2000-vs-dotsandboxes-org-4x4.jsonl
```

Each command prints wins, draws, losses, win rate, and average score margin.
Every live game is stored as replayable JSONL under `runs/dotsandboxes-org/stage-2.5/`.

- [ ] Replay the dotsandboxes.org games in the local viewer.

```bash
uv run python -m dots_boxes_mcts.viewer
```

Then choose one of the `dotsandboxes-org/stage-2.5/*.jsonl` files and inspect
where the browser opponent takes boxes, extends chains, or punishes a bad
sacrifice.

- [ ] Summarize whether search budget changed the odds.

```text
Compare the 10, 50, and 100 simulation dotsandboxes.org batches. Give me a
table of win rate, draws, losses, average score margin, and two replay line
numbers worth watching.
```

## Stage 3: AlphaZero-Style Training

Goal: understand the feedback loop: self-play creates data, the model learns
from improved MCTS decisions, then the model guides future MCTS.

- [x] Before training, read the one-page explanation of the pipeline:
      `STAGE_3_ALPHAZERO_EXPLAINER.md`.

- [ ] Inspect a few training examples.

```text
Generate a tiny self-play training dataset and show me three examples: board
encoding, policy target, and value target.
```

## Stage 3.1: Tiny Overfit Scaffold

Goal: verify that the MLX residual-conv learning plumbing is alive before generating a large
self-play dataset. This is intentionally small and temporary: use existing
MCTS-vs-random records to check that a small residual convolutional MLX network
can memorize a handful of MCTS decision examples.

- [ ] Generate a tiny debug batch.

```bash
uv run python -m dots_boxes_mcts.mcts_vs_random \
  --backend numba \
  --games 10 \
  --rows 3 \
  --cols 3 \
  --simulations 25 \
  --seed 1 \
  --out runs/stage-3.1/debug-mcts-vs-random-10.jsonl
```

- [ ] Preview the examples.

```bash
uv run python -m dots_boxes_mcts.train \
  runs/stage-3.1/debug-mcts-vs-random-10.jsonl \
  --limit 10 \
  --preview \
  --out runs/stage-3.1/debug-examples-10.jsonl
```

Check that:

- `tensorShape` is `[8, 5, 5]` for a 3x3-dot board.
- `legalMoves` equals the number of undrawn edges.
- policy probabilities sum to `1.0`.
- policy targets only include legal moves.
- value targets are in `[-1, 1]` and use the decision player's perspective.

- [ ] Run the tiny overfit test.

```bash
uv run python -m dots_boxes_mcts.train \
  runs/stage-3.1/debug-mcts-vs-random-10.jsonl \
  --limit 20 \
  --overfit-epochs 1000 \
  --learning-rate 0.001 \
  --hidden-size 64 \
  --residual-blocks 2 \
  --diagnostics-every 250 \
  --mlx-device cpu \
  --checkpoint-out runs/stage-3.1/tiny-overfit.npz
```

The pass signal is simple: `policyKl` should fall on the tiny batch, value loss
should stay low or fall, and policy top-1 accuracy should rise. Raw policy loss
will not go to zero because MCTS visit-count targets are soft distributions; the
useful signal is the gap between policy loss and the target distribution's own
entropy. If the tiny network cannot memorize 20 examples, do not move on to
10,000 self-play games yet.

The target distribution's entropy comes directly from the MCTS visit-count
policy target. If search visits produce a policy like:

```text
{"h:0:0": 0.7, "v:0:0": 0.2, "h:1:0": 0.1}
```

then its entropy is:

```text
H(target) = -sum(p * log(p))
```

The policy loss is cross-entropy:

```text
CE(target, prediction) = -sum(target * log(prediction))
```

If the model predicts the target distribution perfectly, then
`CE(target, target) == H(target)`. So a soft target cannot drive raw policy loss
to zero. The useful remaining error is:

```text
policyKl = policyLoss - policyTargetEntropy
```

That is the distance between the model's predicted move distribution and the
MCTS visit-count target distribution.

Use `--mlx-device gpu` from a normal Apple Silicon terminal when you want MLX to
use Metal. The CPU setting is enough for this tiny smoke test.

- [ ] Read the overfit scaffold.

```bash
sed -n '1,320p' dots_boxes_mcts/train.py
```

Look for:

- how snapshots become tensors,
- how MCTS visit counts become policy vectors,
- how final score margin becomes the value target,
- how the diagnostics prove the model can fit the toy batch.

## Stage 3.2: 4x4 MCTS Self-Play Data Ramp

Goal: replace the temporary MCTS-vs-random debug source with true MCTS-vs-MCTS
self-play. Both players use the same plain UCT searcher, and every real move
records a decision with root state, selected move, visit-count policy target,
and final value target. Use 4x4-dot boards here so the first real training
checkpoint sees a slightly richer board than the 3x3-dot smoke tests.

This stage used an older plain MCTS-vs-MCTS self-play generator that has since
been removed. Use the current EpsilonZero loop in the README for new self-play.

- [ ] Convert the smoke batch into examples and preview a few.

```bash
uv run python -m dots_boxes_mcts.train \
  runs/stage-3.2/self-play-4x4-10.jsonl \
  --limit 10 \
  --preview \
  --out runs/stage-3.2/examples-4x4-10-preview.jsonl
```

Check that examples include both `player: 0` and `player: 1`.

The old ramp commands are intentionally omitted here to avoid documenting a
deleted script path.

## Stage 3.3: Train The First Real MLX Checkpoint

Goal: train on the Stage 3.2 examples with a train/validation split and save a
checkpoint. This still learns from plain MCTS targets; it is not yet used to
play moves.

- [ ] Train the first checkpoint.

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

Watch both `train` and `validation` diagnostics. A useful first checkpoint
should lower `policyKl` over epochs without validation getting dramatically
worse. The value head may be noisier because 4x4 self-play outcomes are still
generated by plain MCTS, not by a mature learned player.

- [ ] Save the diagnostics in your notes.

Look for:

- final train vs validation `policyKl`,
- final train vs validation `valueMae`,
- whether top-1 policy accuracy improves,
- whether the checkpoint file was written.

## Stage 3.4: Network-Guided MCTS

Goal: use the residual-conv checkpoint inside search. This is the first stage
where the network affects move choice. PUCT uses the policy head as priors and
the value head instead of random rollouts.

- [ ] Run one guided-search smoke test.

```bash
uv run python -m dots_boxes_mcts.ez_mcts \
  --checkpoint runs/stage-3.3/mlx-resconv-policy-value-4x4-1000.npz \
  --rows 4 \
  --cols 4 \
  --simulations 25 \
  --mlx-device gpu
```

Check that the selected move is legal and that the root stats have visit counts.

## Stage 3.5: Guided Search Evaluation

Goal: compare network-guided MCTS against baselines before trusting it to make
new training data.

The older standalone evaluator has been removed. Use `ez_checkpoint_eval` for
checkpoint-vs-checkpoint gating, and use `dotsandboxes_org_browser_eval` for the
current external-opponent check.

The useful signal is not just win rate. Inspect average score margin and replay
a few losses. If guided MCTS loses badly to plain MCTS, improve the checkpoint
before starting the flywheel.

## Stage 3.6: First Flywheel Iteration

Goal: create stronger examples with network-guided MCTS, train the next
checkpoint, and inspect whether the loop is producing useful diversity.

- [ ] Generate guided self-play.

```bash
uv run python -m dots_boxes_mcts.ez_guided_self_play \
  --checkpoint runs/stage-3.3/mlx-resconv-policy-value-4x4-1000.npz \
  --iteration 1 \
  --games 200 \
  --rows 4 \
  --cols 4 \
  --simulations 250 \
  --seed 6001 \
  --root-dirichlet-alpha 0.3 \
  --root-exploration-fraction 0.25 \
  --temperature-moves 8 \
  --sampling-temperature 1.0 \
  --mlx-device gpu \
  --debug
```

Guided self-play samples from MCTS visit counts for the first 8 moves, then
switches back to the max-visit move. The root Dirichlet noise makes search
visits vary between games; visit-count sampling turns that variation into
different played openings, which keeps the training set from collapsing onto
one deterministic opening trunk.

The preceding step infers these output paths and refuses to overwrite them unless you pass
`--overwrite`:

```text
runs/stage-3.6/guided-self-play-4x4-iter001-games200-sims250.jsonl
runs/stage-3.6/guided-self-play-4x4-iter001-games200-sims250.meta.json
```

- [ ] Convert guided games into examples.

```bash
uv run python -m dots_boxes_mcts.train \
  runs/stage-3.6/guided-self-play-4x4-iter001-games200-sims250.jsonl \
  --out runs/stage-3.6/guided-examples-4x4-iter001-games200-sims250.jsonl
```

- [ ] Train the next checkpoint from the current champion on this batch only.

```bash
uv run python -m dots_boxes_mcts.train \
  runs/stage-3.6/guided-examples-4x4-iter001-games200-sims250.jsonl \
  --init-checkpoint runs/stage-3.3/mlx-resconv-policy-value-4x4-1000.npz \
  --train-epochs 10 \
  --batch-size 256 \
  --learning-rate 0.0005 \
  --validation-fraction 0.1 \
  --diagnostics-every 5 \
  --mlx-device gpu \
  --diagnostics-out runs/stage-3.6/mlx-resconv-policy-value-4x4-iter001-guided-sims250-diagnostics.jsonl \
  --checkpoint-out runs/stage-3.6/mlx-resconv-policy-value-4x4-iter001-guided-sims250.npz
```

- [ ] Evaluate the new checkpoint against the current champion.

```bash
uv run python -m dots_boxes_mcts.ez_checkpoint_eval \
  --candidate runs/stage-3.6/mlx-resconv-policy-value-4x4-iter001-guided-sims250.npz \
  --baseline runs/stage-3.3/mlx-resconv-policy-value-4x4-1000.npz \
  --games 100 \
  --rows 4 \
  --cols 4 \
  --simulations 100 \
  --seed 7001 \
  --mlx-device gpu \
  --out runs/stage-3.6/iter001-vs-champion-sims100.jsonl
```

The evaluator alternates which checkpoint plays first. Read the printed summary
from the candidate checkpoint's perspective, then replay a few wins and losses
to understand what changed. Each game starts with two seeded random opening
moves by default so the match samples multiple positions instead of replaying
the same deterministic game for each player color. For a pure deterministic
mirror match, pass `--opening-random-plies 0`.

The first flywheel checkpoint does not need to beat Stage 3.3 immediately. A
reasonable Stage 3.6 outcome is: self-play games show diverse openings, the
candidate is not catastrophically worse, and the replay viewer reveals concrete
failure cases to improve. If the candidate does not clear the promotion bar,
leave Stage 3.3 as the champion and generate the next candidate from Stage 3.3
again.

- [ ] Or run the current EpsilonZero flywheel.

```bash
uv run python -m dots_boxes_mcts.ez_flywheel init-state \
  --random-checkpoint \
  --random-seed 1
uv run python -m dots_boxes_mcts.ez_flywheel loop \
  --iterations 3 \
  --min-win-rate 0.55 \
  --min-average-score-margin 0.0
```

This is the preferred path once you are ready to use the flywheel regularly. It
writes the ledger and iteration artifacts under `runs/ez-flywheel/`.

## Stage 3.7: Continue The Flywheel

Goal: generate new self-play from the current champion, continue optimizing the
latest training checkpoint, and evaluate whether the new challenger deserves
promotion.

The EpsilonZero flywheel keeps a tiny local ledger in
`runs/ez-flywheel/ez-flywheel-state.json` and an append-only history in
`runs/ez-flywheel/ez-flywheel-history.jsonl`. Use `loop` for normal work so you
do not have to remember the current iteration, champion checkpoint, or latest
candidate checkpoint.

- [ ] Let the flywheel run several iterations with a fixed promotion policy.

```bash
uv run python -m dots_boxes_mcts.ez_flywheel loop \
  --iterations 5 \
  --min-win-rate 0.55 \
  --min-average-score-margin 0.0
```

`loop` repeatedly generates self-play, builds examples, trains a candidate,
evaluates it against the current champion, and promotes it only when both
thresholds pass. Use the replay viewer after each iteration; a better checkpoint
should win more often, lose by smaller margins, and avoid obvious repeated
mistakes.

- [ ] Read the anchor files once they exist.

```bash
sed -n '1,260p' dots_boxes_mcts/ez_mcts.py
sed -n '1,260p' dots_boxes_mcts/ez_checkpoint_eval.py
sed -n '1,260p' dots_boxes_mcts/ez_guided_self_play.py
sed -n '1,260p' dots_boxes_mcts/encoding.py
sed -n '1,260p' dots_boxes_mcts/train.py
sed -n '1,260p' dots_boxes_mcts/ez_flywheel.py
```

- [ ] Ask for training diagnostics.

```text
Show me loss curves, old-model-vs-new-model match results, and three positions
where the learned policy changed over training.
```

- [ ] Replay self-play games from early and later checkpoints in the HTML
      viewer, looking for strategy changes rather than just final scores.

## Stage 3.8: Advanced Evaluation Strategy

Goal: add external and tactical holdouts that catch narrow self-play progress
before it becomes the promotion strategy. These metrics should explain whether
new checkpoints are learning generally useful Dots and Boxes tactics, not just
exploiting the previous checkpoint.

- [ ] Track avoidable box giveaways in every eval summary.

The `strategic` summary reports how often the evaluated player makes a
non-scoring move that creates one or more 3-sided boxes while at least one safe
non-scoring move was still available. This is a direct signal for premature
box giveaways before chain control begins.

Key fields:

- `unsafeOpenerMoves`: moves that opened a 3-sided box while a safe move existed,
- `unsafeOpenedThreeSidedBoxes`: number of 3-sided boxes created by those moves,
- `unsafeOpenerRate`: unsafe opener moves divided by non-scoring moves,
- `forcedOpenerMoves`: opener moves made when no safe move existed.

Historical backfill artifacts from the first PAPG/checkpoint review live under
`runs/stage-3.8/`. Use `uv run python -m dots_boxes_mcts.strategic_eval ...` only when
analyzing replay files that were generated before strategic metrics were added
to the normal eval summaries.

The generated suite files store the position immediately before each avoidable
opener, the move the model chose, and the number of safe/scoring/opener moves
available. Use those as fixed diagnostic sets before changing promotion rules.

- [ ] Compare checkpoints by tactical trend, not only score.

Ask whether `unsafeOpenerRate` and `unsafeOpenerPerGame` fall from early to
later checkpoints. If final score improves but avoidable opener rate stays flat,
the checkpoint may be winning for unrelated reasons and still vulnerable to
PAPG-style punishment.

## Stage 4.0: Strong-Search Restart Diagnosis

Goal: record the tactical lesson from the unsafe-opener probe before restarting
the AlphaZero-style loop.

- [ ] Treat the 50k simulation probe as the Stage 4 motivation.

On the fixed PAPG unsafe-opener suite under
`runs/stage-3.8/papg-stage3.6-unsafe-opener-positions.jsonl`, plain UCT with
random terminal rollouts behaved very differently by budget:

- `5,000` simulations still selected unsafe openers in 36.4% of trials,
- `50,000` simulations selected unsafe openers in 9.1% of trials,
- at `50,000`, safe/scoring moves received about 64.0% of root visits on
  average and unsafe openers still received about 36.0%.

Interpretation: random-rollout UCT can eventually see the premature box-giveaway
penalty, but the search teacher needs a much larger rollout budget than Stage 3
used. Stage 4 should prioritize teacher strength before asking the student
network to learn more.

## Stage 4.1: Numba-Accelerated UCT Backend

Goal: make 50k-simulation search practical enough for self-play and tactical
probes.

- [ ] Validate the fast backend against the Python rule engine.

```bash
uv run python -m pytest -q tests/test_fast_mcts.py
```

The fast backend keeps Python `GameState` snapshots as the public/replay format,
but performs rollout-heavy search with compact arrays. It must preserve:

- legal edge ordering,
- box scoring,
- extra turns after scoring,
- double-box captures,
- terminal score values.

- [ ] Run a single fast-search smoke test.

```bash
uv run python -m dots_boxes_mcts.fast_mcts \
  --rows 4 \
  --cols 4 \
  --simulations 50000 \
  --seed 1
```

- [ ] Re-run the unsafe-opener probe with the Numba backend.

```bash
uv run python -m dots_boxes_mcts.mcts_simulation_probe \
  runs/stage-3.8/papg-stage3.6-unsafe-opener-positions.jsonl \
  --inputs-are-positions \
  --backend numba \
  --simulations 50000 \
  --seeds 1,2,3 \
  --out-dir runs/stage-4/numba-unsafe-opener-probe-sims50000
```

The first acceptance target is not exact bit-for-bit equality with Python UCT,
because both searches are stochastic. The result should be in the same tactical
regime: unsafe opener selection near the Python 50k baseline and ideally at or
below 10% on the same 11-position suite.

- [ ] Record the Numba UCT unsafe-opener ladder.

Results from
`runs/stage-4/numba-unsafe-opener-probe-ladder/combined/summary.json`
on the 11-position PAPG unsafe-opener suite, with seeds `1,2,3`
(`33` trials per budget):

| simulations | unsafe selections | unsafe rate | safe/scoring rate | unsafe visit share | measured time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 12 / 33 | 36.4% | 63.6% | 51.2% | 0.75s |
| 10,000 | 9 / 33 | 27.3% | 72.7% | 48.1% | 0.73s |
| 15,000 | 8 / 33 | 24.2% | 75.8% | 45.4% | 0.84s |
| 20,000 | 7 / 33 | 21.2% | 78.8% | 43.5% | 0.95s |
| 30,000 | 6 / 33 | 18.2% | 81.8% | 40.4% | 1.18s |
| 40,000 | 3 / 33 | 9.1% | 90.9% | 37.4% | 1.42s |
| 50,000 | 1 / 33 | 3.0% | 97.0% | 35.5% | 1.66s |
| 100,000 | 0 / 33 | 0.0% | 100.0% | 25.3% | 2.88s |

Interpretation: high-simulation random-rollout UCT does become a strong enough
tactical teacher for this known failure mode. The unsafe-opener selection rate
falls steadily, crosses the rough 10% target by `40,000` simulations, is nearly
gone by `50,000`, and disappears on this run at `100,000`.

- [ ] Keep the UCT and network-guided runtime stories separate.

The Numba backend accelerates plain UCT MCTS: UCT selection plus random terminal
rollouts in compact arrays. It is not the same backend as network-guided MCTS,
which uses PUCT, policy priors, and value estimates from the MLX checkpoint.
Network-guided search has different runtime behavior because each new leaf may
require a neural-network evaluation. It benefits from checkpoint eval mode,
evaluator caching, and reusing a single search tree for simulation-budget
ladders, but the Numba UCT timing table should not be used as a direct
network-guided runtime estimate.

- [ ] Record the network-guided unsafe-opener ladder for the current champion.

Results from
`runs/stage-4/network-guided-unsafe-opener-probe-champion-iter021/reuse-full-ladder/summary.json`
using champion checkpoint
`runs/stage-3.6/mlx-resconv-policy-value-4x4-iter021-guided-sims250.npz`,
seed `1`, and one trial per suite position:

| simulations | unsafe selections | unsafe rate | safe/scoring rate | unsafe visit share | cumulative time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 3 / 11 | 27.3% | 72.7% | 38.7% | 15.0s |
| 10,000 | 3 / 11 | 27.3% | 72.7% | 31.4% | 26.1s |
| 15,000 | 2 / 11 | 18.2% | 81.8% | 25.8% | 36.2s |
| 20,000 | 2 / 11 | 18.2% | 81.8% | 22.7% | 44.8s |
| 30,000 | 2 / 11 | 18.2% | 81.8% | 20.4% | 62.2s |
| 40,000 | 2 / 11 | 18.2% | 81.8% | 17.5% | 77.4s |
| 50,000 | 2 / 11 | 18.2% | 81.8% | 15.8% | 91.4s |
| 100,000 | 1 / 11 | 9.1% | 90.9% | 10.2% | 149.8s |

Low-budget detail from
`runs/stage-4/network-guided-unsafe-opener-probe-champion-iter021/low-budget-curve/summary.json`:

| simulations | unsafe selections | unsafe rate | unsafe visit share | cumulative time |
| ---: | ---: | ---: | ---: | ---: |
| 25 | 7 / 11 | 63.6% | 54.5% | 0.16s |
| 50 | 6 / 11 | 54.5% | 55.6% | 0.37s |
| 100 | 8 / 11 | 72.7% | 53.6% | 0.71s |
| 250 | 7 / 11 | 63.6% | 51.6% | 1.53s |
| 500 | 7 / 11 | 63.6% | 50.0% | 2.73s |
| 1,000 | 6 / 11 | 54.5% | 49.1% | 4.84s |
| 1,500 | 4 / 11 | 36.4% | 45.7% | 6.87s |
| 2,000 | 3 / 11 | 27.3% | 41.8% | 8.75s |
| 3,000 | 3 / 11 | 27.3% | 39.9% | 12.01s |
| 4,000 | 3 / 11 | 27.3% | 39.5% | 14.89s |
| 5,000 | 3 / 11 | 27.3% | 39.7% | 17.60s |

Interpretation: the current champion improves meaningfully with more guided
simulations, but the curve is not the same as plain UCT. In this probe, the
guided search reaches the 10% unsafe-selection target only at `100,000`
simulations, while the unsafe visit share keeps dropping before the selected
move changes. For planning guided self-play runtimes, use the network-guided
tables rather than the Numba UCT table.

- [x] Record retained network-guided runtime optimizations across real moves.

Network-guided MCTS now reuses the retained search tree across real game moves
by default in guided self-play, guided-vs-baseline evaluation, checkpoint
matches, and guided browser-opponent evaluation. The compatibility escape hatch is
`--disable-tree-reuse`. Fresh single-position calls to `NetworkGuidedMCTS.search`
still rebuild from scratch; reusable game paths call `search_reusing_tree` and
then advance the root with the actual played move. This matters during early
self-play turns, where a sampled move may differ from the search-preferred move.

Network-guided full-game paths also use a game-level evaluator cache by default
(`500,000` entries, set `--evaluator-cache-entries 0` to disable where the CLI
exposes the knob). The cache is scoped to one game/searcher and stores network
policy/value outputs by exact `GameState`, avoiding repeated checkpoint
evaluation and repeated snapshot encoding when tree reuse or transpositions
revisit a state.

Benchmark from `runs/stage-4/optimization-benchmark/` using iter021, CPU MLX,
one 4x4 self-play game, seed `6001`, and `2,000` simulations per decision:

| mode | wall time | decisions | final score | speedup vs baseline | time reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| no tree reuse, no evaluator cache | 20.92s | 24 | 3-6 | 1.00x | 0.0% |
| tree reuse, no evaluator cache | 14.17s | 24 | 2-7 | 1.48x | 32.3% |
| tree reuse + evaluator cache | 11.84s | 24 | 3-6 | 1.77x | 43.4% |

Interpretation: the retained runtime path gives a measured `43.4%` wall-time
reduction on the 2,000-simulation guided self-play game without changing MCTS
scheduling or state representation. Tree reuse can still change a game relative
to a fresh-search baseline because retained subtree visits become part of later
decisions; evaluator caching is intended to be behavior-preserving and only
deduplicates identical network evaluations.

Regression strategy for future optimizations:

- Keep `uv run python -m pytest -q` green.
- Preserve tests that `search()` remains fresh-root, reusable search advances to
  an existing child, mismatched states reset safely, reused roots top up to the
  target budget, and sampled self-play moves advance the tree.
- Preserve tests that cached evaluator calls return identical policy/value
  results and avoid recomputing identical states.
- For runtime-only changes, benchmark both `--disable-tree-reuse` and default
  reuse on the same checkpoint, seed, device, and simulation count.

## Stage 4.2: Pure-Restart AlphaZero-Style Loop

Goal: restart the learning loop with clean Stage 4 lineage and a strong
high-simulation search teacher.

- [ ] Use the independent Stage 4 runner.

Initialize a clean Stage 4 state with a random policy/value network:

```bash
uv run python -m dots_boxes_mcts.stage4_runner init-state \
  --random-checkpoint \
  --random-seed 1
```

By default this writes the random checkpoint to:

```text
runs/stage-4/stage4-random-policy-value-4x4-seed1.npz
```

Use `--checkpoint-out path/to/random.npz` if you want an explicit filename.
Then inspect the state and the first planned commands:

```bash
uv run python -m dots_boxes_mcts.stage4_runner status
uv run python -m dots_boxes_mcts.stage4_runner next --dry-run
uv run python -m dots_boxes_mcts.stage4_runner loop --iterations 2 --dry-run
```

The Stage 4 runner writes only under `runs/stage-4/` by default:

- `stage4-state.json`,
- `stage4-history.jsonl`,
- self-play games,
- self-play metadata,
- converted training examples,
- Stage 4 initialized checkpoints,
- strategic summaries,
- tactical probe outputs.

It does not read or write the Stage 3 flywheel ledger. Stage 4.2 self-play uses
`NetworkGuidedMCTS`, so pure AlphaZero-style runs should start by creating a
random policy/value checkpoint with `init-state --random-checkpoint`. That
checkpoint is only a random initial network, not a pretrained champion. By
default, the guided self-play path uses retained tree reuse and the per-game
evaluator cache. Iteration 1 trains from the same random checkpoint that
generated self-play; later iterations initialize from the promoted Stage 4
champion by default.

The default Stage 4.2 simulation budget is `2,000` simulations per decision.
Pass `--simulations` to override it for smoke tests or larger runs.

- [ ] Run AlphaZero-style learning.

```bash
uv run python -m dots_boxes_mcts.stage4_runner loop \
  --iterations 5
```

The Stage 4 loop uses champion gating, like Stage 3.7: each completed candidate
plays a checkpoint match against the current Stage 4 champion, and the loop
promotes only if the match summary clears the configured thresholds. The loop
also reports the network-guided unsafe-opener selection rate from the tactical
probe so tactical drift is visible even though promotion is currently gated by
head-to-head strength.

Loop promotion policy:

- Default gate: promote if `winRate >= 0.55` and
  `averageScoreMargin >= 0.0` against the current Stage 4 champion.
- Override the thresholds with `--min-win-rate` and
  `--min-average-score-margin`.
- The unsafe-opener selection rate is reported from the tactical probe, but it
  is not yet a promotion gate.
- If promoted, the candidate becomes `championCheckpoint` in
  `stage4-state.json`.
- If rejected, `championCheckpoint` remains unchanged, but
  `latestCandidateCheckpoint` remains the rejected candidate. The next loop
  iteration continues self-play and training from that latest candidate while
  still gating against the unchanged champion.
- If any command in an iteration fails, the loop stops before promotion because
  no completed candidate is recorded.

For a human-gated run, replace `loop` with one `next` call at a time, inspect
the strategic summary, tactical probe, and replayed games, then promote inside
Stage 4 only:

```bash
uv run python -m dots_boxes_mcts.stage4_runner next \
  --games 25 \
  --mlx-device gpu

uv run python -m dots_boxes_mcts.stage4_runner promote \
  --iteration 1 \
  --reason "cleared Stage 4 tactical and checkpoint review"
```

- [ ] Benchmark a trained Stage 4 checkpoint against dotsandboxes.org.

After training the Stage 4 checkpoint through iteration 16, run the live
browser-backed dotsandboxes.org match with an equal player-side split:

```bash
uv run python -m dots_boxes_mcts.dotsandboxes_org_browser_eval \
  --checkpoint runs/stage-4/mlx-resconv-policy-value-4x4-iter021-pure-restart-sims2000.npz \
  --games 8 \
  --simulations 2000 \
  --mlx-device gpu \
  --alternate-players \
  --site-think-time 0.1 \
  --opening-top-k 4 \
  --out runs/dotsandboxes-org/stage-4/iter021-network-guided-sims2000-vs-dotsandboxes-org-4x4-games10-think0p1.jsonl
```

Observed result for `iter016` against the previous live PAPG benchmark:

- 50 completed games, all terminal and replay-valid.
- Equal side split: 25 games as player 0, 25 games as player 1.
- Overall record: 35 wins, 15 losses, 0 draws.
- Win rate: 70%.
- Average score margin: +1.76 boxes.
- By side: 12-13 as player 0, 23-2 as player 1.

## Human Inspection Rhythm

- [ ] After each feature, inspect the command, output file, and one relevant
      function.
- [ ] After each experiment, inspect the table/plot and two or three concrete
      games in the HTML viewer.
- [ ] After each jump in strength, ask:
  - What changed mechanically?
  - What evidence says it improved?
  - What failure cases remain?
- [ ] Before moving from MCTS to AlphaZero-style training, make sure you can
      explain these ideas in your own words:
  - random self-play,
  - legal moves and state transitions,
  - extra turns after scoring,
  - UCT selection,
  - rollouts or value estimates,
  - win-rate evaluation,
  - policy targets and value targets.
