# Dots and Boxes Learning Checklist

Use this checklist when you come back to the repo and want to understand how the
bot is improving without reading every line of code. The recurring pattern is:

1. Run or inspect a small experiment.
2. Look at the evidence: games, stats, visual replays, and failure cases.
3. Read only the one or two files that explain the mechanism you are studying.

## Setup

- [ ] Enter the repo.

```bash
cd /Users/samirchowdhury/dots-and-boxes-mcts
```

- [ ] Activate the Python environment before running Python commands.

```bash
pyenv activate data
```

- [ ] Run the test suite before trusting experiment output.

```bash
python -m pytest -q
```

## Stage 1: Random Self-Play

Goal: understand the game simulator. The question is: can we generate lots of
valid Dots and Boxes games?

- [ ] Generate a tiny batch of random games.

```bash
python -m dots_boxes_mcts.self_play \
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
python -m dots_boxes_mcts.viewer
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
python -m dots_boxes_mcts.evaluate --games 50 --rows 4 --cols 4 --simulations 10 --seed 1 --out runs/mcts-10-vs-random-4x4.jsonl
python -m dots_boxes_mcts.evaluate --games 50 --rows 4 --cols 4 --simulations 50 --seed 1 --out runs/mcts-50-vs-random-4x4.jsonl
python -m dots_boxes_mcts.evaluate --games 50 --rows 4 --cols 4 --simulations 100 --seed 1 --out runs/mcts-100-vs-random-4x4.jsonl
python -m dots_boxes_mcts.evaluate --games 50 --rows 4 --cols 4 --simulations 500 --seed 1 --out runs/mcts-500-vs-random-4x4.jsonl
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
python -m dots_boxes_mcts.evaluate \
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
python -m dots_boxes_mcts.viewer
```

Then replay a few games from `runs/mcts-*.jsonl`. Look for:

- Does MCTS take obvious boxes?
- Does it avoid handing the random player easy boxes?
- Do higher simulation counts choose visibly different moves?
- Are there games where MCTS still makes a silly sacrifice?

- [ ] Inspect the key MCTS files once they exist.

```bash
sed -n '1,260p' dots_boxes_mcts/mcts.py
sed -n '1,260p' dots_boxes_mcts/evaluate.py
```

- [ ] Ask for a move-choice explanation from one position.

```text
Pick one MCTS game where search clearly changed the move choice. Show me the
position, the selected move, visit counts, and why the move makes sense.
```

## Stage 2.5: Play The PAPG Bot

Goal: compare search against a different hand-built bot, not just random.

Important constraint: PAPG is a public website. Keep these batches small,
single-threaded, and deliberately paced. Do not run tight request loops. The
helper defaults to a 5-second delay between requests; only lower it if you have
a good reason.

- [ ] Run small live PAPG batches on the 4x4-dot board.

Use the dedicated Python runner for real batches. It mirrors PAPG's browser
flow, including the `Thinking...` poll step, waits between live requests, and
writes replayable JSONL files.

```bash
python -m dots_boxes_mcts.papg_eval \
  --games 10 \
  --simulations 10 \
  --seed 1 \
  --request-delay 5 \
  --out runs/papg/stage-2.5/mcts-10-vs-papg-4x4.jsonl

python -m dots_boxes_mcts.papg_eval \
  --games 10 \
  --simulations 57 \
  --seed 1001 \
  --request-delay 5 \
  --out runs/papg/stage-2.5/mcts-57-vs-papg-4x4.jsonl

python -m dots_boxes_mcts.papg_eval \
  --games 10 \
  --simulations 100 \
  --seed 2001 \
  --request-delay 5 \
  --out runs/papg/stage-2.5/mcts-100-vs-papg-4x4.jsonl
```

For 50-game batches, change `--games 10` to `--games 50`. Keep the runs
single-threaded and leave `--request-delay 5` in place.

The Codex Browser runner is still useful for a one-game visible-board smoke
test when you specifically want Codex to verify the live page through the
in-app browser. Do not paste this into your shell or Python prompt. It is
JavaScript for Codex's browser automation environment; the easiest way to use
it is to ask Codex something like:

```text
Use the Codex Browser plugin to run one PAPG smoke game with
tools/papg_browser_runner.mjs.
```

Codex will open/connect the in-app browser, load this module, and run:

```js
const { runPapgBrowserBatch } = await import("./tools/papg_browser_runner.mjs");
await runPapgBrowserBatch({
  browser,
  games: 1,
  simulationsList: [10, 50, 100],
  requestDelayMs: 5000,
});
```

For normal experiments, prefer the Python `papg_eval` commands above; they run
from a regular terminal and do not depend on the Codex Browser pane staying
alive.

Each command prints wins, draws, losses, win rate, and average score margin.
Every live game is stored as replayable JSONL under `runs/papg/stage-2.5/`.

- [ ] Replay the PAPG games in the local viewer.

```bash
python -m dots_boxes_mcts.viewer
```

Then choose one of the `papg/stage-2.5/*.jsonl` files and inspect where PAPG
takes boxes, extends chains, or punishes a bad sacrifice.

- [ ] Summarize whether search budget changed the odds.

```text
Compare the 10, 50, and 100 simulation PAPG batches. Give me a table of win
rate, draws, losses, average score margin, and two replay line numbers worth
watching.
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
python -m dots_boxes_mcts.evaluate \
  --games 10 \
  --rows 3 \
  --cols 3 \
  --simulations 25 \
  --seed 1 \
  --out runs/stage-3.1/debug-mcts-vs-random-10.jsonl
```

- [ ] Preview the examples.

```bash
python -m dots_boxes_mcts.train \
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
python -m dots_boxes_mcts.train \
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

- [ ] Run a 10-game smoke test.

```bash
python -m dots_boxes_mcts.az_self_play \
  --games 10 \
  --rows 4 \
  --cols 4 \
  --simulations 25 \
  --seed 1 \
  --out runs/stage-3.2/self-play-4x4-10.jsonl
```

Check the printed summary. On a 4x4-dot board, `averageDecisionsPerGame` should
be `24.0`, because there are 24 edges and both MCTS players record every move.

- [ ] Convert the smoke batch into examples and preview a few.

```bash
python -m dots_boxes_mcts.train \
  runs/stage-3.2/self-play-4x4-10.jsonl \
  --limit 10 \
  --preview \
  --out runs/stage-3.2/examples-4x4-10-preview.jsonl
```

Check that examples include both `player: 0` and `player: 1`.

- [ ] Ramp to 100 games.

```bash
python -m dots_boxes_mcts.az_self_play \
  --games 100 \
  --rows 4 \
  --cols 4 \
  --simulations 25 \
  --seed 1001 \
  --out runs/stage-3.2/self-play-4x4-100.jsonl

python -m dots_boxes_mcts.train \
  runs/stage-3.2/self-play-4x4-100.jsonl \
  --out runs/stage-3.2/examples-4x4-100.jsonl
```

Expected example count for 4x4 dots is `games * 24`, so 100 games should produce
2,400 training examples.

- [ ] Ramp to 1,000 games.

```bash
python -m dots_boxes_mcts.az_self_play \
  --games 1000 \
  --rows 4 \
  --cols 4 \
  --simulations 25 \
  --seed 2001 \
  --out runs/stage-3.2/self-play-4x4-1000.jsonl

python -m dots_boxes_mcts.train \
  runs/stage-3.2/self-play-4x4-1000.jsonl \
  --out runs/stage-3.2/examples-4x4-1000.jsonl
```

Expected example count is 24,000. Inspect file size and runtime before moving
to 10,000 games or larger boards.

## Stage 3.3: Train The First Real MLX Checkpoint

Goal: train on the Stage 3.2 examples with a train/validation split and save a
checkpoint. This still learns from plain MCTS targets; it is not yet used to
play moves.

- [ ] Train the first checkpoint.

```bash
python -m dots_boxes_mcts.train \
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
python -m dots_boxes_mcts.az_mcts \
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

- [ ] Evaluate against random.

```bash
python -m dots_boxes_mcts.az_evaluate \
  --checkpoint runs/stage-3.3/mlx-resconv-policy-value-4x4-1000.npz \
  --opponent random \
  --games 50 \
  --rows 4 \
  --cols 4 \
  --simulations 25 \
  --seed 3001 \
  --mlx-device gpu \
  --out runs/stage-3.5/guided-vs-random-4x4-50.jsonl
```

- [ ] Evaluate against plain MCTS at the same simulation count.

```bash
python -m dots_boxes_mcts.az_evaluate \
  --checkpoint runs/stage-3.3/mlx-resconv-policy-value-4x4-1000.npz \
  --opponent plain_mcts \
  --games 50 \
  --rows 4 \
  --cols 4 \
  --simulations 25 \
  --opponent-simulations 25 \
  --seed 4001 \
  --mlx-device gpu \
  --out runs/stage-3.5/guided-vs-plain-mcts-25-4x4-50.jsonl
```

The useful signal is not just win rate. Inspect average score margin and replay
a few losses. If guided MCTS loses badly to plain MCTS, improve the checkpoint
before starting the flywheel.

## Stage 3.6: First Flywheel Iteration

Goal: create stronger examples with network-guided MCTS, train the next
checkpoint, and inspect whether the loop is producing useful diversity.

- [ ] Generate guided self-play.

```bash
python -m dots_boxes_mcts.az_guided_self_play \
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
python -m dots_boxes_mcts.train \
  runs/stage-3.6/guided-self-play-4x4-iter001-games200-sims250.jsonl \
  --out runs/stage-3.6/guided-examples-4x4-iter001-games200-sims250.jsonl
```

- [ ] Train the next checkpoint from the current champion on this batch only.

```bash
python -m dots_boxes_mcts.train \
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
python -m dots_boxes_mcts.az_checkpoint_eval \
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

- [ ] Or run the same first iteration through the tracked pipeline runner.

```bash
python -m dots_boxes_mcts.az_flywheel init-state \
  --champion-checkpoint runs/stage-3.3/mlx-resconv-policy-value-4x4-1000.npz
python -m dots_boxes_mcts.az_flywheel next --dry-run
python -m dots_boxes_mcts.az_flywheel next
```

This is the preferred path once you are ready to use the flywheel regularly.
It writes the ledger to `runs/az-flywheel/`, keeps iteration artifacts in
`runs/stage-3.6/`, and records the candidate evaluation as pending after the
run completes. Use `init-state --overwrite` only when you intentionally want to
reset the ledger state; use `next --overwrite` only when you intentionally want
to replace existing Stage 3.6 iteration outputs.

## Stage 3.7: Continue The Flywheel

Goal: generate new self-play from the current champion, continue optimizing the
latest training checkpoint, and evaluate whether the new challenger deserves
promotion.

The tracked pipeline runner keeps a tiny local ledger in
`runs/az-flywheel/flywheel-state.json` and an append-only history in
`runs/az-flywheel/flywheel-history.jsonl`. Here `az` means
AlphaZero-style: self-play guided by a policy/value network, training from that
self-play, and champion-gated checkpoint evaluation. The ledger is stage-neutral
because it tracks the flywheel process across Stage 3.6, Stage 3.7, and future
iterations; the generated games, examples, checkpoints, and evaluations still
default to `runs/stage-3.6/`.

The state records the next iteration, the current champion checkpoint, the
latest candidate checkpoint, and the last evaluation summary. Use this mode for
normal work so you do not have to remember which iteration, champion, or
promotion decision is current.

- [ ] Initialize the tracked flywheel state.

```bash
python -m dots_boxes_mcts.az_flywheel init-state \
  --champion-checkpoint runs/stage-3.3/mlx-resconv-policy-value-4x4-1000.npz
```

Use `--overwrite` only when you intentionally want to reset the tracked state.
This does not delete checkpoints or replay files; it only rewrites the ledger
state file.

- [ ] Check what the runner thinks is current.

```bash
python -m dots_boxes_mcts.az_flywheel status
```

- [ ] Dry-run the next tracked iteration.

```bash
python -m dots_boxes_mcts.az_flywheel next --dry-run
```

- [ ] Run the next tracked iteration.

```bash
python -m dots_boxes_mcts.az_flywheel next
```

After `next` finishes, the runner reads the checkpoint-match replay file,
records the candidate evaluation summary, advances `nextIteration`, and leaves
the promotion decision as `pending`.

- [ ] Promote a candidate that clears the bar.

```bash
python -m dots_boxes_mcts.az_flywheel promote \
  --iteration 1 \
  --reason "cleared promotion bar in checkpoint match"
```

Suggested first promotion bar:

- at least 100 evaluation games against the current champion,
- win rate at or above 55%,
- average score margin at or above 0.0,
- replayed wins and losses do not show a new obvious failure mode,
- performance against plain MCTS has not regressed badly.

- [ ] Or reject a candidate and keep the current champion.

```bash
python -m dots_boxes_mcts.az_flywheel reject \
  --iteration 1 \
  --reason "evaluation exposed repeated opening mistakes"
```

`reject` is a bookkeeping command, not a destructive command. It reads that
iteration's evaluation file, writes a `rejected` decision and summary into the
ledger, leaves `championCheckpoint` unchanged, and keeps `nextIteration`
advanced. The candidate checkpoint and replay files stay on disk. The next
`python -m dots_boxes_mcts.az_flywheel next` will still use the current champion
for self-play and evaluation, while training defaults to the previous iteration
candidate unless you override `--init-checkpoint`.

The loop is intentionally champion-gated for data generation and evaluation: the
current champion controls self-play and the evaluation baseline. Training still
defaults to the previous iteration candidate after iteration 1, even if that
candidate was not promoted. Pass `--init-checkpoint` to `next` only when a
checkpoint has a nonstandard name or when you intentionally want to restart
optimization from a different checkpoint.

- [ ] Or let the flywheel run several iterations with a fixed promotion policy.

```bash
python -m dots_boxes_mcts.az_flywheel loop \
  --iterations 5 \
  --min-win-rate 0.55 \
  --min-average-score-margin 0.0
```

`loop` repeatedly runs the tracked `next` pipeline, records the checkpoint-match
summary, and then automatically promotes or rejects the candidate. A candidate
is promoted only if both thresholds pass. If it is rejected, the champion stays
unchanged for the next self-play and evaluation baseline, but training still
continues from that rejected candidate by default. This preserves the current
flywheel behavior: champion-gated data generation and evaluation, continuous
optimization from the latest candidate.

The old explicit form still works for one-off runs:

```bash
python -m dots_boxes_mcts.az_flywheel \
  --iteration 2 \
  --champion-checkpoint runs/stage-3.3/mlx-resconv-policy-value-4x4-1000.npz \
  --dry-run
```

Prefer the tracked `next`/`status`/`promote`/`reject` workflow for real
experiments. Use the replay viewer after each iteration; a better checkpoint
should win more often, lose by smaller margins, and avoid obvious repeated
mistakes.

- [ ] Read the anchor files once they exist.

```bash
sed -n '1,260p' dots_boxes_mcts/az_self_play.py
sed -n '1,260p' dots_boxes_mcts/az_mcts.py
sed -n '1,260p' dots_boxes_mcts/az_evaluate.py
sed -n '1,260p' dots_boxes_mcts/az_checkpoint_eval.py
sed -n '1,260p' dots_boxes_mcts/az_guided_self_play.py
sed -n '1,260p' dots_boxes_mcts/encoding.py
sed -n '1,260p' dots_boxes_mcts/network.py
sed -n '1,260p' dots_boxes_mcts/train.py
sed -n '1,260p' dots_boxes_mcts/az_mcts.py
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

- [ ] Backfill the metric on existing replay files.

```bash
python -m dots_boxes_mcts.strategic_eval \
  runs/papg/stage-3.6/archive-pre-ledger-20260608/iter001-guided-sims250-vs-papg-browser-model-second-retry.jsonl \
  runs/papg/stage-3.6/archive-pre-ledger-20260608/iter007-guided-sims250-vs-papg-browser-model-second-retry.jsonl \
  runs/papg/stage-3.6/archive-pre-ledger-20260608/iter007-guided-sims250-vs-papg-browser-model-second.jsonl \
  runs/papg/stage-3.6/archive-pre-ledger-20260608/iter007-guided-sims250-vs-papg-browser.jsonl \
  runs/papg/stage-3.6/archive-pre-ledger-20260608/stage3.3-1000-sims250-vs-papg-browser-model-second-retry.jsonl \
  runs/papg/stage-3.6/archive-pre-ledger-20260608/stage3.3-1000-sims250-vs-papg-browser-model-second.jsonl \
  runs/papg/stage-3.6/iter016-guided-sims250-vs-papg-browser-model-second.jsonl \
  --summary-out runs/stage-3.8/papg-stage3.6-strategic-summary.json \
  --suite-out runs/stage-3.8/papg-stage3.6-unsafe-opener-positions.jsonl
```

The suite file stores the position immediately before each avoidable opener,
the move the model chose, and the number of safe/scoring/opener moves available.
Use it as a fixed diagnostic set before changing promotion rules.

- [ ] Compare checkpoints by tactical trend, not only score.

Ask whether `unsafeOpenerRate` and `unsafeOpenerPerGame` fall from early to
later checkpoints. If final score improves but avoidable opener rate stays flat,
the checkpoint may be winning for unrelated reasons and still vulnerable to
PAPG-style punishment.

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
