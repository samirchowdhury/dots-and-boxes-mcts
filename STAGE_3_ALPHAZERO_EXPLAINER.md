# Stage 3 AlphaZero-Style Pipeline

This stage turns the repo from "search every move from scratch" into "search,
save what search discovered, train a model, and use that model to search
better next time." The core loop is self-play -> training examples -> neural
network -> stronger MCTS -> new self-play.

## What Already Exists

`dots_boxes_mcts/game.py` is the rule engine. It owns `GameState`,
`legal_moves()`, `apply_move()`, `state_snapshot()`, scoring, extra turns after
completed boxes, terminal detection, and winners. Stage 3 should keep using this
file as the source of truth so learned players cannot quietly drift away from
the browser game's rules.

`dots_boxes_mcts/self_play.py` already writes one game per JSONL line. Each
record has fields like `rows`, `cols`, `moves`, `finalScores`, `winner`,
`terminal`, and a final `state` snapshot. `dots_boxes_mcts/mcts_vs_random.py`
extends that shape for MCTS games with `players`, `mctsPlayer`, `simulations`, and
`decisions`. Each `decisions` entry stores the turn number, player, root
`state`, selected move, and `search.stats` entries with `move`, `visits`, and
`meanValue`.

Those `decisions` records are the bridge to AlphaZero. Plain UCT currently
throws its search tree away after choosing a move. Stage 3 keeps the useful
part: from each root position, visit counts become a policy target and the final
game result becomes a value target.

## The Training Example

Each training row should describe one decision point:

- `state`: a snapshot compatible with `state_snapshot()` from `game.py`.
- `policy`: a probability distribution over legal edge ids such as `h:0:1` or
  `v:2:3`, usually normalized from MCTS visit counts.
- `value`: the final outcome from the root player's perspective, likely using
  the same score-margin scale as `terminal_value()` in `mcts.py`.
- `player`: the player to move, because the same board can mean different
  things depending on whose turn it is.

For example, if MCTS visited `h:0:0` 70 times and `v:0:0` 30 times from a
position, the policy target is roughly `{ "h:0:0": 0.7, "v:0:0": 0.3 }`. If
that player eventually wins by 3 boxes on a 3x3-dot board with 4 boxes total,
the value target is `0.75`.

The policy target and value target are both attached to the same state, but the
value is not conditioned on that exact policy distribution. The policy target
says, "from this state, search preferred these moves in these proportions." The
value target says, "from this state, for the player to move, the eventual game
outcome was this good or bad." In other words, one example is
`state -> (policy_target, value_target)`, not
`state + policy_target -> value_target`.

## The Planned Files

`dots_boxes_mcts/encoding.py` should convert a `GameState` snapshot into tensors
the model can read. Good planes for Dots and Boxes are drawn horizontal edges,
drawn vertical edges, edge owners, claimed boxes, current player, and maybe a
legal-move mask. The important rule is reversibility for debugging: you should
be able to look at an encoded example and explain which board it came from.

`dots_boxes_mcts/network.py` should define the model with two heads. The policy
head predicts which edge is promising. The value head predicts the final score
margin or win/loss value from the current player's perspective.

A sensible first architecture is a small residual convolutional network over a
board-shaped tensor. Encode the Dots and Boxes board as a grid with shape
`2 * rows - 1` by `2 * cols - 1`: dots live at even/even coordinates,
horizontal edges at even/odd coordinates, vertical edges at odd/even
coordinates, and boxes at odd/odd coordinates. Use channels for drawn edges,
edge owner relative to the current player, claimed boxes, current player,
scores or score margin, and a legal-move mask. This keeps the spatial structure
visible to the model instead of flattening the board too early.

For the first version of `network.py`, use something like this:

1. Input tensor: `C x (2 * rows - 1) x (2 * cols - 1)`.
2. Stem: `3x3` convolution with 64 channels, normalization, and ReLU.
3. Body: 4 to 6 residual blocks, each with two `3x3` convolutions.
4. Policy head: `1x1` convolution, flatten, linear layer to one logit per edge
   id, then mask illegal moves before softmax.
5. Value head: `1x1` convolution, flatten, small MLP, and `tanh` output in
   `[-1, 1]`.

This is intentionally small. On 3x3-dot, 4x4-dot, and 5x5-dot boards, the
network should learn local patterns like "this move completes a box" and
"this move creates a three-sided box" before we worry about a larger model.
If training loss falls but evaluations do not improve, increase self-play data
and search quality before making the network much bigger.

For an interactive walkthrough of this architecture, open
`docs/stage-3-architecture.html` in a browser.

`dots_boxes_mcts/train.py` should read self-play JSONL, build examples from
MCTS decisions, optimize policy loss plus value loss, and write checkpoints.
Diagnostics should include policy loss, value loss, held-out loss, and a few
human-readable positions where the top predicted moves changed.

`dots_boxes_mcts/az_mcts.py` should replace random rollouts with network-guided
search. At each root and child state, the model supplies priors for legal moves
and a value estimate. MCTS still explores and verifies through real game rules,
but the network helps it spend simulations on better candidates.

## The Loop To Watch

1. Generate self-play games with the current best searcher.
2. Save replayable JSONL under `runs/`, preserving states, moves, visit counts,
   final scores, and winners.
3. Convert each MCTS decision into `(encoded_state, policy_target, value_target)`.
4. Train a new network checkpoint.
5. Evaluate old checkpoint vs new checkpoint with the Stage 3 evaluator.
6. Replay early and late games in `dots_boxes_mcts.viewer` and look for behavior
   changes: taking boxes, delaying sacrifices, handling chains, and avoiding
   moves that hand the opponent a run.

The success signal is not just "loss went down." The useful evidence is a
triangle: better evaluation results, sensible policy changes on concrete
positions, and replayed games that show the bot making more Dots-and-Boxes-like
decisions.
