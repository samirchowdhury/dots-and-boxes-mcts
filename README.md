# Dots and Boxes MCTS

Python experiments for Dots and Boxes self-play and future MCTS work.

This repo starts deliberately small: it duplicates the browser game's rules in
Python, checks parity against canonical fixtures from `../dots-and-boxes`, and
can generate random self-play games as JSONL.

## Environment

```bash
pyenv activate data
python -m pytest -q
```

## Random Self-Play

```bash
python -m dots_boxes_mcts.self_play \
  --games 10 \
  --seed 1 \
  --out runs/random-self-play.jsonl
```

Each output line is one complete game with the board size, seed, move list, final
scores, winner, and terminal snapshot.
