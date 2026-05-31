# Dots and Boxes MCTS

## Environment

- Use `pyenv activate data` before running Python commands.
- The repo targets Python 3.11+.

## Commands

- Run tests with `python -m pytest -q`.
- Generate random self-play games with:
  `python -m dots_boxes_mcts.self_play --games 10 --seed 1 --out runs/random-self-play.jsonl`

## Project Notes

- Keep this repo focused on Python solver experiments. The browser game remains in `../dots-and-boxes`.
- The canonical public game rules live in `../dots-and-boxes/src/game.js`.
- Do not change Python rule behavior without checking parity against `../dots-and-boxes/fixtures/rules/positions.json`.
- Generated experiment artifacts belong under `runs/`; that directory is ignored by git.
- Start pedagogically: random self-play first, then plain UCT MCTS, then neural-network pieces.
