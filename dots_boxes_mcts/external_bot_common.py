from __future__ import annotations

from pathlib import Path

from dots_boxes_mcts.game import GameState, apply_move, new_game, state_snapshot


def checkpoint_bot_name(*, checkpoint: Path, simulations: int) -> str:
    return f"network_guided_mcts_{simulations}_{checkpoint.stem}"


def external_game_record(
    *,
    opponent: str,
    bot: str,
    rows: int,
    cols: int,
    moves: list[str],
    our_player: int = 0,
    source: str,
    notes: str | None = None,
) -> dict:
    if our_player not in {0, 1}:
        raise ValueError("our_player must be 0 or 1.")

    state = replay_moves(rows=rows, cols=cols, moves=moves)
    opponent_player = 1 if our_player == 0 else 0
    record = {
        "source": source,
        "opponent": opponent,
        "bot": bot,
        "rows": state.rows,
        "cols": state.cols,
        "players": {
            str(our_player): bot,
            str(opponent_player): opponent,
        },
        "ourPlayer": our_player,
        "moves": moves,
        "finalScores": [state.scores[0], state.scores[1]],
        "winner": state.winner,
        "terminal": state.terminal,
        "state": state_snapshot(state),
    }
    if notes:
        record["notes"] = notes
    return record


def replay_moves(*, rows: int, cols: int, moves: list[str]) -> GameState:
    state = new_game(rows=rows, cols=cols)
    for move in moves:
        state = apply_move(state, move)
    return state


def alternating_our_player(game_index: int) -> int:
    return game_index % 2


def summarize_external_records(records: list[dict]) -> dict:
    summary = summarize_external_record_subset(records)
    summary["byOurPlayer"] = {}
    for player in (0, 1):
        subset = [record for record in records if int(record.get("ourPlayer", 0)) == player]
        if subset:
            summary["byOurPlayer"][str(player)] = summarize_external_record_subset(subset)
    return summary


def summarize_external_record_subset(records: list[dict]) -> dict:
    if not records:
        return {
            "games": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "winRate": 0.0,
            "averageScoreMargin": 0.0,
            "strategic": summarize_records_by_perspective([]),
        }
    margins = [
        record["finalScores"][int(record.get("ourPlayer", 0))]
        - record["finalScores"][1 - int(record.get("ourPlayer", 0))]
        for record in records
    ]
    wins = sum(
        1
        for record in records
        if record["winner"] == int(record.get("ourPlayer", 0))
    )
    draws = sum(1 for record in records if record["winner"] == "draw")
    losses = len(records) - wins - draws
    return {
        "games": len(records),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "winRate": wins / len(records),
        "averageScoreMargin": sum(margins) / len(margins),
        "strategic": summarize_records_by_perspective(records),
    }


def summarize_records_by_perspective(records: list[dict]) -> dict:
    from dots_boxes_mcts.strategic_eval import summarize_strategic_records

    return summarize_strategic_records(
        records,
        perspective_player=lambda record: int(record.get("ourPlayer", 0)),
    )


def infer_opponent_reply(
    state: GameState,
    missing_moves: list[str],
    opponent_player: int = 1,
) -> list[str]:
    if not missing_moves:
        return []
    if opponent_player not in {0, 1}:
        raise ValueError("opponent_player must be 0 or 1")

    missing = set(missing_moves)
    solutions: list[list[str]] = []
    our_player = 1 - opponent_player

    def search(current_state: GameState, remaining: set[str], sequence: list[str]) -> None:
        if not remaining:
            solutions.append(sequence)
            return
        if current_state.current_player != opponent_player:
            return

        for move in sorted(remaining):
            next_state = apply_move(current_state, move)
            if next_state.current_player == our_player and len(remaining) > 1:
                continue
            search(next_state, remaining - {move}, [*sequence, move])

    search(state, missing, [])
    if not solutions:
        raise ValueError(f"Could not infer a legal opponent reply from moves: {sorted(missing)}")

    return solutions[0]
