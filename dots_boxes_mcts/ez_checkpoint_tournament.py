from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from dots_boxes_mcts.ez_checkpoint_eval import play_checkpoint_match_game

DEFAULT_CHECKPOINT_DIR = Path("runs/ez-flywheel")
DEFAULT_CHECKPOINT_PATTERN = "ez-policy-value-4x4-iter*-sims2000.npz"
DEFAULT_OUT_DIR = Path("runs/checkpoint-tournaments/ez-flywheel-all-pairs")
DEFAULT_INCLUDE_ITERS = (542,)
DEFAULT_SAMPLE_SIZE = 60
DEFAULT_SIMULATIONS = 500
DEFAULT_SEED = 900_001
MEANINGFUL_RATING_DROP = 50.0

ITERATION_RE = re.compile(r"iter(\d+)")


@dataclass(frozen=True)
class CheckpointEntry:
    iteration: int
    path: Path


@dataclass(frozen=True)
class TournamentGame:
    pair_index: int
    game_in_pair: int
    checkpoint_a: CheckpointEntry
    checkpoint_b: CheckpointEntry
    candidate_player: int
    seed: int

    @property
    def pair_key(self) -> str:
        return f"iter={self.checkpoint_a.iteration:03d}|iter={self.checkpoint_b.iteration:03d}"

    @property
    def game_key(self) -> str:
        return f"{self.pair_key}|game={self.game_in_pair}|candidatePlayer={self.candidate_player}"


def parse_checkpoint_iteration(path: Path) -> int:
    match = ITERATION_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse checkpoint iteration from {path}")
    return int(match.group(1))


def discover_checkpoints(checkpoint_dir: Path, checkpoint_pattern: str) -> list[CheckpointEntry]:
    entries = [
        CheckpointEntry(iteration=parse_checkpoint_iteration(path), path=path)
        for path in checkpoint_dir.glob(checkpoint_pattern)
    ]
    return sorted(entries, key=lambda entry: (entry.iteration, entry.path.as_posix()))


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return items


def sample_checkpoints(
    checkpoints: list[CheckpointEntry],
    *,
    sample_size: int,
    include_iters: tuple[int, ...] = DEFAULT_INCLUDE_ITERS,
) -> list[CheckpointEntry]:
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    if not checkpoints:
        return []
    if sample_size >= len(checkpoints):
        return list(checkpoints)

    by_iteration = {entry.iteration: entry for entry in checkpoints}
    selected: dict[int, CheckpointEntry] = {
        checkpoints[0].iteration: checkpoints[0],
        checkpoints[-1].iteration: checkpoints[-1],
    }
    for iteration in include_iters:
        if iteration in by_iteration:
            selected[iteration] = by_iteration[iteration]

    remaining_slots = max(0, sample_size - len(selected))
    available = [entry for entry in checkpoints if entry.iteration not in selected]
    for entry in evenly_spaced_entries(available, remaining_slots):
        selected[entry.iteration] = entry

    return sorted(selected.values(), key=lambda entry: entry.iteration)


def evenly_spaced_entries(entries: list[CheckpointEntry], count: int) -> list[CheckpointEntry]:
    if count <= 0 or not entries:
        return []
    if count >= len(entries):
        return list(entries)

    indexes: list[int] = []
    max_index = len(entries) - 1
    for offset in range(count):
        raw_index = round(offset * max_index / (count - 1)) if count > 1 else max_index // 2
        while raw_index in indexes and raw_index < max_index:
            raw_index += 1
        while raw_index in indexes and raw_index > 0:
            raw_index -= 1
        indexes.append(raw_index)
    return [entries[index] for index in sorted(indexes)]


def build_schedule(
    checkpoints: list[CheckpointEntry],
    *,
    games_per_pair: int,
    seed: int,
) -> list[TournamentGame]:
    if games_per_pair < 1:
        raise ValueError("games_per_pair must be at least 1")

    games: list[TournamentGame] = []
    for pair_index, (checkpoint_a, checkpoint_b) in enumerate(combinations(checkpoints, 2)):
        for game_in_pair in range(games_per_pair):
            games.append(
                TournamentGame(
                    pair_index=pair_index,
                    game_in_pair=game_in_pair,
                    checkpoint_a=checkpoint_a,
                    checkpoint_b=checkpoint_b,
                    candidate_player=game_in_pair % 2,
                    seed=seed + pair_index * games_per_pair + game_in_pair,
                )
            )
    return games


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open(encoding="utf8") as input_file:
        for line in input_file:
            if line.strip():
                records.append(json.loads(line))
    return records


def append_jsonl(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as output:
        output.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
        output.write("\n")


def completed_game_keys(records: list[dict]) -> set[str]:
    return {
        str(record["tournamentGameKey"])
        for record in records
        if record.get("tournamentGameKey") and record.get("terminal")
    }


def pending_games(schedule: list[TournamentGame], records: list[dict]) -> list[TournamentGame]:
    completed = completed_game_keys(records)
    return [game for game in schedule if game.game_key not in completed]


def add_tournament_metadata(record: dict, game: TournamentGame) -> dict:
    enriched = dict(record)
    winner_iteration = winner_checkpoint_iteration(enriched)
    enriched["tournamentGameKey"] = game.game_key
    enriched["tournament"] = {
        "pairIndex": game.pair_index,
        "pairKey": game.pair_key,
        "gameInPair": game.game_in_pair,
        "checkpointA": checkpoint_for_json(game.checkpoint_a),
        "checkpointB": checkpoint_for_json(game.checkpoint_b),
        "candidateIteration": game.checkpoint_a.iteration,
        "baselineIteration": game.checkpoint_b.iteration,
        "candidatePlayer": game.candidate_player,
        "seed": game.seed,
    }
    enriched["checkpointIterationByRole"] = {
        "candidate": game.checkpoint_a.iteration,
        "baseline": game.checkpoint_b.iteration,
    }
    enriched["winnerCheckpointIteration"] = winner_iteration
    return enriched


def winner_checkpoint_iteration(record: dict) -> int | str:
    winner = record["winner"]
    if winner == "draw":
        return "draw"
    candidate_player = int(record["candidatePlayer"])
    role = "candidate" if int(winner) == candidate_player else "baseline"
    if "checkpointIterationByRole" in record:
        return int(record["checkpointIterationByRole"][role])
    return parse_checkpoint_iteration(Path(record[f"{role}Checkpoint"]))


def checkpoint_for_json(entry: CheckpointEntry) -> dict:
    return {"iteration": entry.iteration, "path": entry.path.as_posix()}


def checkpoint_iterations_from_records(records: list[dict]) -> list[int]:
    iterations: set[int] = set()
    for record in records:
        roles = record.get("checkpointIterationByRole")
        if roles:
            iterations.add(int(roles["candidate"]))
            iterations.add(int(roles["baseline"]))
        else:
            iterations.add(parse_checkpoint_iteration(Path(record["candidateCheckpoint"])))
            iterations.add(parse_checkpoint_iteration(Path(record["baselineCheckpoint"])))
    return sorted(iterations)


def score_for_iteration(record: dict, iteration: int) -> float:
    winner_iteration = winner_checkpoint_iteration(record)
    if winner_iteration == "draw":
        return 0.5
    return 1.0 if int(winner_iteration) == iteration else 0.0


def margin_for_iteration(record: dict, iteration: int) -> int:
    roles = record.get("checkpointIterationByRole")
    if roles is None:
        roles = {
            "candidate": parse_checkpoint_iteration(Path(record["candidateCheckpoint"])),
            "baseline": parse_checkpoint_iteration(Path(record["baselineCheckpoint"])),
        }
    candidate_player = int(record["candidatePlayer"])
    if int(roles["candidate"]) == iteration:
        player = candidate_player
    elif int(roles["baseline"]) == iteration:
        player = 1 - candidate_player
    else:
        raise ValueError(f"Iteration {iteration} did not play in record")
    opponent = 1 - player
    return int(record["finalScores"][player]) - int(record["finalScores"][opponent])


def opponent_iteration(record: dict, iteration: int) -> int:
    roles = record.get("checkpointIterationByRole")
    if roles is None:
        roles = {
            "candidate": parse_checkpoint_iteration(Path(record["candidateCheckpoint"])),
            "baseline": parse_checkpoint_iteration(Path(record["baselineCheckpoint"])),
        }
    candidate = int(roles["candidate"])
    baseline = int(roles["baseline"])
    if iteration == candidate:
        return baseline
    if iteration == baseline:
        return candidate
    raise ValueError(f"Iteration {iteration} did not play in record")


def fit_bradley_terry_elos(records: list[dict], iterations: list[int]) -> dict[int, float]:
    if not iterations:
        return {}

    index_by_iteration = {iteration: index for index, iteration in enumerate(iterations)}
    wins = [0.0 for _ in iterations]
    meetings = [[0.0 for _ in iterations] for _ in iterations]
    for record in records:
        roles = record.get("checkpointIterationByRole")
        if roles is None:
            candidate = parse_checkpoint_iteration(Path(record["candidateCheckpoint"]))
            baseline = parse_checkpoint_iteration(Path(record["baselineCheckpoint"]))
        else:
            candidate = int(roles["candidate"])
            baseline = int(roles["baseline"])
        if candidate not in index_by_iteration or baseline not in index_by_iteration:
            continue
        candidate_index = index_by_iteration[candidate]
        baseline_index = index_by_iteration[baseline]
        candidate_score = score_for_iteration(record, candidate)
        baseline_score = 1.0 - candidate_score
        wins[candidate_index] += candidate_score
        wins[baseline_index] += baseline_score
        meetings[candidate_index][baseline_index] += 1.0
        meetings[baseline_index][candidate_index] += 1.0

    ability = [1.0 for _ in iterations]
    epsilon = 1e-9
    for _ in range(200):
        next_ability = ability.copy()
        for i, _iteration in enumerate(iterations):
            denominator = 0.0
            for j, _opponent in enumerate(iterations):
                if i == j or meetings[i][j] == 0:
                    continue
                denominator += meetings[i][j] / max(ability[i] + ability[j], epsilon)
            if denominator > 0:
                next_ability[i] = max(wins[i], epsilon) / denominator
        geometric_mean = math.exp(sum(math.log(max(value, epsilon)) for value in next_ability) / len(next_ability))
        ability = [max(value / geometric_mean, epsilon) for value in next_ability]

    scale = 400.0 / math.log(10)
    return {
        iteration: 1500.0 + scale * math.log(max(ability[index], epsilon))
        for iteration, index in index_by_iteration.items()
    }


def standings_rows(records: list[dict], sampled: list[CheckpointEntry] | None = None) -> list[dict]:
    iterations = [entry.iteration for entry in sampled] if sampled is not None else checkpoint_iterations_from_records(records)
    ratings = fit_bradley_terry_elos(records, iterations)
    rows: list[dict] = []
    for iteration in iterations:
        played = [record for record in records if iteration in record_iterations(record)]
        wins = sum(1 for record in played if score_for_iteration(record, iteration) == 1.0)
        draws = sum(1 for record in played if score_for_iteration(record, iteration) == 0.5)
        losses = sum(1 for record in played if score_for_iteration(record, iteration) == 0.0)
        margins = [margin_for_iteration(record, iteration) for record in played]
        points = wins + 0.5 * draws
        rows.append(
            {
                "iteration": iteration,
                "checkpoint": checkpoint_path_for_iteration(iteration, sampled),
                "games": len(played),
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "points": points,
                "winRate": points / len(played) if played else 0.0,
                "averageScoreMargin": sum(margins) / len(margins) if margins else 0.0,
                "rating": ratings.get(iteration, 1500.0),
            }
        )
    return sorted(rows, key=lambda row: (-float(row["rating"]), -float(row["points"]), int(row["iteration"])))


def record_iterations(record: dict) -> set[int]:
    roles = record.get("checkpointIterationByRole")
    if roles is not None:
        return {int(roles["candidate"]), int(roles["baseline"])}
    return {
        parse_checkpoint_iteration(Path(record["candidateCheckpoint"])),
        parse_checkpoint_iteration(Path(record["baselineCheckpoint"])),
    }


def checkpoint_path_for_iteration(iteration: int, sampled: list[CheckpointEntry] | None) -> str:
    if sampled is None:
        return ""
    for entry in sampled:
        if entry.iteration == iteration:
            return entry.path.as_posix()
    return ""


def pairing_rows(records: list[dict], schedule: list[TournamentGame]) -> list[dict]:
    records_by_pair: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        pair_key = record.get("tournament", {}).get("pairKey")
        if pair_key:
            records_by_pair[str(pair_key)].append(record)

    rows: list[dict] = []
    scheduled_by_pair: dict[str, list[TournamentGame]] = defaultdict(list)
    for game in schedule:
        scheduled_by_pair[game.pair_key].append(game)

    for pair_key, scheduled_games in scheduled_by_pair.items():
        game = scheduled_games[0]
        pair_records = records_by_pair.get(game.pair_key, [])
        iteration_a = game.checkpoint_a.iteration
        iteration_b = game.checkpoint_b.iteration
        score_a = sum(score_for_iteration(record, iteration_a) for record in pair_records)
        score_b = sum(score_for_iteration(record, iteration_b) for record in pair_records)
        margins_a = [margin_for_iteration(record, iteration_a) for record in pair_records]
        rows.append(
            {
                "pairKey": game.pair_key,
                "iterationA": iteration_a,
                "iterationB": iteration_b,
                "completedGames": len(pair_records),
                "scheduledGames": len(scheduled_games),
                "scoreA": score_a,
                "scoreB": score_b,
                "averageMarginA": sum(margins_a) / len(margins_a) if margins_a else 0.0,
                "winner": pair_winner(iteration_a, iteration_b, score_a, score_b),
            }
        )
    return rows


def pair_winner(iteration_a: int, iteration_b: int, score_a: float, score_b: float) -> int | str:
    if score_a > score_b:
        return iteration_a
    if score_b > score_a:
        return iteration_b
    return "draw"


def summarize_tournament(
    *,
    sampled: list[CheckpointEntry],
    schedule: list[TournamentGame],
    records: list[dict],
    anchor_iteration: int = 542,
) -> dict:
    standings = standings_rows(records, sampled)
    diagnosis = forgetting_diagnosis(standings=standings, records=records, anchor_iteration=anchor_iteration)
    return {
        "sampledCheckpoints": [checkpoint_for_json(entry) for entry in sampled],
        "sampledIterations": [entry.iteration for entry in sampled],
        "totalScheduledGames": len(schedule),
        "completedGames": len(completed_game_keys(records)),
        "pendingGames": len(schedule) - len(completed_game_keys(records)),
        "topCheckpoints": standings[:10],
        "anchorIteration": anchor_iteration,
        "anchorStanding": next((row for row in standings if int(row["iteration"]) == anchor_iteration), None),
        "forgetting": diagnosis,
    }


def forgetting_diagnosis(
    *,
    standings: list[dict],
    records: list[dict],
    anchor_iteration: int = 542,
) -> dict:
    if not standings:
        return {"status": "inconclusive", "forgettingFlag": False}

    by_iteration = {int(row["iteration"]): row for row in standings}
    latest_iteration = max(by_iteration)
    latest = by_iteration[latest_iteration]
    peak = max(standings, key=lambda row: float(row["rating"]))
    anchor = by_iteration.get(anchor_iteration)
    latest_rating = float(latest["rating"])
    peak_rating = float(peak["rating"])
    anchor_rating = float(anchor["rating"]) if anchor else None
    latest_vs_anchor = head_to_head_summary(records, latest_iteration, anchor_iteration)
    post_anchor_trend = rating_slope(
        [
            (iteration, float(row["rating"]))
            for iteration, row in sorted(by_iteration.items())
            if iteration >= anchor_iteration
        ]
    )
    latest_delta_from_peak = latest_rating - peak_rating
    latest_delta_from_anchor = latest_rating - anchor_rating if anchor_rating is not None else None
    latest_anchor_score = latest_vs_anchor["scoreA"] if latest_vs_anchor else None
    latest_anchor_games = latest_vs_anchor["games"] if latest_vs_anchor else 0

    below_peak = latest_delta_from_peak <= -MEANINGFUL_RATING_DROP
    below_anchor = (
        latest_delta_from_anchor is not None
        and latest_delta_from_anchor <= -MEANINGFUL_RATING_DROP
    )
    non_positive_anchor_h2h = latest_anchor_games > 0 and latest_anchor_score <= latest_anchor_games / 2
    forgetting_flag = bool(below_peak and below_anchor and non_positive_anchor_h2h)
    if forgetting_flag:
        status = "likely_forgetting"
    elif latest_delta_from_peak >= -MEANINGFUL_RATING_DROP and (
        latest_delta_from_anchor is None or latest_delta_from_anchor >= -MEANINGFUL_RATING_DROP
    ):
        status = "no_clear_forgetting"
    else:
        status = "inconclusive"

    return {
        "status": status,
        "forgettingFlag": forgetting_flag,
        "latestIteration": latest_iteration,
        "latestRating": latest_rating,
        "peakRatingIteration": int(peak["iteration"]),
        "peakRating": peak_rating,
        "latestRatingDeltaFromPeak": latest_delta_from_peak,
        "anchorIteration": anchor_iteration,
        "anchorRating": anchor_rating,
        "latestRatingDeltaFrom542": latest_delta_from_anchor,
        "latestVs542": latest_vs_anchor,
        "post542Trend": post_anchor_trend,
        "meaningfulRatingDrop": MEANINGFUL_RATING_DROP,
    }


def head_to_head_summary(records: list[dict], iteration_a: int, iteration_b: int) -> dict | None:
    head_to_head = [
        record
        for record in records
        if {iteration_a, iteration_b} == record_iterations(record)
    ]
    if not head_to_head:
        return None
    score_a = sum(score_for_iteration(record, iteration_a) for record in head_to_head)
    score_b = sum(score_for_iteration(record, iteration_b) for record in head_to_head)
    margins_a = [margin_for_iteration(record, iteration_a) for record in head_to_head]
    return {
        "iterationA": iteration_a,
        "iterationB": iteration_b,
        "games": len(head_to_head),
        "scoreA": score_a,
        "scoreB": score_b,
        "averageMarginA": sum(margins_a) / len(margins_a),
        "winner": pair_winner(iteration_a, iteration_b, score_a, score_b),
    }


def rating_slope(points: list[tuple[int, float]]) -> float | None:
    if len(points) < 2:
        return None
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator == 0:
        return None
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in points)
    return numerator / denominator


def write_dict_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    with path.open("w", encoding="utf8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    *,
    out_dir: Path,
    sampled: list[CheckpointEntry],
    schedule: list[TournamentGame],
    records: list[dict],
    anchor_iteration: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_tournament(
        sampled=sampled,
        schedule=schedule,
        records=records,
        anchor_iteration=anchor_iteration,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf8")
    write_dict_csv(standings_rows(records, sampled), out_dir / "standings.csv")
    write_dict_csv(pairing_rows(records, schedule), out_dir / "pairings.csv")


def run_tournament(
    *,
    sampled: list[CheckpointEntry],
    schedule: list[TournamentGame],
    out_dir: Path,
    rows: int,
    cols: int,
    simulations: int,
    c_puct: float,
    opening_random_plies: int,
    mlx_device: str,
    mcts_backend: str,
    mcts_batch_size: int,
    virtual_loss: float,
    reuse_tree: bool,
    evaluator_cache_entries: int,
    anchor_iteration: int,
) -> None:
    games_path = out_dir / "games.jsonl"
    records = read_jsonl(games_path)
    remaining = pending_games(schedule, records)
    write_outputs(
        out_dir=out_dir,
        sampled=sampled,
        schedule=schedule,
        records=records,
        anchor_iteration=anchor_iteration,
    )
    if not remaining:
        print(f"All {len(schedule)} tournament games are already complete in {games_path}.")
        return

    for index, game in enumerate(remaining, start=1):
        print(
            f"[{index}/{len(remaining)}] ITER={game.checkpoint_a.iteration:03d} "
            f"vs ITER={game.checkpoint_b.iteration:03d} candidatePlayer={game.candidate_player}"
        )
        record = play_checkpoint_match_game(
            candidate_checkpoint=game.checkpoint_a.path,
            baseline_checkpoint=game.checkpoint_b.path,
            rows=rows,
            cols=cols,
            simulations=simulations,
            seed=game.seed,
            candidate_player=game.candidate_player,
            c_puct=c_puct,
            opening_random_plies=opening_random_plies,
            device=mlx_device,
            reuse_tree=reuse_tree,
            evaluator_cache_entries=evaluator_cache_entries,
            mcts_backend=mcts_backend,
            mcts_batch_size=mcts_batch_size,
            virtual_loss=virtual_loss,
        )
        enriched = add_tournament_metadata(record, game)
        append_jsonl(enriched, games_path)
        records.append(enriched)
        write_outputs(
            out_dir=out_dir,
            sampled=sampled,
            schedule=schedule,
            records=records,
            anchor_iteration=anchor_iteration,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a dense all-pairs tournament over EpsilonZero checkpoints."
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-pattern", default=DEFAULT_CHECKPOINT_PATTERN)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--include-iters", type=parse_int_list, default=DEFAULT_INCLUDE_ITERS)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--games-per-pair", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--opening-random-plies", type=int, default=2)
    parser.add_argument("--mlx-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--mcts-backend", choices=["python", "cpp"], default="cpp")
    parser.add_argument("--mcts-batch-size", type=int, default=8)
    parser.add_argument("--virtual-loss", type=float, default=1.0)
    parser.add_argument("--enable-tree-reuse", action="store_true")
    parser.add_argument("--evaluator-cache-entries", type=int, default=500_000)
    args = parser.parse_args()

    if args.sample_size < 1:
        raise SystemExit("--sample-size must be at least 1")
    if args.simulations < 1:
        raise SystemExit("--simulations must be at least 1")
    if args.games_per_pair < 1:
        raise SystemExit("--games-per-pair must be at least 1")
    if args.opening_random_plies < 0:
        raise SystemExit("--opening-random-plies must be non-negative")
    if args.mcts_batch_size < 1:
        raise SystemExit("--mcts-batch-size must be at least 1")
    if args.virtual_loss < 0:
        raise SystemExit("--virtual-loss must be non-negative")
    if args.evaluator_cache_entries < 0:
        raise SystemExit("--evaluator-cache-entries must be non-negative")

    checkpoints = discover_checkpoints(args.checkpoint_dir, args.checkpoint_pattern)
    if len(checkpoints) < 2:
        raise SystemExit(
            f"Need at least two checkpoints matching {args.checkpoint_dir / args.checkpoint_pattern}"
        )
    sampled = sample_checkpoints(
        checkpoints,
        sample_size=args.sample_size,
        include_iters=args.include_iters,
    )
    schedule = build_schedule(sampled, games_per_pair=args.games_per_pair, seed=args.seed)
    anchor_iteration = args.include_iters[0] if args.include_iters else 542
    run_tournament(
        sampled=sampled,
        schedule=schedule,
        out_dir=args.out_dir,
        rows=args.rows,
        cols=args.cols,
        simulations=args.simulations,
        c_puct=args.c_puct,
        opening_random_plies=args.opening_random_plies,
        mlx_device=args.mlx_device,
        mcts_backend=args.mcts_backend,
        mcts_batch_size=args.mcts_batch_size,
        virtual_loss=args.virtual_loss,
        reuse_tree=args.enable_tree_reuse,
        evaluator_cache_entries=args.evaluator_cache_entries,
        anchor_iteration=anchor_iteration,
    )


if __name__ == "__main__":
    main()
