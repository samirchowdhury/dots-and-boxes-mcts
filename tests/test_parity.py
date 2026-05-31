import json
from pathlib import Path

from dots_boxes_mcts.game import apply_move, legal_moves, new_game, state_snapshot

FIXTURES_PATH = Path(__file__).resolve().parents[2] / "dots-and-boxes" / "fixtures" / "rules" / "positions.json"


def test_python_engine_matches_canonical_js_fixtures() -> None:
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf8"))

    for fixture in fixtures:
        state = new_game(rows=fixture["rows"], cols=fixture["cols"])
        for move in fixture["moves"]:
            state = apply_move(state, move)

        snapshot = state_snapshot(state)
        expected = fixture["expected"]

        assert snapshot["currentPlayer"] == expected["currentPlayer"], fixture["name"]
        assert snapshot["scores"] == expected["scores"], fixture["name"]
        assert snapshot["boxes"] == expected["boxes"], fixture["name"]
        assert snapshot["terminal"] == expected["terminal"], fixture["name"]
        assert snapshot["winner"] == expected["winner"], fixture["name"]
        assert snapshot["edges"] == expected["edges"], fixture["name"]
        assert snapshot["edgeOwners"] == expected["edgeOwners"], fixture["name"]
        assert len(legal_moves(state)) == expected["legalMovesCount"], fixture["name"]
