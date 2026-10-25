# Tests for the diagnostics report: the Heuristic baseline must show full agreement and no skipped priority moves.

from env.state_encoding import ObservationSpec
from agents.heuristic_agent import HeuristicAgent
from training.diagnostics import build_report, play_diagnostic_games, risk_after
from training.tournament import wrap_choose_action_fn
from env.board import BoardState


# Checks the Heuristic scored by its own report agrees with itself and never skips a capture, exit or home entry.
def test_heuristic_baseline_report_is_clean():
    tallies = play_diagnostic_games(wrap_choose_action_fn(HeuristicAgent()), ObservationSpec(), 4, seed=7)
    report = build_report(tallies)
    assert report["agreement_with_heuristic"] == 1.0
    skipped = {row["mistake"]: row["per_game"] for row in report["ranked_mistakes"]}
    assert skipped["skipped a capture"] == 0 and skipped["skipped leaving base on a 6"] == 0
    assert skipped["skipped entering home"] == 0
    assert report["ranked_mistakes"] == sorted(report["ranked_mistakes"], key=lambda r: r["per_game"], reverse=True)


# Checks risk_after counts the rolls an opponent would need to capture a token after a move.
def test_risk_after_counts_attacking_rolls():
    board = BoardState()
    board.set(0, 0, 19)
    board.set(1, 0, 3)
    assert risk_after(board, 0, 0, 1) == 1
    far = BoardState()
    far.set(0, 0, 19)
    assert risk_after(far, 0, 0, 1) == 0
