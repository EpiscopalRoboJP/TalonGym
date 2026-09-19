from talongym.credits import CREDIT_LINE, TEAMS
from talongym.license_notice import NOTICE


def test_teams_and_startup_notice():
    assert TEAMS == (
        "FTC Team 17986 — 904 Robo Eagles",
        "FTC Team 27268 — Talon Strike",
    )
    assert CREDIT_LINE == "Built by FTC Team 17986 — 904 Robo Eagles and Team 27268 — Talon Strike."
    assert CREDIT_LINE in NOTICE
    assert "GNU GPL" in NOTICE
