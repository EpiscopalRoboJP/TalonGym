from talongym.export.mechanism import (
    DriveOnlyExportError,
    export_mechanism_timeline,
    mechanism_actions_required,
)
from talongym.export.roadrunner import export_from_replay


def _drive_frames():
    return [
        {"t": 0.0, "robots": [{"id": "red_0", "x": 0.0, "y": 0.0, "headingDeg": 0.0, "lastVerb": "idle"}], "pieces": []},
        {"t": 1.0, "robots": [{"id": "red_0", "x": 24.0, "y": 8.0, "headingDeg": 20.0, "lastVerb": "idle"}], "pieces": []},
    ]


def _mechanism_frames():
    return [
        {
            "t": 0.0,
            "robots": [
                {
                    "id": "red_0",
                    "x": 0.0,
                    "y": 0.0,
                    "headingDeg": 0.0,
                    "lastVerb": "idle",
                    "actuators": {"intake": {"command": 0.0}, "flywheel": {"command": 0.0}},
                    "batteryVoltageV": 12.8,
                }
            ],
            "pieces": [],
        },
        {
            "t": 1.2,
            "robots": [
                {
                    "id": "red_0",
                    "x": 18.0,
                    "y": 4.0,
                    "headingDeg": 10.0,
                    "lastVerb": "score",
                    "actuators": {"intake": {"command": 0.0}, "flywheel": {"command": 1.0}, "gate": {"command": 1.0}},
                    "batteryVoltageV": 11.4,
                }
            ],
            "pieces": [{"id": "p1", "x": 20, "y": 8, "inFlight": True}],
        },
    ]


def test_drive_only_replay_stays_roadrunner():
    java = export_from_replay(_drive_frames())
    assert "actionBuilder" in java
    assert "MECHANISM_ACTIONS_REQUIRED" not in java
    assert not mechanism_actions_required(_drive_frames())


def test_mechanism_replay_rejects_drive_only_and_emits_timeline():
    frames = _mechanism_frames()
    assert mechanism_actions_required(frames)
    java = export_from_replay(frames)
    assert "MECHANISM_ACTIONS_REQUIRED" in java
    assert "actionBuilder" in java
    assert "mechanism command timeline" in java
    assert "flywheel=1.000" in java
    assert "t=1.20s" in java
    try:
        export_from_replay(frames, drive_only=True)
        raise AssertionError("drive-only export must reject mechanism actions")
    except DriveOnlyExportError as exc:
        assert exc.code == "MECHANISM_ACTIONS_REQUIRED"
    timeline = export_mechanism_timeline(frames)
    assert "verb=score" in timeline
    assert "batteryV=11.40" in timeline
