from __future__ import annotations

import math
from typing import Any


def decimate(waypoints: list[list[float]], min_dist: float = 8.0) -> list[list[float]]:
    if not waypoints:
        return []
    out = [waypoints[0]]
    for p in waypoints[1:]:
        prev = out[-1]
        if math.hypot(p[0] - prev[0], p[1] - prev[1]) >= min_dist:
            out.append(p)
    if out[-1] != waypoints[-1]:
        out.append(waypoints[-1])
    return out


def to_roadrunner_java(
    waypoints: list[list[float]],
    *,
    dialect: str = "rr1_actions",
    start: list[float] | None = None,
) -> str:
    pts = decimate(waypoints)
    if not pts:
        start = start or [0.0, 0.0, 0.0]
        pts = [start]
    sx, sy, sh = pts[0][0], pts[0][1], pts[0][2] if len(pts[0]) > 2 else 0.0
    if dialect == "rr05_trajectory_sequence":
        lines = [
            "// TalonGym export — Road Runner 0.5.x compatibility",
            f"drive.trajectorySequenceBuilder(new Pose2d({sx:.2f}, {sy:.2f}, {sh:.4f}))",
        ]
        for p in pts[1:]:
            lines.append(f"    .lineToLinearHeading(new Pose2d({p[0]:.2f}, {p[1]:.2f}, {p[2] if len(p) > 2 else 0:.4f}))")
        lines.append("    .build();")
        return "\n".join(lines) + "\n"
    lines = [
        "// TalonGym export — Road Runner 1.0 Actions",
        "// Paste into an AUTO OpMode. Control Hub runs it. Inches, FTC official coords.",
        "Actions.runBlocking(",
        f"    drive.actionBuilder(new Pose2d({sx:.2f}, {sy:.2f}, {sh:.4f}))",
    ]
    for p in pts[1:]:
        heading = p[2] if len(p) > 2 else 0.0
        lines.append(f"        .splineTo(new Vector2d({p[0]:.2f}, {p[1]:.2f}), {heading:.4f})")
    lines.append("        .build());")
    return "\n".join(lines) + "\n"


def export_from_replay(frames: list[dict[str, Any]], dialect: str = "rr1_actions") -> str:
    wps: list[list[float]] = []
    for fr in frames:
        robots = fr.get("robots") or []
        if not robots:
            continue
        r = robots[0]
        heading = math.radians(float(r.get("headingDeg", 0.0)))
        wps.append([float(r["x"]), float(r["y"]), heading])
    return to_roadrunner_java(wps, dialect=dialect)
