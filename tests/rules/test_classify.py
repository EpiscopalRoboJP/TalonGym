import numpy as np

from talongym.presets.loader import load_bundle
from talongym.sim.world import World


def _intake_then_score(world: World, n_intake: int = 20) -> None:
    rs = world.actor()
    piece = next(p for p in world.pieces.values() if not p.scored)
    rs.body.x, rs.body.y = piece.x, piece.y
    for _ in range(n_intake):
        world.step(np.array([rs.body.x, rs.body.y, rs.body.heading]), 0.2, 1)
        rs = world.actor()
    assert rs.held, "intake did not capture a piece"
    world.step(np.array([rs.body.x, rs.body.y, rs.body.heading]), 0.2, 2)
    for _ in range(80):
        world.step(np.array([16.0, 12.0, 1.57]), 0.3, 0)


def test_classify_scores_three():
    bundle = load_bundle()
    world = World(bundle, seed=3)
    world.reset(seed=3, static_teammate=False)
    _intake_then_score(world)
    assert world.accumulators.get("classified_count", 0) >= 1
    assert world.true_score >= 3


def test_overflow_when_queue_full():
    bundle = load_bundle()
    world = World(bundle, seed=1)
    world.reset(seed=1, static_teammate=False)
    world.accumulators["red_ramp_seq"] = ["P"] * 9
    world.queues["red_ramp_seq"] = ["P"] * 9
    _intake_then_score(world)
    assert world.accumulators.get("overflow_count", 0) >= 1


def test_pattern_requires_closed_gate():
    bundle = load_bundle()
    world = World(bundle, seed=2)
    world.reset(seed=2, static_teammate=False)
    world.accumulators["red_ramp_seq"] = list(world.match_vars["motif"])
    world.queues["red_ramp_seq"] = list(world.match_vars["motif"])
    world.accumulators["left_launch_line"] = True
    world.gate_state["red_gate"] = "closed"
    world.step(np.array([0.0, 0.0, 1.57]), 0.2, 0, end_phase=True)
    with_gate = world.true_score
    world.reset(seed=2, static_teammate=False)
    world.accumulators["red_ramp_seq"] = list(world.match_vars["motif"])
    world.queues["red_ramp_seq"] = list(world.match_vars["motif"])
    world.accumulators["left_launch_line"] = True
    world.gate_state["red_gate"] = "open"
    world.step(np.array([0.0, 0.0, 1.57]), 0.2, 0, end_phase=True)
    assert with_gate > world.true_score


def test_gate_open_clears_ramp_queue():
    bundle = load_bundle()
    world = World(bundle, seed=5)
    world.reset(seed=5, static_teammate=False)
    world.accumulators["red_ramp_seq"] = ["P", "G"]
    world.queues["red_ramp_seq"] = ["P", "G"]
    gate = next(e for e in world.elements if e["id"] == "red_gate")
    pose = gate["pose"]
    rs = world.actor()
    rs.body.x, rs.body.y = float(pose["x"]), float(pose["y"])
    world.step(np.array([rs.body.x, rs.body.y, 0.0]), 0.2, 3)
    assert world.gate_state.get("red_gate") == "open"
    assert world.queues.get("red_ramp_seq") == []
