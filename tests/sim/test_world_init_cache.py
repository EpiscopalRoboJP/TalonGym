"""World init shares compiled robots, MJCF, and MuJoCo indexes across envs."""

from __future__ import annotations

import copy

import pytest

from talongym.presets.loader import load_preset
from talongym.robot import assembly
from talongym.robot.assembly import clear_materialize_cache, materialize_sim_robot


def test_materialize_sim_robot_caches_by_preset_identity(monkeypatch):
    clear_materialize_cache()
    calls = {"n": 0}
    orig = assembly.compile_assembly_to_preset

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(assembly, "compile_assembly_to_preset", wrapped)
    robot = load_preset("robot", "gobilda_mecanum_starter")
    first = materialize_sim_robot(robot)
    second = materialize_sim_robot(robot)
    assert calls["n"] == 1
    assert first is second
    materialize_sim_robot(copy.deepcopy(robot))
    assert calls["n"] == 2


@pytest.mark.require_mesh
def test_second_world_reuses_mjcf_model_and_keeps_private_data(monkeypatch):
    from talongym.assets import mjcf_field
    from talongym.presets.loader import load_bundle
    from talongym.sim import world as world_mod
    from talongym.sim.mujoco_backend import clear_mj_index_cache
    from talongym.sim.world import World

    clear_materialize_cache()
    world_mod.clear_collision_xml_cache()
    clear_mj_index_cache()
    calls = {"n": 0}
    orig = mjcf_field.build_field_mjcf

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(mjcf_field, "build_field_mjcf", wrapped)
    bundle = load_bundle()
    first = World(bundle)
    second = World(bundle)
    assert calls["n"] == 1
    assert first.backend._mj is second.backend._mj
    assert first.backend._data is not second.backend._data
    first.backend._owned.add("probe")
    assert "probe" not in second.backend._owned
    first.backend._id_to_slot["piece"] = 0
    assert "piece" not in second.backend._id_to_slot
