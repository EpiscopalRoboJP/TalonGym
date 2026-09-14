from talongym.sim.geometry import AABB
from talongym.sim.physics import Body, _chassis_vs_circle, _resolve_chassis


DIAMOND = ((8.0, 0.0), (0.0, 4.0), (-8.0, 0.0), (0.0, -4.0))


def test_diamond_hull_misses_aabb_corner():
    box = AABB(7.2, 3.6, 0.35, 0.35)
    aabb_bot = Body("red_0", 0.0, 0.0, 0.0, hx=8.0, hy=4.0)
    mesh_bot = Body("red_1", 0.0, 0.0, 0.0, hx=8.0, hy=4.0, kind="mesh", footprint=DIAMOND)
    wall_aabb, _ = _resolve_chassis(aabb_bot, [box], [], 72.0, 72.0)
    wall_mesh, _ = _resolve_chassis(mesh_bot, [box], [], 72.0, 72.0)
    assert wall_aabb is True
    assert wall_mesh is False
    assert abs(mesh_bot.x) < 1e-6
    assert abs(mesh_bot.y) < 1e-6


def test_diamond_hull_hits_along_axis():
    box = AABB(6.0, 0.0, 1.0, 1.0)
    mesh_bot = Body("red_0", 0.0, 0.0, 0.0, hx=8.0, hy=4.0, kind="mesh", footprint=DIAMOND)
    wall, _ = _resolve_chassis(mesh_bot, [box], [], 72.0, 72.0)
    assert wall is True


def test_mesh_chassis_vs_circle_outside_corner():
    robot = Body("red_0", 0.0, 0.0, 0.0, hx=8.0, hy=4.0, kind="mesh", footprint=DIAMOND)
    piece = Body("p", 7.0, 3.5, 0.0, kind="circle", radius=0.3)
    assert _chassis_vs_circle(robot, piece) is False
    piece2 = Body("p2", 8.2, 0.0, 0.0, kind="circle", radius=0.4)
    assert _chassis_vs_circle(robot, piece2) is True
