from talongym.sim.geometry import AABB
from talongym.sim.physics import Body, Planar2DBackend, WorldStep, perimeter_walls


def _backend():
    return Planar2DBackend(72.0, 72.0)


def test_chassis_pushes_piece():
    bot = Body("r", -10.0, 0.0, 0.0, vx=20.0, hx=9.0, hy=9.0)
    piece = Body("p", 4.0, 0.0, 0.0, kind="circle", radius=2.5, hx=2.5, hy=2.5, mass=0.1)
    backend = _backend()
    x0 = piece.x
    for _ in range(40):
        backend.step_world(
            WorldStep(robots=[bot], pieces=[piece], obstacles=[], walls=perimeter_walls(72, 72), max_vel=40),
            0.02,
        )
        bot.vx = 20.0
    assert piece.x > x0 + 0.5
    assert backend.contacts().piece or piece.x != x0


def test_two_pieces_separate():
    a = Body("a", 0.0, 0.0, 0.0, kind="circle", radius=3.0, hx=3, hy=3, mass=0.1)
    b = Body("b", 1.0, 0.0, 0.0, kind="circle", radius=3.0, hx=3, hy=3, mass=0.1)
    backend = _backend()
    backend.step_world(WorldStep(robots=[], pieces=[a, b], obstacles=[], walls=[]), 0.02)
    dist = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
    assert dist >= 5.9


def test_static_obstacle_stops_chassis():
    bot = Body("r", 0.0, 0.0, 0.0, vx=30.0, hx=9.0, hy=9.0)
    wall = AABB(20.0, 0.0, 4.0, 20.0)
    backend = _backend()
    for _ in range(80):
        backend.step_world(
            WorldStep(robots=[bot], pieces=[], obstacles=[wall], walls=perimeter_walls(72, 72), max_vel=30),
            0.02,
        )
        bot.vx = 30.0
    assert bot.x < 20.0
