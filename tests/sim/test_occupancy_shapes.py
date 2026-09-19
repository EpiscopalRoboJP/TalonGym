from talongym.presets.loader import load_bundle
from talongym.sim.geometry import AABB, Circle, point_in_shape, point_in_volume, shape_from_element, tag_set
from talongym.sim.world import World


def _world() -> World:
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    return World(bundle, seed=0, allow_missing_mesh=True)


def test_point_in_shape_ignores_non_geometry():
    assert point_in_shape(AABB(0, 0, 2, 2), 0.0, 0.0)
    assert point_in_shape(Circle(0, 0, 2), 0.0, 0.0)
    assert not point_in_shape(None, 0.0, 0.0)
    assert not point_in_shape(iter({"a": 1}.items()), 0.0, 0.0)
    assert not point_in_shape({"kind": "aabb", "width": 4, "depth": 4}, 0.0, 0.0)


def test_shape_from_element_tolerates_garbage():
    assert shape_from_element("not-a-dict") is None  # type: ignore[arg-type]
    assert shape_from_element({"id": "x", "shape": iter({"kind": "aabb"}.items())}) is None
    assert tag_set(iter({"flower": True}.items())) == set()
    flower = {"id": "flower_1", "type": "flower", "tags": ["flower"], "pose": {"x": 0, "y": 0, "z": 0}, "shape": {"kind": "aabb", "width": 4, "depth": 4, "height": 8}}
    sh = shape_from_element(flower)
    flower["shape"] = iter({"height": 8}.items())
    flower["tags"] = iter({"flower": True}.items())
    assert point_in_volume(flower, sh, 0.0, 0.0, 0.0, 0.0)


def test_occupancy_survives_corrupt_live_lookups():
    world = _world()
    world.reset(seed=0, static_teammate=True, full_noise=False)
    key = next(iter(world.element_shapes))
    world.element_shapes[key] = iter({"x": 1}.items())
    piece = next(iter(world.pieces.values()))
    world.triggers = piece  # type: ignore[assignment]
    occ = world._occupancy()
    assert occ
    assert all(isinstance(ids, set) for ids in occ.values())
