"""Emit MuJoCo MJCF for a field preset (Y-up inches, gravity on)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

IN_G = 386.0886  # 9.80665 m/s^2 in inches/s^2


def _yup(x: float, y: float, z: float) -> tuple[float, float, float]:
    return float(x), float(z), float(-y)


def _size(hx: float, hy: float, hz: float) -> str:
    return f"{hx:.4f} {hy:.4f} {hz:.4f}"


def build_mjcf(
    field: dict[str, Any],
    *,
    n_robots: int = 4,
    n_pieces: int = 80,
    robot_hx: float = 9.0,
    robot_hy: float = 9.0,
    robot_hz: float = 5.0,
    robot_mesh: Path | None = None,
) -> str:
    fw = float(field["fieldSizeIn"]["width"])
    fd = float(field["fieldSizeIn"]["depth"])
    wall_h = float(field["fieldSizeIn"].get("wallHeight") or 12)
    geoms: list[str] = [
        f'    <geom name="floor" type="plane" size="{fw/2:.3f} {fd/2:.3f} 1" pos="0 0 0" zaxis="0 1 0" rgba="0.1 0.3 0.2 1" contype="1" conaffinity="1"/>',
        f'    <geom name="wall_n" type="box" size="{_size(fw/2+1, wall_h/2, 1)}" pos="{" ".join(f"{v:.3f}" for v in _yup(0, fd/2, wall_h/2))}" rgba="0.5 0.6 0.65 1" contype="1" conaffinity="1"/>',
        f'    <geom name="wall_s" type="box" size="{_size(fw/2+1, wall_h/2, 1)}" pos="{" ".join(f"{v:.3f}" for v in _yup(0, -fd/2, wall_h/2))}" rgba="0.5 0.6 0.65 1" contype="1" conaffinity="1"/>',
        f'    <geom name="wall_w" type="box" size="{_size(1, wall_h/2, fd/2)}" pos="{" ".join(f"{v:.3f}" for v in _yup(-fw/2, 0, wall_h/2))}" rgba="0.5 0.6 0.65 1" contype="1" conaffinity="1"/>',
        f'    <geom name="wall_e" type="box" size="{_size(1, wall_h/2, fd/2)}" pos="{" ".join(f"{v:.3f}" for v in _yup(fw/2, 0, wall_h/2))}" rgba="0.5 0.6 0.65 1" contype="1" conaffinity="1"/>',
    ]
    skip_types = {"tape", "zone"}
    skip_tags = {"leave", "park", "loading_zone", "garden", "restricted_for_red", "restricted_for_blue", "perimeter"}
    for el in field.get("elements") or []:
        tags = set(el.get("tags") or [])
        if el.get("type") in skip_types or tags.intersection(skip_tags):
            continue
        pose = el.get("pose") or {}
        shape = el.get("shape") or {}
        if shape.get("kind") != "aabb":
            continue
        w = float(shape.get("width") or 8)
        d = float(shape.get("depth") or 8)
        z = float(pose.get("z") or 6)
        is_cell = "cell" in tags or el.get("type") in {"goal", "cell"}
        is_frame = "frame" in tags or el.get("type") == "hive_frame"
        height = float(shape.get("height") or (14 if is_cell else 24 if is_frame else 10))
        px, py, pz = _yup(float(pose.get("x") or 0), float(pose.get("y") or 0), z)
        eid = escape(str(el["id"]))
        if is_cell:
            # Occupancy is evaluated in World; these geoms are visual only so balls can enter.
            geoms.append(
                f'    <geom name="{eid}" type="box" size="{_size(w/2, height/2, d/2)}" pos="{px:.3f} {py:.3f} {pz:.3f}" '
                f'rgba="0.7 0.3 0.3 0.25" contype="0" conaffinity="0" group="3" density="0"/>'
            )
        else:
            geoms.append(
                f'    <geom name="{eid}" type="box" size="{_size(w/2, height/2, d/2)}" pos="{px:.3f} {py:.3f} {pz:.3f}" '
                f'rgba="0.4 0.35 0.3 1" contype="1" conaffinity="1"/>'
            )

    robot_ids = ["red_0", "red_1", "blue_0", "blue_1"][: max(1, n_robots)]
    mesh_file = None
    meshdir = None
    if robot_mesh is not None:
        mesh_path = Path(robot_mesh)
        if mesh_path.is_file():
            mesh_file = mesh_path.name
            meshdir = mesh_path.parent.resolve().as_posix()
    bodies: list[str] = []
    for rid in robot_ids:
        if mesh_file:
            geom = (
                f'<geom name="{rid}_geom" type="mesh" mesh="robot_hull" mass="15" '
                f'rgba="0.9 0.8 0.6 1" contype="1" conaffinity="1"/>'
            )
        else:
            geom = (
                f'<geom name="{rid}_geom" type="box" size="{robot_hx:.3f} {robot_hz:.3f} {robot_hy:.3f}" '
                f'mass="15" rgba="0.9 0.8 0.6 1" contype="1" conaffinity="1"/>'
            )
        bodies.append(
            f"""    <body name="{rid}" pos="0 {robot_hz + 0.2:.3f} 0">
      <joint name="{rid}_sx" type="slide" axis="1 0 0" damping="2"/>
      <joint name="{rid}_sz" type="slide" axis="0 0 1" damping="2"/>
      <joint name="{rid}_yaw" type="hinge" axis="0 1 0" damping="0.4"/>
      {geom}
    </body>"""
        )
    for i in range(n_pieces):
        name = f"piece_{i:02d}"
        # Body at the origin so slide qpos is world XYZ (Y-up inches).
        bodies.append(
            f"""    <body name="{name}" pos="0 0 0">
      <joint name="{name}_sx" type="slide" axis="1 0 0" damping="0.05"/>
      <joint name="{name}_sy" type="slide" axis="0 1 0" damping="0.02"/>
      <joint name="{name}_sz" type="slide" axis="0 0 1" damping="0.05"/>
      <geom name="{name}_geom" type="sphere" size="1.4" mass="0.05" rgba="0.9 0.8 0.2 1" contype="3" conaffinity="3" friction="0.6 0.05 0.01"/>
    </body>"""
        )

    compiler = '<compiler angle="radian" inertiafromgeom="true"/>'
    assets = ""
    if mesh_file and meshdir:
        compiler = f'<compiler angle="radian" inertiafromgeom="true" meshdir="{escape(meshdir)}"/>'
        assets = f"""
  <asset>
    <mesh name="robot_hull" file="{escape(mesh_file)}"/>
  </asset>"""
    xml = f"""<mujoco model="talongym_field">
  {compiler}
  <option gravity="0 {-IN_G:.4f} 0" timestep="0.01" integrator="Euler" cone="pyramidal"/>
  <default>
    <geom condim="3" solref="0.02 1" solimp="0.9 0.95 0.001"/>
  </default>{assets}
  <worldbody>
{chr(10).join(geoms)}
{chr(10).join(bodies)}
  </worldbody>
</mujoco>
"""
    return xml
