"""Slow MuJoCo chassis validation adapter. Never used by train_ppo."""

from __future__ import annotations

from typing import Any

import numpy as np

from talongym.sim.physics import Body, ContactSet, WorldStep, _clip_chassis, _resolve_chassis


class MujocoValidationBackend:
    name = "mujoco"

    def __init__(self, field_half_w: float, field_half_d: float) -> None:
        self.hw = field_half_w
        self.hd = field_half_d
        self._mj = None
        self._data = None
        self._contacts = [ContactSet()]
        try:
            import mujoco

            xml = f"""
            <mujoco>
              <option gravity="0 0 0" timestep="0.02"/>
              <worldbody>
                <geom type="plane" size="{field_half_w} {field_half_d} 0.1"/>
                <body name="chassis" pos="0 0 0.1">
                  <joint name="slide_x" type="slide" axis="1 0 0"/>
                  <joint name="slide_y" type="slide" axis="0 1 0"/>
                  <joint name="yaw" type="hinge" axis="0 0 1"/>
                  <geom type="box" size="9 9 5" mass="15"/>
                </body>
              </worldbody>
            </mujoco>
            """
            self._mj = mujoco.MjModel.from_xml_string(xml)
            self._data = mujoco.MjData(self._mj)
            self._mujoco = mujoco
        except Exception:
            self._mj = None

    def reset_batch(self, n: int) -> None:
        self._contacts = [ContactSet() for _ in range(max(1, n))]
        if self._data is not None:
            self._mujoco.mj_resetData(self._mj, self._data)

    def step_batch(self, states: list[WorldStep], dt: float) -> list[ContactSet]:
        out: list[ContactSet] = []
        for state in states:
            for body in state.robots:
                _clip_chassis(body, state.max_vel, state.max_omega)
                if self._data is not None and body.id == "red_0":
                    self._data.qvel[0] = body.vx
                    self._data.qvel[1] = body.vy
                    self._data.qvel[2] = body.omega
                    self._mujoco.mj_step(self._mj, self._data)
                    body.x = float(self._data.qpos[0])
                    body.y = float(self._data.qpos[1])
                    body.heading = float(self._data.qpos[2])
                else:
                    body.x += body.vx * dt
                    body.y += body.vy * dt
                    body.heading += body.omega * dt
                w, r = _resolve_chassis(body, state.walls + state.obstacles, state.robots, self.hw, self.hd)
                out.append(ContactSet(wall=w, robot=r, piece=False))
        self._contacts = out or [ContactSet()]
        return self._contacts

    def contacts(self, index: int = 0) -> ContactSet:
        return self._contacts[min(index, len(self._contacts) - 1)]

    def step_world(self, state: WorldStep, dt: float) -> ContactSet:
        return self.step_batch([state], dt)[0]


def available() -> bool:
    try:
        import mujoco  # noqa: F401

        return True
    except ImportError:
        return False


def pose_rmse(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    err = np.asarray(a[:n]) - np.asarray(b[:n])
    return float(np.sqrt(np.mean(err**2)))
