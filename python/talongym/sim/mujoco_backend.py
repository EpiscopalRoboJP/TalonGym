"""MuJoCo adapters: planar validation plus mesh-field 3D (Y-up inches)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from talongym.assets.mjcf_field import GEOM_GROUP_FIELD, GEOM_GROUP_PIECE, GEOM_GROUP_ROBOT
from talongym.sim.physics import (
    NM_TO_INCH_TORQUE,
    Body,
    ContactSet,
    WorldStep,
    _clip_chassis,
    _resolve_chassis,
    aerodynamic_wrench,
    mechanism_piece_force,
    relative_velocity_robot,
    rotate_robot_to_world,
    world_to_robot_xy,
)

_MJ_MODEL_CACHE: dict[str, Any] = {}
# Conservative world-frame radius covering chassis, intake reach, and muzzle.
_MECHANISM_FORCE_REACH_IN = 40.0
_MECHANISM_FORCE_REACH2 = _MECHANISM_FORCE_REACH_IN * _MECHANISM_FORCE_REACH_IN


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


class MeshFieldRequiredError(RuntimeError):
    """Raised when a field requires mesh collision but MuJoCo is not installed."""


class PhysicalSimulationFault(RuntimeError):
    """The authoritative simulation reached a physically invalid state."""


def ftc_to_mj(x: float, y: float, z: float) -> tuple[float, float, float]:
    return float(x), float(z), float(-y)


def mj_to_ftc(mx: float, my: float, mz: float) -> tuple[float, float, float]:
    return float(mx), float(-mz), float(my)


def ftc_yaw_to_mj_quat(heading_rad: float) -> tuple[float, float, float, float]:
    """FTC yaw about vertical → MuJoCo Y-up quaternion (w, x, y, z)."""
    half = 0.5 * float(heading_rad)
    return math.cos(half), 0.0, math.sin(half), 0.0


def mj_quat_to_ftc_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    return math.atan2(2.0 * (qw * qy - qz * qx), 1.0 - 2.0 * (qy * qy + qz * qz))


def ftc_omega_to_mj(wx: float, wy: float, wz: float) -> tuple[float, float, float]:
    return float(wx), float(wz), float(-wy)


def mj_omega_to_ftc(wx: float, wy: float, wz: float) -> tuple[float, float, float]:
    return float(wx), float(-wz), float(wy)


def _compile_mj_model(xml: str, xml_path: Path | None):
    import hashlib

    import mujoco

    key = str(xml_path.resolve()) if xml_path is not None else hashlib.sha256(xml.encode("utf-8")).hexdigest()
    model = _MJ_MODEL_CACHE.get(key)
    if model is None:
        if xml_path is not None:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
        else:
            model = mujoco.MjModel.from_xml_string(xml)
        _MJ_MODEL_CACHE[key] = model
    return mujoco, model


class MujocoFieldBackend:
    """3D mesh field: planar chassis joints, free CAD pieces, gravity on.

    Coordinates: FTC inches, MuJoCo Y-up (x=ftc.x, y=height, z=-ftc.y).
    Piece slots are pooled by typeId so each scoring-element hull is used.
    Contacts use geom groups (field/robot/piece), not name substrings.
    """

    name = "mujoco_field"

    def __init__(
        self,
        xml: str,
        field_half_w: float,
        field_half_d: float,
        robot_hz: float = 5.0,
        xml_path: Path | None = None,
        slot_plan: dict[str, int] | None = None,
        floor_y: float = 0.0,
        cad_stats: dict[str, Any] | None = None,
    ) -> None:
        if not available():
            raise MeshFieldRequiredError("MuJoCo extra missing; pip install -e '.[mujoco]'")
        self.hw = field_half_w
        self.hd = field_half_d
        self.robot_hz = robot_hz
        self.floor_y = float(floor_y)
        self.cad_stats = dict(cad_stats or {})
        self._mujoco, self._mj = _compile_mj_model(xml, xml_path)
        self._data = self._mujoco.MjData(self._mj)
        self._contacts = [ContactSet()]
        self._robot_jnt: dict[str, tuple[int, int, int]] = {}
        self._piece_body: list[int] = []
        self._piece_qpos: list[int] = []
        self._piece_qvel: list[int] = []
        self._piece_nq: list[int] = []
        self._piece_nv: list[int] = []
        self._piece_type: list[str] = []
        self._pools: dict[str, list[int]] = {}
        self._owned: set[str] = set()
        self._id_to_slot: dict[str, int] = {}
        self._robot_placed: set[str] = set()
        self._robot_mechanism_placed: set[str] = set()
        self._field_mechanism_joints: dict[str, tuple[int, int]] = {}
        self._robot_mechanism_joints: dict[tuple[str, str], tuple[int, int]] = {}
        self._robot_part_body: dict[tuple[str, str], int] = {}
        self._robot_body_id: dict[str, int] = {}
        self._robot_dof: dict[str, int] = {}
        self._body_name_by_id: list[str] = []
        self._floor_geom_id = -1
        self._robot_targets: dict[str, tuple[float, float, float, float, float, float]] = {}
        mujoco = self._mujoco
        try:
            enable = int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
            self._mj.opt.enableflags = int(self._mj.opt.enableflags) | enable
        except Exception:
            pass
        for bid in ("red_0", "red_1", "blue_0", "blue_1"):
            try:
                jx = mujoco.mj_name2id(self._mj, mujoco.mjtObj.mjOBJ_JOINT, f"{bid}_sx")
                jz = mujoco.mj_name2id(self._mj, mujoco.mjtObj.mjOBJ_JOINT, f"{bid}_sz")
                jy = mujoco.mj_name2id(self._mj, mujoco.mjtObj.mjOBJ_JOINT, f"{bid}_yaw")
            except Exception:
                continue
            if min(jx, jz, jy) < 0:
                continue
            self._robot_jnt[bid] = (
                int(self._mj.jnt_qposadr[jx]),
                int(self._mj.jnt_qposadr[jz]),
                int(self._mj.jnt_qposadr[jy]),
            )
            self._robot_dof[bid] = int(self._mj.jnt_dofadr[jx])
        for joint_id in range(int(self._mj.njnt)):
            raw = mujoco.mj_id2name(self._mj, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            name = str(raw or "")
            if not name.startswith("field_mech_"):
                for robot_id in ("red_0", "red_1", "blue_0", "blue_1"):
                    prefix = f"{robot_id}_joint_"
                    if name.startswith(prefix):
                        self._robot_mechanism_joints[
                            (robot_id, name[len(prefix) :])
                        ] = (
                            int(self._mj.jnt_qposadr[joint_id]),
                            int(self._mj.jnt_dofadr[joint_id]),
                        )
                        break
            else:
                mechanism_id = name[len("field_mech_") :]
                self._field_mechanism_joints[mechanism_id] = (
                    int(self._mj.jnt_qposadr[joint_id]),
                    int(self._mj.jnt_dofadr[joint_id]),
                )
        nbody = int(self._mj.nbody)
        self._body_name_by_id = [""] * nbody
        for body_id in range(nbody):
            raw = mujoco.mj_id2name(self._mj, mujoco.mjtObj.mjOBJ_BODY, body_id)
            name = str(raw or "")
            chassis_name = name.split("_part_", 1)[0] if "_part_" in name else name
            self._body_name_by_id[body_id] = chassis_name
            if body_id == 0:
                continue
            if name in ("red_0", "red_1", "blue_0", "blue_1"):
                self._robot_body_id[name] = int(body_id)
            if "_part_" in name:
                robot_id, part_id = name.split("_part_", 1)
                self._robot_part_body[(robot_id, part_id)] = int(body_id)
        floor = mujoco.mj_name2id(self._mj, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._floor_geom_id = int(floor) if int(floor) >= 0 else -1
        type_ids = [tid for tid in (slot_plan or {}) if tid]
        self._index_piece_bodies(type_ids)
        self._park_all_pieces()

    def _index_piece_bodies(self, type_ids: list[str]) -> None:
        mujoco = self._mujoco
        nbody = int(self._mj.nbody)
        ordered_types = sorted(type_ids, key=len, reverse=True)
        for bid in range(1, nbody):
            raw = mujoco.mj_id2name(self._mj, mujoco.mjtObj.mjOBJ_BODY, bid)
            name = str(raw or "")
            parsed = self._parse_piece_body(name, ordered_types)
            if parsed is None:
                continue
            type_id, _idx = parsed
            jnt = int(self._mj.body_jntadr[bid])
            if jnt < 0:
                continue
            slot = len(self._piece_body)
            nq = int(self._mj.jnt_qposadr[jnt + 1] - self._mj.jnt_qposadr[jnt]) if jnt + 1 < int(self._mj.njnt) else 7
            nv = int(self._mj.jnt_dofadr[jnt + 1] - self._mj.jnt_dofadr[jnt]) if jnt + 1 < int(self._mj.njnt) else 6
            # Last joint in the model: jnt+1 is out of range; infer from type.
            jnt_type = int(self._mj.jnt_type[jnt])
            if jnt_type == int(mujoco.mjtJoint.mjJNT_FREE):
                nq, nv = 7, 6
            elif nq <= 0 or nv <= 0:
                nq, nv = 3, 3
            self._piece_body.append(int(bid))
            self._piece_qpos.append(int(self._mj.jnt_qposadr[jnt]))
            self._piece_qvel.append(int(self._mj.jnt_dofadr[jnt]))
            self._piece_nq.append(nq)
            self._piece_nv.append(nv)
            self._piece_type.append(type_id)
            self._pools.setdefault(type_id, []).append(slot)

    @staticmethod
    def _parse_piece_body(name: str, type_ids: list[str]) -> tuple[str, int] | None:
        if name.startswith("gp_"):
            rest = name[3:]
            for tid in type_ids:
                prefix = f"{tid}_"
                if rest.startswith(prefix):
                    tail = rest[len(prefix) :]
                    if tail.isdigit():
                        return tid, int(tail)
            return None
        if name.startswith("piece_"):
            tail = name[6:]
            if tail.isdigit():
                return "", int(tail)
        return None

    def _park_slot(self, slot: int) -> None:
        adr = self._piece_qpos[slot]
        vel = self._piece_qvel[slot]
        nq = self._piece_nq[slot]
        nv = self._piece_nv[slot]
        self._data.qpos[adr] = 180.0 + slot * 3.0
        self._data.qpos[adr + 1] = 80.0
        self._data.qpos[adr + 2] = 0.0
        if nq >= 7:
            self._data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self._data.qvel[vel : vel + nv] = 0.0

    def reseat_piece(self, piece_id: str) -> None:
        """Allow the next write to move a live piece (magazine-to-muzzle feed)."""
        self._owned.discard(str(piece_id))

    def _park_all_pieces(self) -> None:
        for i in range(len(self._piece_qpos)):
            self._park_slot(i)
        self._owned.clear()
        self._id_to_slot.clear()

    def _park_robot(self, bid: str, slot: int) -> None:
        addrs = self._robot_jnt.get(bid)
        if addrs is None:
            return
        ax, az, ay = addrs
        self._data.qpos[ax] = 220.0 + slot * 24.0
        self._data.qpos[az] = 220.0
        self._data.qpos[ay] = 0.0
        dof = self._robot_dof.get(bid)
        if dof is not None:
            self._data.qvel[dof : dof + 3] = 0.0

    def _park_unused_robots(self, live_ids: set[str]) -> None:
        for i, bid in enumerate(self._robot_jnt):
            if bid not in live_ids:
                self._park_robot(bid, i)
                self._robot_placed.discard(bid)

    def reset_batch(self, n: int) -> None:
        self._contacts = [ContactSet() for _ in range(max(1, n))]
        self._mujoco.mj_resetData(self._mj, self._data)
        self._park_all_pieces()
        for i, bid in enumerate(self._robot_jnt):
            self._park_robot(bid, i)
        self._robot_placed.clear()
        self._robot_mechanism_placed.clear()

    def field_mechanism_positions(self) -> dict[str, float]:
        return {
            mechanism_id: float(self._data.qpos[qpos])
            for mechanism_id, (qpos, _dof) in self._field_mechanism_joints.items()
        }

    def robot_mechanism_transforms(self) -> dict[str, list[dict[str, Any]]]:
        transforms: dict[str, list[dict[str, Any]]] = {}
        for body_id in range(1, int(self._mj.nbody)):
            raw = self._mujoco.mj_id2name(
                self._mj,
                self._mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )
            name = str(raw or "")
            if "_part_" not in name:
                continue
            robot_id, part_id = name.split("_part_", 1)
            position = self._data.xpos[body_id]
            x, y, z = mj_to_ftc(
                float(position[0]),
                float(position[1]),
                float(position[2]),
            )
            quat = self._data.xquat[body_id]
            cvel = self._data.cvel[body_id]
            vx, vy, vz = mj_to_ftc(float(cvel[3]), float(cvel[4]), float(cvel[5]))
            transforms.setdefault(robot_id, []).append(
                {
                    "id": part_id,
                    "x": x,
                    "y": y,
                    "z": z,
                    "qw": float(quat[0]),
                    "qx": float(quat[1]),
                    "qy": float(quat[2]),
                    "qz": float(quat[3]),
                    "vx": vx,
                    "vy": vy,
                    "vz": vz,
                }
            )
        return transforms

    def robot_mechanism_joint_states(self) -> dict[str, dict[str, dict[str, float]]]:
        states: dict[str, dict[str, dict[str, float]]] = {}
        for (robot_id, joint_id), (qpos, dof) in self._robot_mechanism_joints.items():
            states.setdefault(robot_id, {})[joint_id] = {
                "positionRad": float(self._data.qpos[qpos]),
                "velocityRadS": float(self._data.qvel[dof]),
            }
        return states

    def _slot_for(self, pid: str, type_id: str = "") -> int | None:
        if pid in self._id_to_slot:
            return self._id_to_slot[pid]
        pool = self._pools.get(type_id)
        if pool is None and type_id:
            pool = self._pools.get("")
        if pool is None:
            pool = next(iter(self._pools.values()), list(range(len(self._piece_qpos))))
        used = set(self._id_to_slot.values())
        for slot in pool:
            if slot not in used:
                self._id_to_slot[pid] = slot
                return slot
        return None

    def _write_robot(self, body: Body) -> None:
        addrs = self._robot_jnt.get(body.id)
        if addrs is None:
            return
        ax, az, ay = addrs
        if body.id not in self._robot_placed:
            mx, _my, mz = ftc_to_mj(body.x, body.y, self.robot_hz)
            self._data.qpos[ax] = mx
            self._data.qpos[az] = mz
            self._data.qpos[ay] = body.heading
            self._robot_placed.add(body.id)
        self._robot_targets[body.id] = (
            float(body.vx),
            -float(body.vy),
            float(body.omega),
            max(0.1, float(body.mass)),
            max(0.1, float(body.hx)),
            max(0.1, float(body.hy)),
        )

    def _write_piece(self, body: Body) -> None:
        slot = self._slot_for(body.id, getattr(body, "type_id", "") or "")
        if slot is None:
            return
        adr = self._piece_qpos[slot]
        vel = self._piece_qvel[slot]
        nq = self._piece_nq[slot]
        nv = self._piece_nv[slot]
        z = float(getattr(body, "z", body.radius or 1.4) or 1.4)
        mx, my, mz = ftc_to_mj(body.x, body.y, z)
        if body.id not in self._owned:
            self._data.qpos[adr : adr + 3] = [mx, my, mz]
            if nq >= 7:
                qw = float(getattr(body, "qw", 1.0) or 1.0)
                qx = float(getattr(body, "qx", 0.0) or 0.0)
                qy = float(getattr(body, "qy", 0.0) or 0.0)
                qz = float(getattr(body, "qz", 0.0) or 0.0)
                if abs(qw) + abs(qx) + abs(qy) + abs(qz) < 1e-9:
                    qw, qx, qy, qz = ftc_yaw_to_mj_quat(float(body.heading))
                self._data.qpos[adr + 3 : adr + 7] = [qw, qx, qy, qz]
            vz = float(getattr(body, "vz", 0.0) or 0.0)
            self._data.qvel[vel : vel + 3] = [body.vx, vz, -body.vy]
            if nv >= 6:
                wx, wy, wz = ftc_omega_to_mj(
                    float(getattr(body, "wx", 0.0) or 0.0),
                    float(getattr(body, "wy", 0.0) or 0.0),
                    float(getattr(body, "wz", 0.0) or 0.0),
                )
                self._data.qvel[vel + 3 : vel + 6] = [wx, wy, wz]
            self._owned.add(body.id)

    def _read_robot(self, body: Body) -> None:
        addrs = self._robot_jnt.get(body.id)
        if addrs is None:
            return
        ax, az, ay = addrs
        bid = self._robot_body_id.get(body.id)
        if bid is not None:
            xpos = self._data.xpos[bid]
            x, y, _z = mj_to_ftc(float(xpos[0]), float(xpos[1]), float(xpos[2]))
            body.x, body.y = x, y
        else:
            x, y, _z = mj_to_ftc(float(self._data.qpos[ax]), self.robot_hz, float(self._data.qpos[az]))
            body.x, body.y = x, y
        body.heading = float(self._data.qpos[ay])
        body.z = self.robot_hz
        dof = self._robot_dof.get(body.id)
        if dof is None:
            return
        body.vx = float(self._data.qvel[dof])
        body.vy = -float(self._data.qvel[dof + 1])
        body.omega = float(self._data.qvel[dof + 2])

    def _read_piece(self, body: Body) -> None:
        slot = self._id_to_slot.get(body.id)
        if slot is None:
            return
        bid = self._piece_body[slot]
        xpos = self._data.xpos[bid]
        x, y, z = mj_to_ftc(float(xpos[0]), float(xpos[1]), float(xpos[2]))
        body.x, body.y, body.z = x, y, z
        vel = self._piece_qvel[slot]
        nq = self._piece_nq[slot]
        nv = self._piece_nv[slot]
        body.vx = float(self._data.qvel[vel])
        body.vz = float(self._data.qvel[vel + 1])
        body.vy = -float(self._data.qvel[vel + 2])
        if nq >= 7:
            xquat = self._data.xquat[bid]
            body.qw, body.qx, body.qy, body.qz = (float(xquat[0]), float(xquat[1]), float(xquat[2]), float(xquat[3]))
            body.heading = mj_quat_to_ftc_yaw(body.qw, body.qx, body.qy, body.qz)
        if nv >= 6:
            local = np.array(
                [float(self._data.qvel[vel + 3]), float(self._data.qvel[vel + 4]), float(self._data.qvel[vel + 5])],
                dtype=float,
            )
            rot = np.asarray(self._data.xmat[bid], dtype=float).reshape(3, 3)
            world = rot @ local
            body.wx, body.wy, body.wz = mj_omega_to_ftc(float(world[0]), float(world[1]), float(world[2]))
        # Competitive mesh simulation fails closed instead of teleporting a
        # tunneled body back onto the field.
        min_z = self.floor_y - 4.0
        if body.z < min_z:
            raise PhysicalSimulationFault(
                f"piece {body.id} tunneled below the field floor: z={body.z:.4f}"
            )

    def _geom_group(self, geom_id: int) -> int:
        if geom_id < 0:
            return -1
        return int(self._mj.geom_group[geom_id])

    def _body_name(self, geom_id: int) -> str:
        if geom_id < 0:
            return ""
        bid = int(self._mj.geom_bodyid[geom_id])
        if 0 <= bid < len(self._body_name_by_id):
            return self._body_name_by_id[bid]
        return ""

    def _contact_flags(self, robot_ids: set[str], piece_ids: set[str]) -> ContactSet:
        flags = ContactSet()
        live_robots = set(robot_ids)

        def _is_live_robot(name: str) -> bool:
            if name in live_robots:
                return True
            return any(name.startswith(f"{rid}_") for rid in live_robots)

        ncon = int(self._data.ncon)
        for i in range(ncon):
            c = self._data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == self._floor_geom_id or g2 == self._floor_geom_id:
                continue
            grp1, grp2 = self._geom_group(g1), self._geom_group(g2)
            b1, b2 = self._body_name(g1), self._body_name(g2)
            hits_live_robot = _is_live_robot(b1) or _is_live_robot(b2)
            hits_piece = GEOM_GROUP_PIECE in {grp1, grp2}
            hits_field = GEOM_GROUP_FIELD in {grp1, grp2}
            if hits_live_robot and hits_field:
                flags.wall = True
            if grp1 == GEOM_GROUP_ROBOT and grp2 == GEOM_GROUP_ROBOT and _is_live_robot(b1) and _is_live_robot(b2) and b1 != b2:
                flags.robot = True
            if hits_piece and (hits_field or hits_live_robot):
                flags.piece = True
        return flags

    def _robot_pose_ftc(self, robot_id: str) -> tuple[float, float, float, float, float, float] | None:
        addrs = self._robot_jnt.get(robot_id)
        if addrs is None:
            return None
        ax, az, ay = addrs
        x, y, _z = mj_to_ftc(float(self._data.qpos[ax]), self.robot_hz, float(self._data.qpos[az]))
        heading = float(self._data.qpos[ay])
        dof = self._robot_dof.get(robot_id)
        if dof is None:
            return x, y, heading, 0.0, 0.0, 0.0
        return (
            x,
            y,
            heading,
            float(self._data.qvel[dof]),
            -float(self._data.qvel[dof + 1]),
            float(self._data.qvel[dof + 2]),
        )

    def _piece_spin_ftc(self, slot: int) -> tuple[float, float, float]:
        if self._piece_nv[slot] < 6:
            return 0.0, 0.0, 0.0
        bid = self._piece_body[slot]
        vel = self._piece_qvel[slot]
        lx = float(self._data.qvel[vel + 3])
        ly = float(self._data.qvel[vel + 4])
        lz = float(self._data.qvel[vel + 5])
        mat = self._data.xmat[bid]
        wx = float(mat[0]) * lx + float(mat[1]) * ly + float(mat[2]) * lz
        wy = float(mat[3]) * lx + float(mat[4]) * ly + float(mat[5]) * lz
        wz = float(mat[6]) * lx + float(mat[7]) * ly + float(mat[8]) * lz
        return mj_omega_to_ftc(wx, wy, wz)

    def _aero_coefficients(self, state: WorldStep, owner_id: str | None) -> tuple[float, float]:
        mechanism = None
        if owner_id:
            mechanism = state.robot_mechanism_states.get(owner_id)
        if mechanism is None and state.robot_mechanism_states:
            mechanism = next(iter(state.robot_mechanism_states.values()))
        path = dict((mechanism or {}).get("piecePath") or {})
        return (
            float(path.get("dragCoefficient") or 0.47),
            float(path.get("magnusCoefficient") or 0.12),
        )

    def _apply_piece_flow_forces(self, state: WorldStep) -> None:
        robot_rows: list[tuple[str, dict[str, Any], tuple[float, float, float, float, float, float], float]] = []
        for robot_id, mechanism in state.robot_mechanism_states.items():
            pose = self._robot_pose_ftc(robot_id)
            if pose is None:
                continue
            chassis = mechanism.get("chassis") or {}
            height = float(chassis.get("heightIn") or 14.0)
            robot_rows.append((robot_id, mechanism, pose, height))
        flywheel_loads: dict[str, float] = {}
        recoil: dict[str, tuple[float, float, float]] = {}
        for piece in state.pieces:
            slot = self._id_to_slot.get(piece.id)
            if slot is None:
                continue
            bid = self._piece_body[slot]
            xpos = self._data.xpos[bid]
            px, py, pz = mj_to_ftc(float(xpos[0]), float(xpos[1]), float(xpos[2]))
            vel = self._piece_qvel[slot]
            vx = float(self._data.qvel[vel])
            vz = float(self._data.qvel[vel + 1])
            vy = -float(self._data.qvel[vel + 2])
            mass = max(float(piece.mass), 1e-4)
            radius = float(piece.radius or 1.4)
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            airborne = pz > self.floor_y + radius + 0.75 or speed > 36.0
            fx = fy = fz = tx = ty = tz = 0.0
            owner_id = state.piece_owners.get(piece.id)

            for robot_id, mechanism, pose, height in robot_rows:
                rx, ry, heading, rvx, rvy, omega = pose
                dx = px - rx
                dy = py - ry
                if owner_id != robot_id and dx * dx + dy * dy > _MECHANISM_FORCE_REACH2:
                    continue
                if airborne and pz > height + 2.0 and speed > 24.0:
                    continue
                lx, ly = world_to_robot_xy(px, py, rx, ry, heading)
                lvx, lvy, lvz = relative_velocity_robot(
                    vx, vy, vz, rvx, rvy, omega, heading, dx, dy
                )
                interaction = mechanism_piece_force(
                    piece_local=(lx, ly, pz),
                    piece_vel_local=(lvx, lvy, lvz),
                    piece_radius=radius,
                    piece_mass=mass,
                    mechanism=mechanism,
                    owned=owner_id == robot_id,
                )
                if (
                    abs(interaction.fx)
                    + abs(interaction.fy)
                    + abs(interaction.fz)
                    + abs(interaction.flywheel_load_inch)
                    < 1e-9
                ):
                    continue
                wfx, wfy = rotate_robot_to_world(interaction.fx, interaction.fy, heading)
                fx += wfx
                fy += wfy
                fz += interaction.fz
                wtx, wty = rotate_robot_to_world(interaction.tx, interaction.ty, heading)
                tx += wtx
                ty += wty
                tz += interaction.tz
                flywheel_loads[robot_id] = (
                    flywheel_loads.get(robot_id, 0.0) + interaction.flywheel_load_inch
                )
                prev = recoil.get(robot_id, (0.0, 0.0, 0.0))
                recoil[robot_id] = (
                    prev[0] - wfx,
                    prev[1] - wfy,
                    prev[2] + dx * (-wfy) - dy * (-wfx),
                )

            if airborne:
                wx, wy, wz = self._piece_spin_ftc(slot)
                drag_c, magnus_c = self._aero_coefficients(state, owner_id)
                ax, ay, az, atx, aty, atz = aerodynamic_wrench(
                    vx,
                    vy,
                    vz,
                    wx,
                    wy,
                    wz,
                    radius,
                    drag_coefficient=drag_c,
                    magnus_coefficient=magnus_c,
                )
                fx += ax
                fy += ay
                fz += az
                tx += atx
                ty += aty
                tz += atz

            if abs(fx) + abs(fy) + abs(fz) + abs(tx) + abs(ty) + abs(tz) < 1e-12:
                continue
            mx, my, mz = ftc_to_mj(fx, fy, fz)
            twx, twy, twz = ftc_to_mj(tx, ty, tz)
            self._data.xfrc_applied[bid, 0] += mx
            self._data.xfrc_applied[bid, 1] += my
            self._data.xfrc_applied[bid, 2] += mz
            self._data.xfrc_applied[bid, 3] += twx
            self._data.xfrc_applied[bid, 4] += twy
            self._data.xfrc_applied[bid, 5] += twz

        for robot_id, load in flywheel_loads.items():
            mechanism = state.robot_mechanism_states.get(robot_id) or {}
            path = mechanism.get("piecePath") or {}
            flywheel = (mechanism.get("actuators") or {}).get(str(path.get("flywheelActuatorId") or "")) or {}
            joint_id = str(flywheel.get("jointId") or "")
            binding = self._robot_mechanism_joints.get((robot_id, joint_id))
            if binding is None:
                continue
            _qpos, dof = binding
            self._data.qfrc_applied[dof] += load

        for robot_id, (rfx, rfy, rtau) in recoil.items():
            dof = self._robot_dof.get(robot_id)
            if dof is None:
                continue
            mx, _my, mz = ftc_to_mj(rfx, rfy, 0.0)
            self._data.qfrc_applied[dof] += mx
            self._data.qfrc_applied[dof + 1] += mz
            self._data.qfrc_applied[dof + 2] += rtau

    def step_batch(self, states: list[WorldStep], dt: float) -> list[ContactSet]:
        out: list[ContactSet] = []
        for state in states:
            out.append(self._step_one(state, dt))
        self._contacts = out or [ContactSet()]
        return self._contacts

    def _step_one(self, state: WorldStep, dt: float) -> ContactSet:
        live_piece_ids = {p.id for p in state.pieces}
        live_robot_ids = {b.id for b in state.robots}
        self._park_unused_robots(live_robot_ids)
        for pid in list(self._owned):
            if pid not in live_piece_ids:
                slot = self._id_to_slot.pop(pid, None)
                self._owned.discard(pid)
                if slot is not None:
                    self._park_slot(slot)
        for body in state.robots:
            _clip_chassis(body, state.max_vel, state.max_omega)
            self._write_robot(body)
        for piece in state.pieces:
            self._write_piece(piece)
        for robot_id, mechanism in state.robot_mechanism_states.items():
            if robot_id in self._robot_mechanism_placed:
                continue
            for actuator in (mechanism.get("actuators") or {}).values():
                if actuator.get("kind") == "velocity_motor":
                    continue
                joint_id = str(actuator.get("jointId") or "")
                binding = self._robot_mechanism_joints.get((robot_id, joint_id))
                if binding is None:
                    continue
                qpos, dof = binding
                self._data.qpos[qpos] = math.radians(
                    float(actuator.get("position") or 0.0)
                )
                self._data.qvel[dof] = 0.0
            self._robot_mechanism_placed.add(robot_id)
        self._mujoco.mj_forward(self._mj, self._data)
        timestep = max(float(self._mj.opt.timestep), 1e-6)
        nsub = max(1, int(round(dt / max(timestep, 1e-4))))
        drive_targets: list[tuple[int, tuple[float, float, float, float, float, float]]] = []
        for robot_id, target in self._robot_targets.items():
            dof = self._robot_dof.get(robot_id)
            if dof is None:
                continue
            drive_targets.append((dof, target))
        actuator_torques: list[tuple[int, float]] = []
        for robot_id, mechanism in state.robot_mechanism_states.items():
            for actuator in (mechanism.get("actuators") or {}).values():
                joint_id = str(actuator.get("jointId") or "")
                binding = self._robot_mechanism_joints.get((robot_id, joint_id))
                if binding is None:
                    continue
                actuator_torques.append(
                    (binding[1], float(actuator.get("torqueNm") or 0.0) * NM_TO_INCH_TORQUE)
                )
        max_accel = float(state.max_accel)
        max_ang_accel = float(state.max_ang_accel)
        for _ in range(nsub):
            self._data.qfrc_applied[:] = 0.0
            self._data.xfrc_applied[:] = 0.0
            for dof, target in drive_targets:
                vx, vz, omega, mass, hx, hy = target
                max_force = mass * max_accel
                fx = mass * (vx - float(self._data.qvel[dof])) / timestep
                fz_force = mass * (vz - float(self._data.qvel[dof + 1])) / timestep
                self._data.qfrc_applied[dof] = max(-max_force, min(max_force, fx))
                self._data.qfrc_applied[dof + 1] = max(-max_force, min(max_force, fz_force))
                inertia = mass * (hx * hx + hy * hy) / 3.0
                max_torque = inertia * max_ang_accel
                tau = inertia * (omega - float(self._data.qvel[dof + 2])) / timestep
                self._data.qfrc_applied[dof + 2] = max(-max_torque, min(max_torque, tau))
            for mechanism_id, (qpos, dof) in self._field_mechanism_joints.items():
                target = float(state.field_mechanism_targets.get(mechanism_id, 0.0))
                error = target - float(self._data.qpos[qpos])
                velocity = float(self._data.qvel[dof])
                effort = 6000.0 * error - 1700.0 * velocity
                self._data.qfrc_applied[dof] = max(-6000.0, min(6000.0, effort))
            for dof, torque in actuator_torques:
                self._data.qfrc_applied[dof] += torque
            self._apply_piece_flow_forces(state)
            self._mujoco.mj_step(self._mj, self._data)
            for mechanism_id, (qpos, dof) in self._field_mechanism_joints.items():
                target = float(state.field_mechanism_targets.get(mechanism_id, 0.0))
                if abs(target) < 1e-9:
                    # The real HIVE is latched before a scored tip; staged
                    # NECTAR must not sag the inactive basket under gravity.
                    self._data.qpos[qpos] = 0.0
                    self._data.qvel[dof] = 0.0
        for body in state.robots:
            self._read_robot(body)
        for piece in state.pieces:
            self._read_piece(piece)
        flags = self._contact_flags({b.id for b in state.robots if b.dynamic}, live_piece_ids)
        for body in state.robots:
            if not body.dynamic:
                continue
            w, _r = _resolve_chassis(body, state.walls, state.robots, self.hw, self.hd)
            flags.wall = flags.wall or w
        return flags

    def contacts(self, index: int = 0) -> ContactSet:
        return self._contacts[min(index, len(self._contacts) - 1)]

    def step_world(self, state: WorldStep, dt: float) -> ContactSet:
        return self.step_batch([state], dt)[0]

