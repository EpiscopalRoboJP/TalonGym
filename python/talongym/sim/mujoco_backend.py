"""MuJoCo adapters: planar validation plus mesh-field 3D (Y-up inches)."""

from __future__ import annotations

import math
from typing import Any
from pathlib import Path

import numpy as np

from talongym.assets.mjcf_field import GEOM_GROUP_FIELD, GEOM_GROUP_PIECE, GEOM_GROUP_ROBOT
from talongym.sim.physics import Body, ContactSet, WorldStep, _clip_chassis, _resolve_chassis

_MJ_MODEL_CACHE: dict[str, Any] = {}


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
        self._field_mechanism_joints: dict[str, tuple[int, int]] = {}
        mujoco = self._mujoco
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
        for joint_id in range(int(self._mj.njnt)):
            raw = mujoco.mj_id2name(self._mj, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            name = str(raw or "")
            if not name.startswith("field_mech_"):
                continue
            mechanism_id = name[len("field_mech_") :]
            self._field_mechanism_joints[mechanism_id] = (
                int(self._mj.jnt_qposadr[joint_id]),
                int(self._mj.jnt_dofadr[joint_id]),
            )
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
        jx = self._mujoco.mj_name2id(self._mj, self._mujoco.mjtObj.mjOBJ_JOINT, f"{bid}_sx")
        if jx >= 0:
            dof = int(self._mj.jnt_dofadr[jx])
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

    def field_mechanism_positions(self) -> dict[str, float]:
        return {
            mechanism_id: float(self._data.qpos[qpos])
            for mechanism_id, (qpos, _dof) in self._field_mechanism_joints.items()
        }

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
        jx = self._mujoco.mj_name2id(self._mj, self._mujoco.mjtObj.mjOBJ_JOINT, f"{body.id}_sx")
        dof = int(self._mj.jnt_dofadr[jx])
        self._data.qvel[dof] = body.vx
        self._data.qvel[dof + 1] = -body.vy
        self._data.qvel[dof + 2] = body.omega

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
        kick = bool(getattr(body, "kick", False))
        if body.id not in self._owned or kick:
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
        try:
            bid = self._mujoco.mj_name2id(self._mj, self._mujoco.mjtObj.mjOBJ_BODY, body.id)
            xpos = self._data.xpos[bid]
            x, y, _z = mj_to_ftc(float(xpos[0]), float(xpos[1]), float(xpos[2]))
            body.x, body.y = x, y
        except Exception:
            x, y, _z = mj_to_ftc(float(self._data.qpos[ax]), self.robot_hz, float(self._data.qpos[az]))
            body.x, body.y = x, y
        body.heading = float(self._data.qpos[ay])
        body.z = self.robot_hz
        jx = self._mujoco.mj_name2id(self._mj, self._mujoco.mjtObj.mjOBJ_JOINT, f"{body.id}_sx")
        dof = int(self._mj.jnt_dofadr[jx])
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
        # Safety net if a piece fell through the floor plane; do not sphere-clamp onto CAD rims.
        min_z = self.floor_y - 4.0
        if body.z < min_z:
            body.z = self.floor_y + float(body.radius or 1.4)
            body.vz = 0.0
            adr = self._piece_qpos[slot]
            mx, my, mz = ftc_to_mj(body.x, body.y, body.z)
            self._data.qpos[adr : adr + 3] = [mx, my, mz]
            self._data.qvel[vel : vel + 3] = [body.vx, body.vz, -body.vy]

    def _geom_group(self, geom_id: int) -> int:
        if geom_id < 0:
            return -1
        return int(self._mj.geom_group[geom_id])

    def _body_name(self, geom_id: int) -> str:
        if geom_id < 0:
            return ""
        bid = int(self._mj.geom_bodyid[geom_id])
        raw = self._mujoco.mj_id2name(self._mj, self._mujoco.mjtObj.mjOBJ_BODY, bid)
        return str(raw or "")

    def _contact_flags(self, robot_ids: set[str], piece_ids: set[str]) -> ContactSet:
        flags = ContactSet()
        live_robots = set(robot_ids)
        ncon = int(self._data.ncon)
        for i in range(ncon):
            c = self._data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            grp1, grp2 = self._geom_group(g1), self._geom_group(g2)
            b1, b2 = self._body_name(g1), self._body_name(g2)
            names = {b1, b2}
            hits_live_robot = bool(live_robots & names)
            hits_piece = GEOM_GROUP_PIECE in {grp1, grp2}
            hits_field = GEOM_GROUP_FIELD in {grp1, grp2}
            if hits_live_robot and hits_field:
                flags.wall = True
            if grp1 == GEOM_GROUP_ROBOT and grp2 == GEOM_GROUP_ROBOT and b1 in live_robots and b2 in live_robots and b1 != b2:
                flags.robot = True
            if hits_piece and (hits_field or hits_live_robot):
                flags.piece = True
        return flags

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
        self._mujoco.mj_forward(self._mj, self._data)
        nsub = max(1, int(round(dt / max(float(self._mj.opt.timestep), 1e-4))))
        for _ in range(nsub):
            for mechanism_id, (qpos, dof) in self._field_mechanism_joints.items():
                target = float(state.field_mechanism_targets.get(mechanism_id, 0.0))
                error = target - float(self._data.qpos[qpos])
                velocity = float(self._data.qvel[dof])
                self._data.qfrc_applied[dof] = float(np.clip(900.0 * error - 90.0 * velocity, -700.0, 700.0))
            self._mujoco.mj_step(self._mj, self._data)
        for body in state.robots:
            self._read_robot(body)
        for piece in state.pieces:
            self._read_piece(piece)
        flags = self._contact_flags({b.id for b in state.robots}, live_piece_ids)
        for body in state.robots:
            w, _r = _resolve_chassis(body, state.walls, state.robots, self.hw, self.hd)
            flags.wall = flags.wall or w
        return flags

    def contacts(self, index: int = 0) -> ContactSet:
        return self._contacts[min(index, len(self._contacts) - 1)]

    def step_world(self, state: WorldStep, dt: float) -> ContactSet:
        return self.step_batch([state], dt)[0]

