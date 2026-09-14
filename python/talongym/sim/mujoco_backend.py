"""MuJoCo adapters: planar validation plus mesh-field 3D (Y-up inches)."""

from __future__ import annotations

from pathlib import Path

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


class MeshFieldRequiredError(RuntimeError):
    """Raised when a field requires mesh collision but MuJoCo is not installed."""


def ftc_to_mj(x: float, y: float, z: float) -> tuple[float, float, float]:
    return float(x), float(z), float(-y)


def mj_to_ftc(mx: float, my: float, mz: float) -> tuple[float, float, float]:
    return float(mx), float(-mz), float(my)


class MujocoFieldBackend:
    """3D mesh field: planar chassis joints, free sphere pieces, gravity on.

    Coordinates: FTC inches, MuJoCo Y-up (x=ftc.x, y=height, z=-ftc.y).
    """

    name = "mujoco_field"

    def __init__(
        self,
        xml: str,
        field_half_w: float,
        field_half_d: float,
        robot_hz: float = 5.0,
        xml_path: Path | None = None,
    ) -> None:
        if not available():
            raise MeshFieldRequiredError("MuJoCo extra missing; pip install -e '.[mujoco]'")
        import mujoco

        self.hw = field_half_w
        self.hd = field_half_d
        self.robot_hz = robot_hz
        self._mujoco = mujoco
        if xml_path is not None:
            self._mj = mujoco.MjModel.from_xml_path(str(xml_path))
        else:
            self._mj = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._mj)
        self._contacts = [ContactSet()]
        self._robot_jnt: dict[str, tuple[int, int, int]] = {}
        self._piece_body: list[int] = []
        self._piece_qpos: list[int] = []
        self._piece_qvel: list[int] = []
        self._owned: set[str] = set()
        self._id_to_slot: dict[str, int] = {}
        self._robot_placed: set[str] = set()
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
        i = 0
        while True:
            name = f"piece_{i:02d}"
            try:
                bid = mujoco.mj_name2id(self._mj, mujoco.mjtObj.mjOBJ_BODY, name)
            except Exception:
                break
            if bid < 0:
                break
            jnt = int(self._mj.body_jntadr[bid])
            self._piece_body.append(int(bid))
            self._piece_qpos.append(int(self._mj.jnt_qposadr[jnt]))
            self._piece_qvel.append(int(self._mj.jnt_dofadr[jnt]))
            i += 1
        self._park_all_pieces()

    def _park_all_pieces(self) -> None:
        for i, adr in enumerate(self._piece_qpos):
            self._data.qpos[adr : adr + 3] = [180.0 + i * 3.0, 80.0, 0.0]
            vel = self._piece_qvel[i]
            self._data.qvel[vel : vel + 3] = 0.0
        self._owned.clear()
        self._id_to_slot.clear()

    def reset_batch(self, n: int) -> None:
        self._contacts = [ContactSet() for _ in range(max(1, n))]
        self._mujoco.mj_resetData(self._mj, self._data)
        self._park_all_pieces()
        self._robot_placed.clear()

    def _slot_for(self, pid: str) -> int | None:
        if pid in self._id_to_slot:
            return self._id_to_slot[pid]
        used = set(self._id_to_slot.values())
        for i in range(len(self._piece_qpos)):
            if i not in used:
                self._id_to_slot[pid] = i
                return i
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
        slot = self._slot_for(body.id)
        if slot is None:
            return
        adr = self._piece_qpos[slot]
        vel = self._piece_qvel[slot]
        z = float(getattr(body, "z", body.radius or 1.4) or 1.4)
        mx, my, mz = ftc_to_mj(body.x, body.y, z)
        kick = bool(getattr(body, "kick", False))
        if body.id not in self._owned or kick:
            self._data.qpos[adr : adr + 3] = [mx, my, mz]
            vz = float(getattr(body, "vz", 0.0) or 0.0)
            self._data.qvel[vel : vel + 3] = [body.vx, vz, -body.vy]
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
        body.vx = float(self._data.qvel[vel])
        body.vz = float(self._data.qvel[vel + 1])
        body.vy = -float(self._data.qvel[vel + 2])
        floor_z = float(body.radius or 1.4)
        if body.z < floor_z:
            body.z = floor_z
            if body.vz < 0:
                body.vz *= -0.2
            body.vx *= 0.8
            body.vy *= 0.8
            adr = self._piece_qpos[slot]
            mx, my, mz = ftc_to_mj(body.x, body.y, body.z)
            self._data.qpos[adr : adr + 3] = [mx, my, mz]
            self._data.qvel[vel : vel + 3] = [body.vx, body.vz, -body.vy]

    def _contact_flags(self, robot_ids: set[str], piece_ids: set[str]) -> ContactSet:
        flags = ContactSet()
        ncon = int(self._data.ncon)
        for i in range(ncon):
            c = self._data.contact[i]
            n1 = self._geom_name(int(c.geom1))
            n2 = self._geom_name(int(c.geom2))
            names = {n1, n2}
            hits_robot = any(n.startswith(rid) for rid in robot_ids for n in names)
            hits_piece = any(n.startswith("piece_") for n in names)
            hits_wall = any(n.startswith("wall_") or "frame" in n for n in names)
            if hits_robot and hits_wall:
                flags.wall = True
            if hits_robot and any(other != rid and n.startswith(other) for rid in robot_ids for other in robot_ids for n in names):
                flags.robot = True
            if hits_piece and hits_wall:
                flags.piece = True
            if hits_robot and hits_piece:
                flags.piece = True
        return flags

    def _geom_name(self, geom_id: int) -> str:
        try:
            return str(self._mujoco.mj_id2name(self._mj, self._mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
        except Exception:
            return ""

    def step_batch(self, states: list[WorldStep], dt: float) -> list[ContactSet]:
        out: list[ContactSet] = []
        for state in states:
            out.append(self._step_one(state, dt))
        self._contacts = out or [ContactSet()]
        return self._contacts

    def _step_one(self, state: WorldStep, dt: float) -> ContactSet:
        live_piece_ids = {p.id for p in state.pieces}
        for pid in list(self._owned):
            if pid not in live_piece_ids:
                slot = self._id_to_slot.pop(pid, None)
                self._owned.discard(pid)
                if slot is not None:
                    adr = self._piece_qpos[slot]
                    self._data.qpos[adr : adr + 3] = [180.0 + slot * 3.0, 80.0, 0.0]
                    vel = self._piece_qvel[slot]
                    self._data.qvel[vel : vel + 3] = 0.0
        for body in state.robots:
            _clip_chassis(body, state.max_vel, state.max_omega)
            self._write_robot(body)
        for piece in state.pieces:
            self._write_piece(piece)
        self._mujoco.mj_forward(self._mj, self._data)
        nsub = max(1, int(round(dt / max(float(self._mj.opt.timestep), 1e-4))))
        for _ in range(nsub):
            self._mujoco.mj_step(self._mj, self._data)
        for body in state.robots:
            self._read_robot(body)
        for piece in state.pieces:
            self._read_piece(piece)
        flags = self._contact_flags({b.id for b in state.robots}, live_piece_ids)
        # Keep chassis inside the perimeter even if a mesh gap exists.
        for body in state.robots:
            w, _r = _resolve_chassis(body, state.walls, state.robots, self.hw, self.hd)
            flags.wall = flags.wall or w
        return flags

    def contacts(self, index: int = 0) -> ContactSet:
        return self._contacts[min(index, len(self._contacts) - 1)]

    def step_world(self, state: WorldStep, dt: float) -> ContactSet:
        return self.step_batch([state], dt)[0]

