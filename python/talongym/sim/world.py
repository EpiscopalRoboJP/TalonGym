from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from talongym.presets.loader import LoadedPresets
from talongym.robot.drivetrain import clip_twist
from talongym.robot.dynamics import MechanismDynamics
from talongym.robot.mechanisms import chassis_moving, launcher_aim, muzzle_velocity, piece_in_intake, pose_world
from talongym.robot.sensors import camera_world_pose, detect_tags
from talongym.rules.engine import MECHANISM_VERBS, RuleContext, RuleEngine, TickEvent
from talongym.sim.mujoco_backend import ftc_yaw_to_mj_quat
from talongym.sim.geometry import (
    AABB,
    deg_to_rad,
    point_in_shape,
    point_in_volume,
    polygons_overlap,
    rad_to_deg,
    shape_from_element,
    wrap_angle,
)
from talongym.sim.physics import Body, WorldStep, default_backend, perimeter_walls


@dataclass
class Piece:
    id: str
    type_id: str
    x: float
    y: float
    radius: float
    attrs: dict[str, Any]
    held_by: str | None = None
    in_flight: bool = False
    scored: bool = False
    vx: float = 0.0
    vy: float = 0.0
    restitution: float = 0.3
    mass: float = 0.1
    z: float = 1.4
    vz: float = 0.0
    ballistic: bool = False
    kick: bool = False
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0
    visual_asset: str | None = None
    collision_asset: str | None = None


@dataclass
class RobotState:
    body: Body
    held: list[str]
    intake_timer: float = 0.0
    score_timer: float = 0.0
    last_verb: str = "idle"
    spinup_timer: float = 0.0
    collision_time_s: float = 0.0
    first_contact_s: float | None = None
    entered_restricted: bool = False
    vision_hits: list[dict[str, Any]] = field(default_factory=list)
    mechanism: MechanismDynamics | None = None
    drive_effort_scale: float = 1.0


class World:
    def __init__(self, bundle: LoadedPresets, seed: int = 0, control_hz: int = 25, substeps: int = 2, allow_missing_mesh: bool = False) -> None:
        self.bundle = bundle
        self.field = bundle.field
        self.robot = bundle.robot
        self.scoring = bundle.scoring
        self.rng = np.random.default_rng(seed)
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz / substeps
        self.substeps = substeps
        self.engine = RuleEngine(self.scoring)
        fw = float(self.field["fieldSizeIn"]["width"])
        fd = float(self.field["fieldSizeIn"]["depth"])
        playable = self.field.get("playableBoundaryIn") or {}
        self.playable_half_w = float(playable.get("halfWidth") or fw / 2.0)
        self.playable_half_d = float(playable.get("halfDepth") or fd / 2.0)
        self.allow_missing_mesh = allow_missing_mesh
        # Mesh seasons never use scripted flight or scoring teleports.
        self.ballistic_launch = True
        self.walls = perimeter_walls(fw / 2.0, fd / 2.0)
        self.elements = list(self.field.get("elements") or [])
        self.element_shapes = {el["id"]: shape_from_element(el) for el in self.elements}
        self.triggers = {
            (el.get("triggerId") or el["id"]): el
            for el in self.elements
            if el.get("isTrigger")
        }
        self.obstacles: list[AABB] = []
        for el in self.elements:
            if not el.get("isCollider"):
                continue
            if "perimeter" in (el.get("tags") or []):
                continue
            sh = self.element_shapes.get(el["id"])
            if isinstance(sh, AABB) and sh.hx < 70 and sh.hy < 70:
                self.obstacles.append(sh)
        self.occluders: list[AABB] = []
        for el in self.elements:
            if el.get("isOccluder"):
                sh = self.element_shapes.get(el["id"])
                if isinstance(sh, AABB):
                    self.occluders.append(sh)
        self.piece_types = {p["typeId"]: p for p in self.field.get("gamePieces") or []}
        self._cad_source_sha256 = str((self.field.get("provenance") or {}).get("contentSha256") or "") or None
        self._cad_asset_version = self._cad_source_sha256
        self.tags = list(self.field.get("aprilTags") or [])
        self.constraints = self.robot.get("constraints") or {}
        self.max_vel = float(self.constraints.get("maxVelInPerS", 30))
        self.max_accel = float(self.constraints.get("maxAccelInPerS2", 30))
        self.max_ang_vel = math.radians(float(self.constraints.get("maxAngVelDegPerS", 60)))
        self.max_ang_accel = math.radians(float(self.constraints.get("maxAngAccelDegPerS2", 60)))
        chassis = self.robot.get("chassis") or {}
        self.robot_hx = float(chassis.get("widthIn", 18)) / 2.0
        self.robot_hy = float(chassis.get("lengthIn", 18)) / 2.0
        self.robot_hz = float(chassis.get("heightIn") or 10) / 2.0
        self._robot_footprint: tuple[tuple[float, float], ...] | None = None
        self._robot_kind = str(chassis.get("collisionShape") or "aabb")
        if self._robot_kind == "mesh":
            raw = chassis.get("footprint") or []
            pts = []
            for p in raw:
                if isinstance(p, dict) and "x" in p and "y" in p:
                    pts.append((float(p["x"]), float(p["y"])))
            if len(pts) >= 3:
                self._robot_footprint = tuple(pts)
            else:
                self._robot_kind = "aabb"
        self.backend = self._make_backend(fw / 2.0, fd / 2.0)
        self.floor_y = float(getattr(self.backend, "floor_y", 0.0) or 0.0)
        cad_stats = dict(getattr(self.backend, "cad_stats", None) or {})
        self.field_mechanism_tip_angles = {
            str(key): float(value)
            for key, value in (cad_stats.get("fieldMechanismTargets") or {}).items()
        }
        self.field_mechanisms = {key: 0.0 for key in self.field_mechanism_tip_angles}
        if getattr(self.backend, "name", "") == "mujoco_field":
            self.obstacles = []
        mech = self.robot.get("mechanisms") or {}
        self.capacity = int(mech.get("capacity", 3))
        self.intake_time = float(mech.get("intakeCycleTimeS", 0.4))
        self.score_time = float(mech.get("scoreCycleTimeS", 0.6))
        self.can_intake_moving = bool(mech.get("canIntakeWhileMoving", True))
        self.can_score_moving = bool(mech.get("canScoreWhileMoving", True))
        self.intakes = list(self.robot.get("intakes") or [])
        self.launchers = list(self.robot.get("launchers") or [])
        odo = self.robot.get("odometry") or {}
        self._base_pos_noise = float(odo.get("positionNoiseStdIn", 0.3))
        self._base_heading_noise = math.radians(float(odo.get("headingNoiseStdDeg", 1.0)))
        self.pos_noise = self._base_pos_noise
        self.heading_noise = self._base_heading_noise
        self.motor_strength = 1.0
        self.cameras = [s for s in self.robot.get("sensors") or [] if s.get("kind") == "apriltag_camera"]
        self.gate_ids = [
            el["id"]
            for el in self.elements
            if el.get("type") == "gate" or "gate" in (el.get("tags") or [])
        ]
        self.seq_accs = [
            a["id"] for a in (self.scoring.get("accumulators") or []) if a.get("valueType") == "sequence"
        ]
        self.queue_caps = {
            a["id"]: int(a.get("maxLen") or 9)
            for a in (self.scoring.get("accumulators") or [])
            if a.get("valueType") == "sequence"
        }
        self.time_s = 0.0
        self.phase = "AUTO"
        self.auto_s = 30.0
        for ph in self.scoring.get("phases") or []:
            if ph.get("id") == "AUTO":
                self.auto_s = float(ph.get("durationS", 30))
        self.robots: dict[str, RobotState] = {}
        self.pieces: dict[str, Piece] = {}
        self.accumulators: dict[str, Any] = {}
        self.match_vars: dict[str, Any] = {}
        self.observed_vars: dict[str, Any] = {}
        self.gate_state: dict[str, str] = {gid: "closed" for gid in self.gate_ids}
        self.queues: dict[str, list[str]] = {sid: [] for sid in self.seq_accs}
        self.fire_counts: dict[str, int] = {}
        self.prev_occupancy: dict[str, set[str]] = {}
        self.explains: list[dict[str, Any]] = []
        self.step_explains: list[dict[str, Any]] = []
        self.true_score = 0.0
        self.wall_hit = False
        self.robot_hit = False
        self.piece_hit = False
        self.vision_hits: list[dict[str, Any]] = []
        self.last_events: list[TickEvent] = []
        self.pending_piece_ops: list[tuple[str, str | None, str | None]] = []

    def needs_mesh(self) -> bool:
        caps = self.field.get("requiredCapabilities") or []
        return "mesh_field_collision" in caps or bool(self.field.get("collisionAsset"))

    def _make_backend(self, half_w: float, half_d: float):
        if not self.needs_mesh():
            return default_backend(half_w, half_d)
        from talongym.sim.mujoco_backend import MeshFieldRequiredError, MujocoFieldBackend, available

        if not available():
            if self.allow_missing_mesh:
                return default_backend(half_w, half_d)
            raise MeshFieldRequiredError(
                "This field requires mesh_field_collision; pip install -e '.[mujoco]'"
            )
        xml, xml_path, built = self._collision_xml()
        return MujocoFieldBackend(
            xml,
            half_w,
            half_d,
            robot_hz=self.robot_hz,
            xml_path=xml_path,
            slot_plan=built.slot_plan,
            floor_y=built.floor_y,
            cad_stats=built.stats,
        )

    def _robot_mesh_path(self):
        from talongym.assets.import_robot_cad import RobotCadError, resolve_robot_asset

        if self._robot_kind != "mesh":
            return None
        rel = self.robot.get("collisionAsset")
        if not rel:
            return None
        try:
            dest = resolve_robot_asset(str(rel))
        except RobotCadError:
            return None
        return dest if dest.is_file() else None

    def _collision_xml(self):
        from talongym.assets.cad_common import CadImportError
        from talongym.assets.mjcf_field import (
            CAD_MJCF_MARKER,
            CAD_MJCF_VERSION,
            FieldMjcf,
            build_field_mjcf,
            load_field_manifest,
            piece_slot_plan,
            select_field_collision_parts,
            verify_collision_asset,
        )
        from talongym.sim.mujoco_backend import MeshFieldRequiredError

        mesh = self._robot_mesh_path()
        try:
            committed = verify_collision_asset(self.field)
            if (
                committed is not None
                and mesh is None
                and self.field.get("cadManifest")
                and not self.robot.get("rigidParts")
            ):
                doc = load_field_manifest(self.field, require=True)
                xml = committed.read_text(encoding="utf-8")
                if CAD_MJCF_MARKER not in xml[:1600]:
                    raise CadImportError(
                        f"collisionAsset {committed} is not CAD-assembled; rebuild with write_collision_mjcf"
                    )
                parts = list((doc.get("field") or {}).get("collisionParts") or [])
                _kept, filter_stats = select_field_collision_parts(parts)
                slot_plan = piece_slot_plan(self.field)
                digest = str((doc.get("field") or {}).get("sha256") or "") or self._cad_source_sha256
                self._cad_source_sha256 = digest
                generator_version = str(doc.get("generatorVersion") or "")
                self._cad_asset_version = f"{digest}:{generator_version}" if generator_version else digest
                built = FieldMjcf(
                    xml=xml,
                    stats={
                        "cad": True,
                        "filter": filter_stats.as_dict(),
                        "slotPlan": dict(slot_plan),
                        "fieldMechanismTargets": {
                            str(mechanism.get("id")): math.radians(float(mechanism.get("tipAngleDeg") or 0.0))
                            for mechanism in ((doc.get("field") or {}).get("mechanisms") or [])
                            if mechanism.get("id")
                        },
                        "loadedCommitted": True,
                    },
                    floor_y=float(filter_stats.floor_y),
                    slot_plan=slot_plan,
                    cad=True,
                )
                return xml, committed, built
            built = build_field_mjcf(
                self.field,
                robot=self.robot,
                robot_hx=self.robot_hx,
                robot_hy=self.robot_hy,
                robot_hz=self.robot_hz,
                robot_mesh=mesh,
            )
            if built.cad:
                digest = str((self.field.get("provenance") or {}).get("contentSha256") or "") or self._cad_source_sha256
                self._cad_source_sha256 = digest
                generator_version = str(CAD_MJCF_VERSION)
                self._cad_asset_version = f"{digest}:{generator_version}" if digest else generator_version
                built.stats["loadedCommitted"] = True
                built.stats["articulatedRobot"] = bool(self.robot.get("rigidParts"))
        except CadImportError as exc:
            raise MeshFieldRequiredError(str(exc)) from exc
        if committed is not None and built.cad:
            head = committed.read_text(encoding="utf-8")[:1200]
            if CAD_MJCF_MARKER not in head:
                raise MeshFieldRequiredError(
                    f"collisionAsset {committed} is not CAD-assembled; rebuild with write_collision_mjcf"
                )
        xml_path = None
        if mesh is not None:
            dest = mesh.parent / "mjcf_robots.xml"
            dest.write_text(built.xml, encoding="utf-8")
            xml_path = dest
        return built.xml, xml_path, built

    def _domain_randomization(self) -> dict[str, Any]:
        return ((self.bundle.training or {}).get("domainRandomization") or {})

    def spawn_jitter_sigma(self, kind: str = "robot") -> tuple[float, float]:
        """Return (xy inches, heading radians) Gaussian sigmas from the training preset."""
        dr = self._domain_randomization()
        if kind == "field_piece":
            xy = float(dr.get("pieceSpawnJitterIn", 0.0))
            heading = math.radians(float(dr.get("pieceHeadingJitterDeg", 0.0)))
        else:
            # Field-build tolerance moves CAD, not the robot off the wall (G304.C / LEAVE).
            xy = float(dr.get("robotStartJitterIn", dr.get("poseJitterIn", 0.2)))
            heading = math.radians(float(dr.get("robotHeadingJitterDeg", dr.get("headingJitterDeg", 0.0))))
        return xy, heading

    def apply_episode_randomization(self, *, full_noise: bool = True) -> None:
        dr = self._domain_randomization()
        scale = float(dr.get("sensorNoiseScale") or 1.0)
        if not full_noise:
            scale *= 0.15
        self.pos_noise = self._base_pos_noise * scale
        self.heading_noise = self._base_heading_noise * scale
        lo_hi = list(dr.get("motorStrengthRange") or [1.0, 1.0])
        lo = float(lo_hi[0] if lo_hi else 1.0)
        hi = float(lo_hi[1] if len(lo_hi) > 1 else lo)
        if hi < lo:
            lo, hi = hi, lo
        self.motor_strength = float(self.rng.uniform(lo, hi)) if hi > lo else lo

    def reset(
        self,
        seed: int | None = None,
        static_teammate: bool = True,
        opponent_mode: str = "none",
        live_teammate: bool = False,
        full_noise: bool = True,
        ballistic_launch: bool | None = None,
        match_setup: dict[str, Any] | None = None,
    ) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if ballistic_launch is not None and not self.needs_mesh():
            self.ballistic_launch = bool(ballistic_launch)
        elif self.needs_mesh():
            self.ballistic_launch = True
        self._episode_full_noise = bool(full_noise)
        self.apply_episode_randomization(full_noise=full_noise)
        self.time_s = 0.0
        self.phase = "AUTO"
        self.true_score = 0.0
        self.explains = []
        self.step_explains = []
        self.fire_counts = {}
        self.gate_state = {gid: "closed" for gid in self.gate_ids}
        self.queues = {sid: [] for sid in self.seq_accs}
        self.accumulators = self.engine.init_accumulators()
        match_vals, acc_seed = self.engine.init_match_vars(self.rng)
        self.match_vars = match_vals
        self.accumulators.update(acc_seed)
        self.observed_vars = {k: None for k in self.match_vars}
        self.pieces = {}
        self.field_mechanisms = {key: 0.0 for key in self.field_mechanism_tip_angles}
        preload_ids: dict[str, list[str]] = {}
        pid = 0
        for spawn in self.field.get("spawns") or []:
            ptype = spawn["pieceTypeId"]
            spec = self.piece_types[ptype]
            sh = spec.get("shape") or {}
            if sh.get("kind") == "circle":
                rad = float(sh.get("radius") or 2.5)
            else:
                rad = max(float(sh.get("width") or 3.0), float(sh.get("depth") or 3.0)) / 2.0
            color = (spec.get("attributes") or {}).get("color")
            xy_sigma, _ = self.spawn_jitter_sigma("field_piece")
            for pose in spawn.get("poses") or []:
                jx = float(self.rng.normal(0, xy_sigma)) if xy_sigma > 0 else 0.0
                jy = float(self.rng.normal(0, xy_sigma)) if xy_sigma > 0 else 0.0
                name = f"p{pid}"
                pid += 1
                heading0 = deg_to_rad(float(pose.get("headingDeg") or 0.0))
                qw, qx, qy, qz = ftc_yaw_to_mj_quat(heading0)
                spawn_z = float(pose.get("z") or (rad + self.floor_y))
                piece = Piece(
                    id=name,
                    type_id=ptype,
                    x=float(pose["x"]) + jx,
                    y=float(pose["y"]) + jy,
                    radius=rad,
                    attrs={
                        "color": color,
                        "passed_goal_top": False,
                        "passed_archway": False,
                        "initial_volume_id": spawn.get("initialVolumeId"),
                        "field_mechanism_id": spawn.get("containedByMechanismId"),
                    },
                    restitution=float(spec.get("restitution") or 0.3),
                    mass=float(spec.get("massKg") or 0.1),
                    z=spawn_z,
                    qw=qw,
                    qx=qx,
                    qy=qy,
                    qz=qz,
                    visual_asset=spec.get("visualAsset"),
                    collision_asset=spec.get("collisionAsset"),
                )
                self.pieces[name] = piece
                preload_robot = spawn.get("heldByRobotId")
                if preload_robot:
                    preload_ids.setdefault(str(preload_robot), []).append(name)
        self._piece_serial = pid
        self.robots = {}
        self._spawn_robots(
            static_teammate=static_teammate,
            opponent_mode=opponent_mode,
            live_teammate=live_teammate,
            match_setup=match_setup,
        )
        required_preload = int(self.field.get("requiredPreloadCount") or 0)
        for robot_id, piece_ids in preload_ids.items():
            holder = self.robots.get(robot_id)
            if holder is None:
                for piece_id in piece_ids:
                    self.pieces.pop(piece_id, None)
                continue
            if required_preload and len(piece_ids) != required_preload:
                raise ValueError(
                    f"{robot_id} requires exactly {required_preload} preload pieces; got {len(piece_ids)}"
                )
            if required_preload and self.capacity < required_preload:
                raise ValueError(
                    f"robot capacity {self.capacity} cannot hold required {required_preload}-piece preload"
                )
            storage_slots = list((self.robot.get("piecePath") or {}).get("storageSlots") or [])
            for index, piece_id in enumerate(piece_ids):
                piece = self.pieces[piece_id]
                piece.held_by = robot_id
                slot = storage_slots[index] if index < len(storage_slots) else {}
                local_x = float(slot.get("x") or 0.0)
                local_y = float(slot.get("y") or 0.0)
                cos_h, sin_h = math.cos(holder.body.heading), math.sin(holder.body.heading)
                piece.x = holder.body.x + cos_h * local_x - sin_h * local_y
                piece.y = holder.body.y + sin_h * local_x + cos_h * local_y
                piece.z = float(slot.get("z") or holder.body.z)
                piece.vx = piece.vy = piece.vz = 0.0
                holder.held.append(piece_id)
        self.prev_occupancy = self._occupancy()
        self.vision_hits = []
        self.pending_piece_ops = []
        self.backend.reset_batch(1)
        self._sense()

    def spawn_nectar_in_garden(self, alliance: str, x: float, y: float) -> str:
        """Create one alliance NECTAR at a chosen GARDEN center point."""
        if alliance not in {"red", "blue"}:
            raise ValueError("alliance must be 'red' or 'blue'")
        garden_id = f"{alliance}_garden"
        garden = next((el for el in self.elements if el.get("id") == garden_id), None)
        if garden is None:
            raise ValueError(f"field has no {garden_id}")
        shape = self.element_shapes[garden_id]
        if not point_in_volume(garden, shape, float(x), float(y), self.floor_y, 0.0):
            raise ValueError(f"NECTAR spawn ({x:.3f}, {y:.3f}) is outside {garden_id}")
        type_id = f"nectar_{alliance}"
        spec = self.piece_types.get(type_id)
        if spec is None:
            raise ValueError(f"field has no {type_id} piece type")
        radius = float((spec.get("shape") or {}).get("radius") or 1.8)
        piece_id = f"p{self._piece_serial}"
        self._piece_serial += 1
        self.pieces[piece_id] = Piece(
            id=piece_id,
            type_id=type_id,
            x=float(x),
            y=float(y),
            radius=radius,
            attrs={
                "color": (spec.get("attributes") or {}).get("color"),
                "passed_goal_top": False,
                "passed_archway": False,
                "spawned_in_garden": garden_id,
            },
            restitution=float(spec.get("restitution") or 0.3),
            mass=float(spec.get("massKg") or 0.1),
            z=self.floor_y + radius,
            visual_asset=spec.get("visualAsset"),
            collision_asset=spec.get("collisionAsset"),
        )
        return piece_id

    def _start_slot(self, alliance: str, slot: int) -> dict[str, Any] | None:
        found = [
            s
            for s in (self.field.get("startSlots") or [])
            if s.get("alliance") == alliance and int(s.get("slot", 0)) == slot
        ]
        if found:
            return found[0]
        alts = [s for s in (self.field.get("startSlots") or []) if s.get("alliance") == alliance]
        alts = sorted(alts, key=lambda s: int(s.get("slot", 0)))
        if slot < len(alts):
            return alts[slot]
        return None

    def _start_slot_by_id(self, slot_id: str, alliance: str) -> dict[str, Any]:
        slot = next(
            (s for s in (self.field.get("startSlots") or []) if str(s.get("id")) == slot_id),
            None,
        )
        if slot is None:
            raise ValueError(f"unknown start slot {slot_id!r}")
        if slot.get("alliance") != alliance:
            raise ValueError(f"start slot {slot_id!r} is not valid for {alliance}")
        return slot

    def _require_start_slot(self, alliance: str, slot: int) -> dict[str, Any]:
        found = self._start_slot(alliance, slot)
        if found is None:
            raise ValueError(f"field preset is missing a {alliance} slot {slot} start")
        return found

    def _robot_planar_extents(self, heading: float) -> tuple[float, float]:
        c, s = abs(math.cos(heading)), abs(math.sin(heading))
        return c * self.robot_hx + s * self.robot_hy, s * self.robot_hx + c * self.robot_hy

    def _robot_aabb(self, x: float, y: float, heading: float) -> AABB:
        ex, ey = self._robot_planar_extents(heading)
        return AABB(x, y, ex, ey)

    def _offset_limits(self, slot: dict[str, Any]) -> tuple[float, float, float]:
        legal = slot.get("legalRegion") or {}
        kind = legal.get("kind")
        if kind == "aabb":
            return float(legal.get("width") or 0.0) / 2.0, float(legal.get("depth") or 0.0) / 2.0, 0.0
        if kind == "circle":
            radius = float(legal.get("radius") or 0.0)
            return radius, radius, radius
        return 0.0, 0.0, 0.0

    def _validate_start_offset(self, slot: dict[str, Any], ox: float, oy: float) -> None:
        legal = slot.get("legalRegion") or {}
        kind = legal.get("kind")
        slot_id = slot.get("id")
        if kind == "aabb":
            max_ox, max_oy, _ = self._offset_limits(slot)
            if abs(ox) > max_ox + 1e-6:
                raise ValueError(f"x offset {ox} is outside start slot {slot_id}")
            if abs(oy) > max_oy + 1e-6:
                raise ValueError(f"y offset {oy} is outside start slot {slot_id}")
            return
        if kind == "circle":
            radius = float(legal.get("radius") or 0.0)
            if math.hypot(ox, oy) > radius + 1e-6:
                raise ValueError(f"offset ({ox}, {oy}) is outside start slot {slot_id}")
            return
        if abs(ox) > 1e-6 or abs(oy) > 1e-6:
            raise ValueError(f"start slot {slot_id} has no legalRegion; offsets must be 0")

    def _slot_wall(self, slot: dict[str, Any]) -> str:
        pose = slot.get("pose") or {}
        x, y = float(pose.get("x") or 0.0), float(pose.get("y") or 0.0)
        walls = {
            "neg_x": abs(x + self.playable_half_w),
            "pos_x": abs(x - self.playable_half_w),
            "neg_y": abs(y + self.playable_half_d),
            "pos_y": abs(y - self.playable_half_d),
        }
        return min(walls, key=walls.get)

    def _snap_touching_wall(self, wall: str, x: float, y: float, heading: float) -> tuple[float, float]:
        ex, ey = self._robot_planar_extents(heading)
        if wall == "neg_x":
            return -self.playable_half_w + ex, y
        if wall == "pos_x":
            return self.playable_half_w - ex, y
        if wall == "neg_y":
            return x, -self.playable_half_d + ey
        return x, self.playable_half_d - ey

    def _start_pose_error(self, x: float, y: float, heading: float, alliance: str) -> str | None:
        box = self._robot_aabb(x, y, heading)
        if (
            abs(x) + box.hx > self.playable_half_w + 1e-6
            or abs(y) + box.hy > self.playable_half_d + 1e-6
        ):
            return f"robot start ({x:.3f}, {y:.3f}) intersects the STEP perimeter"
        if alliance == "red" and box.maxx > 1e-6:
            return f"robot start ({x:.3f}, {y:.3f}) must stay on the red alliance half (G304.A)"
        if alliance == "blue" and box.minx < -1e-6:
            return f"robot start ({x:.3f}, {y:.3f}) must stay on the blue alliance half (G304.A)"
        if not self._touching_perimeter(x, y, heading):
            return f"robot start ({x:.3f}, {y:.3f}) must touch the FIELD perimeter wall (G304.C)"
        for el in self.elements:
            tags = set(el.get("tags") or [])
            shape = self.element_shapes.get(el["id"])
            if not isinstance(shape, AABB):
                continue
            if not box.overlaps_aabb(shape):
                continue
            if tags.intersection({"loading_zone", "park"}):
                return f"robot start ({x:.3f}, {y:.3f}) must not start in the LOADING ZONE (G304.E)"
            if "flower" in tags:
                return f"robot start ({x:.3f}, {y:.3f}) must not contact a FLOWER (G304.D)"
        return None

    def _touching_perimeter(self, x: float, y: float, heading: float) -> bool:
        box = self._robot_aabb(x, y, heading)
        return (
            box.minx <= -self.playable_half_w + 0.35
            or box.maxx >= self.playable_half_w - 0.35
            or box.miny <= -self.playable_half_d + 0.35
            or box.maxy >= self.playable_half_d - 0.35
        )

    def _validate_start_pose(self, x: float, y: float, heading: float, alliance: str = "red") -> None:
        error = self._start_pose_error(x, y, heading, alliance)
        if error:
            raise ValueError(error)

    def _make_robot(
        self,
        rid: str,
        alliance: str,
        slot: dict[str, Any],
        dynamic: bool,
        offset: dict[str, Any] | None = None,
    ) -> RobotState:
        pose = slot["pose"]
        offset = offset or {}
        ox = float(offset.get("x") or 0.0)
        oy = float(offset.get("y") or 0.0)
        oh = float(offset.get("headingDeg") or 0.0)
        self._validate_start_offset(slot, ox, oy)
        wall = self._slot_wall(slot)
        base_heading = wrap_angle(deg_to_rad(float(pose["headingDeg"]) + oh))
        base_x, base_y = self._snap_touching_wall(wall, float(pose["x"]) + ox, float(pose["y"]) + oy, base_heading)
        self._validate_start_pose(base_x, base_y, base_heading, alliance)
        xy_sigma, heading_sigma = self.spawn_jitter_sigma()
        if not getattr(self, "_episode_full_noise", True):
            xy_sigma = 0.0
            heading_sigma = 0.0
        max_ox, max_oy, max_r = self._offset_limits(slot)
        x, y, heading = base_x, base_y, base_heading
        for _ in range(16):
            jx = float(self.rng.normal(0, xy_sigma)) if xy_sigma > 0 else 0.0
            jy = float(self.rng.normal(0, xy_sigma)) if xy_sigma > 0 else 0.0
            jh = float(self.rng.normal(0, heading_sigma)) if heading_sigma > 0 else 0.0
            if max_r > 0:
                total_x, total_y = ox + jx, oy + jy
                scale = math.hypot(total_x, total_y) / max_r if max_r else 0.0
                if scale > 1.0:
                    total_x /= scale
                    total_y /= scale
                cand_x = float(pose["x"]) + total_x
                cand_y = float(pose["y"]) + total_y
            else:
                total_x = float(np.clip(ox + jx, -max_ox, max_ox))
                total_y = float(np.clip(oy + jy, -max_oy, max_oy))
                cand_x = float(pose["x"]) + total_x
                cand_y = float(pose["y"]) + total_y
            cand_heading = wrap_angle(base_heading + jh)
            cand_x, cand_y = self._snap_touching_wall(wall, cand_x, cand_y, cand_heading)
            if self._start_pose_error(cand_x, cand_y, cand_heading, alliance) is None:
                x, y, heading = cand_x, cand_y, cand_heading
                break
        else:
            x, y, heading = base_x, base_y, base_heading
        self._validate_start_pose(x, y, heading, alliance)
        mechanism = None
        if self.robot.get("actuators") and self.robot.get("powerSystem"):
            physical_robot = copy.deepcopy(self.robot)
            piece_path = physical_robot.get("piecePath") or {}
            variation = float(piece_path.get("coefficientVariation") or 0.0)
            if not getattr(self, "_episode_full_noise", True):
                variation *= 0.15
            scale = float(
                np.clip(self.rng.normal(1.0, variation), 1.0 - variation, 1.0 + variation)
            ) if variation > 0 else 1.0
            for key in ("launchEfficiency", "dragCoefficient", "magnusCoefficient"):
                if piece_path.get(key) is not None:
                    piece_path[key] = float(piece_path[key]) * scale
            piece_path["episodeCoefficientScale"] = scale
            mechanism = MechanismDynamics(physical_robot)
        return RobotState(
            body=Body(
                rid,
                x,
                y,
                heading,
                hx=self.robot_hx,
                hy=self.robot_hy,
                z=self.robot_hz,
                dynamic=dynamic,
                alliance=alliance,
                kind=self._robot_kind,
                footprint=self._robot_footprint,
                mass=float((self.robot.get("chassis") or {}).get("massKg") or 15.0),
            ),
            held=[],
            mechanism=mechanism,
        )

    def _spawn_robots(
        self,
        static_teammate: bool,
        opponent_mode: str,
        live_teammate: bool,
        match_setup: dict[str, Any] | None = None,
    ) -> None:
        if not (self.field.get("startSlots") or []):
            raise ValueError("field preset is missing startSlots")
        configured = {
            str(row.get("id")): row
            for row in ((match_setup or {}).get("robots") or [])
            if isinstance(row, dict) and row.get("id")
        }
        if configured:
            if configured.get("red_0", {}).get("enabled", True) is False:
                raise ValueError("red_0 is the learner and cannot be disabled")
            for rid, row in configured.items():
                if not bool(row.get("enabled", True)):
                    continue
                alliance = rid.split("_", 1)[0]
                if alliance not in {"red", "blue"}:
                    raise ValueError(f"invalid robot id {rid!r}")
                try:
                    default_index = int(rid.rsplit("_", 1)[1])
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"invalid robot id {rid!r}") from exc
                slot_id = row.get("startSlotId")
                slot = (
                    self._start_slot_by_id(str(slot_id), alliance)
                    if slot_id
                    else self._require_start_slot(alliance, default_index)
                )
                default_dynamic = rid == "red_0" or (
                    alliance == "red" and live_teammate
                ) or (alliance == "blue" and opponent_mode not in {"none", "static"})
                dynamic_value = row.get("dynamic")
                dynamic = default_dynamic if dynamic_value is None else bool(dynamic_value)
                self.robots[rid] = self._make_robot(
                    rid,
                    alliance,
                    slot,
                    dynamic,
                    row.get("offset"),
                )
            self._validate_robot_start_separation()
            return
        red0 = self._require_start_slot("red", 0)
        self.robots["red_0"] = self._make_robot("red_0", "red", red0, True)
        red1 = self._start_slot("red", 1)
        if red1 is not None and (live_teammate or static_teammate):
            self.robots["red_1"] = self._make_robot("red_1", "red", red1, bool(live_teammate))
        mode = opponent_mode or "none"
        if mode != "none":
            dynamic_opp = mode != "static"
            for i, key in enumerate(("blue_0", "blue_1")):
                sl = self._start_slot("blue", i)
                if sl is None:
                    continue
                self.robots[key] = self._make_robot(key, "blue", sl, dynamic_opp)
        self._validate_robot_start_separation()

    def _validate_robot_start_separation(self) -> None:
        rows = list(self.robots.values())
        for index, left in enumerate(rows):
            lx, ly = self._robot_planar_extents(left.body.heading)
            for right in rows[index + 1 :]:
                rx, ry = self._robot_planar_extents(right.body.heading)
                if (
                    abs(left.body.x - right.body.x) < lx + rx - 1e-6
                    and abs(left.body.y - right.body.y) < ly + ry - 1e-6
                ):
                    raise ValueError(
                        f"robot starts {left.body.id} and {right.body.id} overlap"
                    )

    def actor(self) -> RobotState:
        return self.robots["red_0"]

    def _occupancy(self) -> dict[str, set[str]]:
        occ: dict[str, set[str]] = {tid: set() for tid in self.triggers}
        for tid, el in self.triggers.items():
            sh = self.element_shapes[el["id"]]
            for rid, rs in self.robots.items():
                if tid == "leave_interior" and self._touching_perimeter(rs.body.x, rs.body.y, rs.body.heading):
                    continue
                if point_in_volume(el, sh, rs.body.x, rs.body.y, getattr(rs.body, "z", self.robot_hz), 0.0):
                    occ[tid].add(rid)
            for p in self.pieces.values():
                if p.held_by or p.scored:
                    continue
                if point_in_volume(el, sh, p.x, p.y, p.z, p.radius):
                    occ[tid].add(p.id)
        return occ

    def _events_from_occupancy(self, occ: dict[str, set[str]]) -> list[TickEvent]:
        events: list[TickEvent] = []
        for tid, now in occ.items():
            prev = self.prev_occupancy.get(tid, set())
            for ident in now - prev:
                if ident in self.robots:
                    events.append(TickEvent("volumeEnter", volume_id=tid, robot_id=ident))
                    events.append(TickEvent("contactStart", volume_id=tid, robot_id=ident))
                elif ident in self.pieces:
                    p = self.pieces[ident]
                    events.append(
                        TickEvent(
                            "volumeEnter",
                            volume_id=tid,
                            piece_id=p.id,
                            piece_type=p.type_id,
                            piece_attrs=dict(p.attrs),
                        )
                    )
            for ident in prev - now:
                if ident in self.robots:
                    events.append(TickEvent("volumeExit", volume_id=tid, robot_id=ident))
                elif ident in self.pieces:
                    p = self.pieces[ident]
                    events.append(
                        TickEvent(
                            "volumeExit",
                            volume_id=tid,
                            piece_id=p.id,
                            piece_type=p.type_id,
                            piece_attrs=dict(p.attrs),
                        )
                    )
        return events

    def _apply_follower(self, rs: RobotState, target: np.ndarray, speed_frac: float, dt: float) -> None:
        if not rs.body.dynamic:
            rs.body.vx = rs.body.vy = rs.body.omega = 0.0
            return
        tx, ty, th = float(target[0]), float(target[1]), float(target[2])
        ex, ey = tx - rs.body.x, ty - rs.body.y
        dist = math.hypot(ex, ey)
        effort = float(rs.drive_effort_scale)
        speed = self.max_vel * float(self.motor_strength) * effort * float(np.clip(speed_frac, 0.2, 1.0))
        if dist > 1e-3:
            des_vx = speed * ex / dist
            des_vy = speed * ey / dist
        else:
            des_vx = des_vy = 0.0
        heading_err = wrap_angle(th - rs.body.heading)
        des_w = float(np.clip(2.5 * heading_err, -self.max_ang_vel, self.max_ang_vel))
        dvx = float(np.clip(des_vx - rs.body.vx, -self.max_accel * effort * dt, self.max_accel * effort * dt))
        dvy = float(np.clip(des_vy - rs.body.vy, -self.max_accel * effort * dt, self.max_accel * effort * dt))
        dw = float(np.clip(des_w - rs.body.omega, -self.max_ang_accel * effort * dt, self.max_ang_accel * effort * dt))
        rs.body.vx += dvx
        rs.body.vy += dvy
        rs.body.omega += dw
        vx, vy, om = clip_twist(rs.body.vx, rs.body.vy, rs.body.omega, self.robot)
        rs.body.vx, rs.body.vy, rs.body.omega = vx, vy, om

    def _apply_velocity(self, rs: RobotState, vx: float, vy: float, omega: float, dt: float) -> None:
        if not rs.body.dynamic:
            rs.body.vx = rs.body.vy = rs.body.omega = 0.0
            return
        s = float(self.motor_strength)
        effort = float(rs.drive_effort_scale)
        des_vx, des_vy, des_w = clip_twist(vx * s, vy * s, omega * s, self.robot)
        dvx = float(np.clip(des_vx - rs.body.vx, -self.max_accel * effort * dt, self.max_accel * effort * dt))
        dvy = float(np.clip(des_vy - rs.body.vy, -self.max_accel * effort * dt, self.max_accel * effort * dt))
        dw = float(np.clip(des_w - rs.body.omega, -self.max_ang_accel * effort * dt, self.max_ang_accel * effort * dt))
        rs.body.vx += dvx
        rs.body.vy += dvy
        rs.body.omega += dw
        rs.body.vx, rs.body.vy, rs.body.omega = clip_twist(rs.body.vx, rs.body.vy, rs.body.omega, self.robot)

    def _apply_command(self, rs: RobotState, action: dict[str, Any], dt: float) -> None:
        vel = action.get("velocity")
        if vel is not None:
            arr = np.asarray(vel, dtype=np.float64).reshape(-1)
            self._apply_velocity(
                rs,
                float(arr[0]),
                float(arr[1]) if arr.size > 1 else 0.0,
                float(arr[2]) if arr.size > 2 else 0.0,
                dt,
            )
            return
        target = np.asarray(
            action.get("target_pose", [rs.body.x, rs.body.y, rs.body.heading]),
            dtype=np.float64,
        ).reshape(3)
        speed = float(action.get("speed_frac", 0.8))
        self._apply_follower(rs, target, speed, dt)

    def _piece_in_reach(self, rs: RobotState, p: Piece) -> tuple[bool, dict[str, Any] | None]:
        if self.intakes:
            for intake in self.intakes:
                if piece_in_intake(p.x, p.y, p.z, p.radius, rs.body.x, rs.body.y, rs.body.heading, intake):
                    return True, intake
            return False, None
        hull = self.robot_hx + p.radius + 4.0
        return math.hypot(p.x - rs.body.x, p.y - rs.body.y) <= hull, None

    def _mechanism_commands(
        self,
        rs: RobotState,
        verb: str,
        overrides: dict[str, Any] | None,
    ) -> dict[str, float]:
        if rs.mechanism is None:
            return {}
        path = self.robot.get("piecePath") or {}
        commands = {ident: 0.0 for ident in rs.mechanism.actuators}
        gate_id = str(path.get("gateActuatorId") or "")
        hood_id = str(path.get("hoodActuatorId") or "")
        if gate_id in commands:
            commands[gate_id] = -1.0
        if hood_id in commands:
            launcher = self.launchers[0] if self.launchers else {}
            pitch = float((launcher.get("poseOnRobot") or {}).get("pitchDeg") or 45.0)
            travel = rs.mechanism.actuators[hood_id].config.get("travelLimit") or [0.0, 90.0]
            lo, hi = float(travel[0]), float(travel[1])
            commands[hood_id] = float(np.clip(2.0 * (pitch - lo) / max(hi - lo, 1e-6) - 1.0, -1.0, 1.0))
        if verb == "intake":
            for key in ("intakeActuatorId", "conveyorActuatorId"):
                ident = str(path.get(key) or "")
                if ident in commands:
                    commands[ident] = 1.0
        elif verb == "score":
            flywheel_id = str(path.get("flywheelActuatorId") or "")
            if flywheel_id in commands:
                commands[flywheel_id] = 1.0
                actuator = rs.mechanism.actuators[flywheel_id]
                target_rpm = float(actuator.config.get("targetRpm") or 0.0)
                actual_rpm = abs(actuator.state.velocity_rad_s) * 60.0 / (2.0 * math.pi)
                if actual_rpm >= 0.9 * target_rpm:
                    conveyor_id = str(path.get("conveyorActuatorId") or "")
                    if conveyor_id in commands:
                        commands[conveyor_id] = 1.0
                    if gate_id in commands:
                        commands[gate_id] = 1.0
        for ident, value in (overrides or {}).items():
            if ident in commands:
                commands[ident] = float(np.clip(float(value), -1.0, 1.0))
        return commands

    def _step_robot_dynamics(
        self,
        rs: RobotState,
        action: dict[str, Any],
        verb: str,
        dt: float,
    ) -> None:
        if rs.mechanism is None:
            rs.drive_effort_scale = 1.0
            return
        speed_fraction = 0.0
        if action.get("velocity") is not None:
            velocity = np.asarray(action["velocity"], dtype=np.float64).reshape(-1)
            linear = math.hypot(float(velocity[0]), float(velocity[1]) if velocity.size > 1 else 0.0)
            speed_fraction = max(speed_fraction, min(1.0, linear / max(self.max_vel, 1e-6)))
        elif action.get("target_pose") is not None:
            target = np.asarray(action["target_pose"], dtype=np.float64).reshape(-1)
            distance = math.hypot(float(target[0]) - rs.body.x, float(target[1]) - rs.body.y)
            heading_error = abs(wrap_angle(float(target[2]) - rs.body.heading))
            if distance > 0.1 or heading_error > math.radians(1.0):
                speed_fraction = abs(float(action.get("speed_frac", 0.8)))
        motors = self.robot.get("motors") or {}
        drivetrain_current = (
            float(motors.get("count") or 4)
            * float(motors.get("currentLimitA") or 20.0)
            * min(1.0, 0.08 + 0.7 * speed_fraction)
        )
        commands = self._mechanism_commands(
            rs,
            verb,
            action.get("actuators") if isinstance(action.get("actuators"), dict) else None,
        )
        rs.mechanism.step(commands, dt, drivetrain_current_a=drivetrain_current)
        nominal = float((self.robot.get("powerSystem") or {}).get("openCircuitVoltageV") or 12.0)
        rs.drive_effort_scale = float(
            np.clip(rs.mechanism.power.voltage_v / max(nominal, 1e-6), 0.0, 1.0)
        )

    def _mechanisms(self, rs: RobotState, verb: str, dt: float, events: list[TickEvent]) -> None:
        moving = chassis_moving(rs.body.vx, rs.body.vy)
        if verb != "score":
            rs.spinup_timer = 0.0
        rs.last_verb = verb
        if rs.mechanism is None and verb == "intake" and len(rs.held) < self.capacity:
            grabbed = False
            cycle = self.intake_time
            for p in self.pieces.values():
                if p.held_by or p.in_flight or p.scored:
                    continue
                hit, intake = self._piece_in_reach(rs, p)
                if not hit:
                    continue
                can_move = self.can_intake_moving
                if intake is not None:
                    can_move = bool(intake.get("canRunWhileMoving", can_move))
                    cycle = float(intake.get("cycleTimeS") if intake.get("cycleTimeS") is not None else cycle)
                if moving and not can_move:
                    continue
                rs.intake_timer += dt
                if rs.intake_timer >= cycle:
                    p.held_by = rs.body.id
                    p.vx = p.vy = p.vz = 0.0
                    p.wx = p.wy = p.wz = 0.0
                    p.x, p.y = rs.body.x, rs.body.y
                    rs.held.append(p.id)
                    rs.intake_timer = 0.0
                grabbed = True
                break
            if not grabbed:
                rs.intake_timer = 0.0
        else:
            rs.intake_timer = 0.0
        launcher = self.launchers[0] if self.launchers else None
        if verb == "score":
            if launcher is None:
                rs.spinup_timer = 0.0
                rs.score_timer = max(0.0, rs.score_timer - dt)
                return
            can_launch_moving = self.can_score_moving
            spinup = 0.0
            cycle = self.score_time
            if launcher is not None:
                can_launch_moving = bool(launcher.get("canLaunchWhileMoving", can_launch_moving))
                spinup = float(launcher.get("spinupTimeS") or 0.0)
                if launcher.get("cycleTimeS") is not None:
                    cycle = float(launcher["cycleTimeS"])
            blocked = moving and not can_launch_moving
            if rs.mechanism is not None:
                path = self.robot.get("piecePath") or {}
                flywheel_id = str(path.get("flywheelActuatorId") or "")
                flywheel = rs.mechanism.actuators.get(flywheel_id)
                if flywheel is None:
                    blocked = True
                else:
                    target_rpm = float(flywheel.config.get("targetRpm") or 0.0)
                    actual_rpm = abs(flywheel.state.velocity_rad_s) * 60.0 / (2.0 * math.pi)
                    blocked = blocked or actual_rpm < 0.9 * target_rpm
                    rs.spinup_timer = actual_rpm / max(target_rpm, 1e-6)
            if blocked:
                if rs.mechanism is None:
                    rs.spinup_timer = 0.0
            elif rs.mechanism is None and rs.spinup_timer < spinup:
                rs.spinup_timer += dt
            elif rs.mechanism is None and rs.held and rs.score_timer <= 0:
                pid = rs.held.pop(0)
                p = self.pieces[pid]
                p.held_by = None
                p.vx = p.vy = 0.0
                p.vz = 0.0
                p.attrs["passed_goal_top"] = False
                p.attrs["passed_archway"] = False
                self._launch_ballistic(rs, p, launcher)
                rs.score_timer = cycle
        if verb == "open_gate":
            gate_el = next(
                (e for e in self.elements if e.get("type") == "gate" or "gate" in (e.get("tags") or [])),
                None,
            )
            if gate_el:
                sh = self.element_shapes[gate_el["id"]]
                if point_in_shape(sh, rs.body.x, rs.body.y):
                    self.gate_state[gate_el["id"]] = "open"
                    events.append(
                        TickEvent(
                            "contactStart",
                            volume_id=gate_el.get("triggerId") or gate_el["id"],
                            robot_id=rs.body.id,
                            fsm_id=gate_el["id"],
                        )
                    )
                    self.pending_piece_ops.append(("release_queue", None, None))
        rs.score_timer = max(0.0, rs.score_timer - dt)

    def _update_piece_ownership(self) -> None:
        storage_count = len((self.robot.get("piecePath") or {}).get("storageSlots") or [])
        capacity = storage_count or self.capacity
        for p in self.pieces.values():
            owner = self.robots.get(p.held_by or "")
            if owner is not None:
                dx, dy = p.x - owner.body.x, p.y - owner.body.y
                cos_h, sin_h = math.cos(owner.body.heading), math.sin(owner.body.heading)
                local_x = cos_h * dx + sin_h * dy
                local_y = -sin_h * dx + cos_h * dy
                retained = (
                    -self.robot_hx - p.radius <= local_x <= self.robot_hx + p.radius
                    and abs(local_y) <= self.robot_hy + p.radius
                    and self.floor_y <= p.z <= 0.5 * float(
                        (self.robot.get("chassis") or {}).get("heightIn") or 14.0
                    ) + owner.body.z + p.radius
                )
                if retained:
                    continue
                if p.id in owner.held:
                    owner.held.remove(p.id)
                p.held_by = None
                speed = math.sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz)
                p.in_flight = speed > 24.0 and p.z > self.floor_y + p.radius
                p.ballistic = p.in_flight

            if p.held_by or p.scored:
                continue
            for rs in self.robots.values():
                if rs.last_verb != "intake" or len(rs.held) >= capacity:
                    continue
                dx, dy = p.x - rs.body.x, p.y - rs.body.y
                cos_h, sin_h = math.cos(rs.body.heading), math.sin(rs.body.heading)
                local_x = cos_h * dx + sin_h * dy
                local_y = -sin_h * dx + cos_h * dy
                inside_magazine = (
                    -self.robot_hx + p.radius <= local_x <= self.robot_hx - p.radius
                    and abs(local_y) <= self.robot_hy - p.radius
                    and self.floor_y + p.radius <= p.z <= 6.0
                )
                if inside_magazine:
                    p.held_by = rs.body.id
                    rs.held.append(p.id)
                    p.in_flight = False
                    p.ballistic = False
                    break

    def _launch_ballistic(self, rs: RobotState, p: Piece, launcher: dict[str, Any] | None = None) -> None:
        if launcher is None:
            raise ValueError("physical launch requires a configured launcher")
        p.in_flight = True
        p.ballistic = True
        p.scored = False
        p.kick = True
        pose = dict(launcher.get("poseOnRobot") or {})
        yaw_deg, pitch_deg = launcher_aim(launcher)
        pose["headingDeg"] = yaw_deg
        pose["pitchDeg"] = pitch_deg
        mx, my, mz, yaw, pitch = pose_world(rs.body.x, rs.body.y, rs.body.heading, pose)
        speed = float(launcher.get("muzzleSpeedInPerS") or 180.0)
        p.x, p.y, p.z = mx, my, max(mz, p.radius)
        p.vx, p.vy, p.vz = muzzle_velocity(speed, yaw, pitch)
        # Galilean launch velocity: a moving robot cannot emit a world-fixed shot.
        p.vx += rs.body.vx
        p.vy += rs.body.vy

    def _advance_ballistic(self, dt: float) -> None:
        if getattr(self.backend, "name", "") == "mujoco_field":
            return
        g = 386.0886
        for p in self.pieces.values():
            if not p.ballistic or p.held_by or p.scored:
                continue
            p.vz -= g * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.z += p.vz * dt
            if p.z <= p.radius:
                p.z = p.radius
                p.vz *= -0.25
                p.vx *= 0.6
                p.vy *= 0.6
                p.ballistic = abs(p.vz) > 8.0
                p.in_flight = p.ballistic

    def _sense_robot(self, rs: RobotState) -> list[dict[str, Any]]:
        hits_all: list[dict[str, Any]] = []
        for cam in self.cameras:
            cx, cy, ch = camera_world_pose(rs.body.x, rs.body.y, rs.body.heading, cam)
            hits = detect_tags(cx, cy, ch, cam, self.tags, self.occluders, self.rng)
            hits_all.extend(hits)
            for hit in hits:
                if not hit.get("visible") or not hit.get("mapsTo"):
                    continue
                for var in self.scoring.get("matchVariables") or []:
                    role = (var.get("observeVia") or {}).get("tagRole")
                    if role and hit.get("role") == role:
                        self.observed_vars[var["id"]] = hit["mapsTo"]
        rs.vision_hits = hits_all
        return hits_all

    def _sense(self) -> None:
        self.vision_hits = []
        actor_id = self.actor().body.id if self.robots else "red_0"
        for rs in self.robots.values():
            hits = self._sense_robot(rs)
            if rs.body.id == actor_id:
                self.vision_hits = hits

    def _chassis_hits_other(self, rs: RobotState, slack: float = 0.6) -> bool:
        me_poly = rs.body.world_footprint()
        me = rs.body.aabb()
        for other in self.robots.values():
            if other.body.id == rs.body.id:
                continue
            other_poly = other.body.world_footprint()
            if me_poly is not None and other_poly is not None:
                hit, _mtv = polygons_overlap(me_poly, other_poly)
                if hit:
                    return True
                continue
            ob = other.body.aabb()
            if (
                me.minx < ob.maxx + slack
                and me.maxx + slack > ob.minx
                and me.miny < ob.maxy + slack
                and me.maxy + slack > ob.miny
            ):
                return True
        return False

    def _update_contacts_and_restricted(self, dt: float, occ: dict[str, set[str]]) -> None:
        for rs in self.robots.values():
            if self._chassis_hits_other(rs) or (self.robot_hit and rs.body.dynamic):
                rs.collision_time_s += dt
                if rs.first_contact_s is None:
                    rs.first_contact_s = self.time_s
            tag = f"restricted_for_{rs.body.alliance}"
            for tid, el in self.triggers.items():
                tags = el.get("tags") or []
                if tag in tags or el.get("id") == tag or el.get("triggerId") == tag:
                    if rs.body.id in occ.get(tid, set()):
                        rs.entered_restricted = True

    def _run_rules(self, events: list[TickEvent]) -> float:
        occ = self._occupancy()
        events.extend(self._events_from_occupancy(occ))
        rs = self.actor()
        actor_vols = {tid for tid, ids in occ.items() if rs.body.id in ids}
        ctx = RuleContext(
            phase=self.phase,
            events=events,
            volume_occupancy=occ,
            prev_occupancy=self.prev_occupancy,
            accumulators=self.accumulators,
            match_vars=self.match_vars,
            gate_state=self.gate_state,
            queues=self.queues,
            queue_caps=self.queue_caps,
            actor_id=rs.body.id,
            actor_volumes=actor_vols,
            fire_counts=self.fire_counts,
            ident_types={p.id: p.type_id for p in self.pieces.values()},
            piece_ops=self.pending_piece_ops,
        )
        explains, delta = self.engine.evaluate(ctx)
        self.step_explains = list(explains)
        self.explains.extend(explains)
        self.true_score = float(self.accumulators.get(self.engine.true_score_id) or self.true_score)
        self.prev_occupancy = occ
        self.last_events = events
        self._apply_piece_ops()
        return float(delta)

    def _apply_piece_ops(self) -> None:
        ops = list(self.pending_piece_ops)
        self.pending_piece_ops = []
        for kind, piece_id, extra in ops:
            if kind == "despawn" and piece_id and piece_id in self.pieces:
                p = self.pieces[piece_id]
                p.scored = True
                p.in_flight = False
                p.held_by = None
                p.vx = p.vy = 0.0
                for rs in self.robots.values():
                    if piece_id in rs.held:
                        rs.held.remove(piece_id)
            elif kind == "transfer" and piece_id and piece_id in self.pieces:
                if self.needs_mesh():
                    raise RuntimeError(
                        "transferPiece is prohibited in physically authoritative seasons"
                    )
                p = self.pieces[piece_id]
                p.scored = False
                p.in_flight = False
                p.ballistic = False
                p.held_by = None
                if extra:
                    el = next((e for e in self.elements if e["id"] == extra or e.get("triggerId") == extra), None)
                    if el:
                        p.attrs["initial_volume_id"] = el.get("triggerId") or el["id"]
            elif kind == "release_queue":
                for sid in list(self.queues):
                    self.queues[sid] = []
                    if sid in self.accumulators and isinstance(self.accumulators[sid], list):
                        self.accumulators[sid] = []
                for gid in self.gate_state:
                    self.gate_state[gid] = "open"

    def step(
        self,
        target_pose: np.ndarray | None = None,
        speed_frac: float = 1.0,
        mechanism: int = 0,
        end_phase: bool = False,
        actions: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if actions is None:
            if target_pose is None:
                rs0 = self.actor()
                target_pose = np.array([rs0.body.x, rs0.body.y, rs0.body.heading], dtype=np.float64)
            actions = {
                "red_0": {
                    "target_pose": np.asarray(target_pose, dtype=np.float64).reshape(3),
                    "speed_frac": float(speed_frac),
                    "mechanism": int(mechanism),
                }
            }
        events: list[TickEvent] = []
        self.wall_hit = False
        self.robot_hit = False
        self.piece_hit = False
        for _ in range(self.substeps):
            for rid, rs in self.robots.items():
                act = actions.get(rid) or {"target_pose": [rs.body.x, rs.body.y, rs.body.heading], "speed_frac": 0.2, "mechanism": 0}
                verb = MECHANISM_VERBS[int(act.get("mechanism", 0)) % len(MECHANISM_VERBS)]
                self._step_robot_dynamics(rs, act, verb, self.dt)
                self._apply_command(rs, act, self.dt)
                self._mechanisms(rs, verb, self.dt, events)
            floor_bodies: list[Body] = []
            floor_index: dict[str, Piece] = {}
            for p in self.pieces.values():
                if p.scored:
                    continue
                body = Body(
                    p.id,
                    p.x,
                    p.y,
                    0.0,
                    vx=p.vx,
                    vy=p.vy,
                    hx=p.radius,
                    hy=p.radius,
                    kind="circle",
                    radius=p.radius,
                    restitution=p.restitution,
                    mass=p.mass,
                    z=p.z,
                    vz=p.vz,
                    kick=p.kick,
                    type_id=p.type_id,
                    qw=p.qw,
                    qx=p.qx,
                    qy=p.qy,
                    qz=p.qz,
                    wx=p.wx,
                    wy=p.wy,
                    wz=p.wz,
                )
                p.kick = False
                floor_bodies.append(body)
                floor_index[p.id] = p
            flags = self.backend.step_world(
                # A HIVE alternates between its two stable positions on each tip.
                WorldStep(
                    robots=[o.body for o in self.robots.values()],
                    pieces=floor_bodies,
                    obstacles=self.obstacles,
                    walls=self.walls,
                    max_vel=self.max_vel,
                    max_accel=self.max_accel,
                    max_omega=self.max_ang_vel,
                    max_ang_accel=self.max_ang_accel,
                    field_mechanism_targets={
                        mechanism_id: (
                            tip_angle
                            * (
                                int(
                                    self.accumulators.get(
                                        f"{mechanism_id.removesuffix('_hive')}_tip_count"
                                    )
                                    or 0
                                )
                                % 2
                            )
                        )
                        for mechanism_id, tip_angle in self.field_mechanism_tip_angles.items()
                    },
                    robot_mechanism_states={
                        rid: rs.mechanism.state_dict()
                        for rid, rs in self.robots.items()
                        if rs.mechanism is not None
                    },
                    piece_owners={
                        p.id: p.held_by
                        for p in self.pieces.values()
                        if p.held_by is not None
                    },
                ),
                self.dt,
            )
            read_mechanisms = getattr(self.backend, "field_mechanism_positions", None)
            if callable(read_mechanisms):
                self.field_mechanisms.update(read_mechanisms())
            read_robot_joints = getattr(
                self.backend,
                "robot_mechanism_joint_states",
                None,
            )
            if callable(read_robot_joints):
                for rid, joints in read_robot_joints().items():
                    rs_joint = self.robots.get(rid)
                    if rs_joint is None or rs_joint.mechanism is None:
                        continue
                    for actuator in rs_joint.mechanism.actuators.values():
                        joint_id = str(actuator.config.get("jointId") or "")
                        measured = joints.get(joint_id)
                        if measured is None:
                            continue
                        actuator.state.velocity_rad_s = float(
                            measured["velocityRadS"]
                        )
                        if actuator.config.get("kind") != "velocity_motor":
                            actuator.state.position = math.degrees(
                                float(measured["positionRad"])
                            )
            self.wall_hit = self.wall_hit or flags.wall
            self.robot_hit = self.robot_hit or flags.robot
            self.piece_hit = self.piece_hit or flags.piece
            for body in floor_bodies:
                p = floor_index[body.id]
                p.x, p.y, p.vx, p.vy = body.x, body.y, body.vx, body.vy
                p.z = float(getattr(body, "z", p.z))
                p.vz = float(getattr(body, "vz", p.vz))
                p.qw = float(getattr(body, "qw", p.qw))
                p.qx = float(getattr(body, "qx", p.qx))
                p.qy = float(getattr(body, "qy", p.qy))
                p.qz = float(getattr(body, "qz", p.qz))
                p.wx = float(getattr(body, "wx", p.wx))
                p.wy = float(getattr(body, "wy", p.wy))
                p.wz = float(getattr(body, "wz", p.wz))
            self._advance_ballistic(self.dt)
            self._update_piece_ownership()
            self.time_s += self.dt
            occ_mid = self._occupancy()
            events.extend(self._events_from_occupancy(occ_mid))
            self._update_contacts_and_restricted(self.dt, occ_mid)
            self.prev_occupancy = occ_mid
        self._sense()
        rs = self.actor()
        if end_phase or self.time_s >= self.auto_s - 1e-9:
            events.append(TickEvent("phaseEnd", phase="AUTO", robot_id=rs.body.id))
        verb = MECHANISM_VERBS[int((actions.get("red_0") or {}).get("mechanism", 0)) % len(MECHANISM_VERBS)]
        delta = self._run_rules(events)
        occ = self.prev_occupancy
        self._update_contacts_and_restricted(0.0, occ)
        return {"true_score_delta": delta, "verb": verb}

    def _snapshot_robot(
        self,
        robot: RobotState,
        part_transforms: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        mechanism = robot.mechanism.state_dict() if robot.mechanism is not None else None
        actuators = mechanism.get("actuators") if isinstance((mechanism or {}).get("actuators"), dict) else {}
        power = self.robot.get("powerSystem") or {}
        return {
            "id": robot.body.id,
            "x": robot.body.x,
            "y": robot.body.y,
            "headingDeg": rad_to_deg(robot.body.heading),
            "held": list(robot.held),
            "dynamic": robot.body.dynamic,
            "alliance": robot.body.alliance,
            "collisionTimeS": robot.collision_time_s,
            "firstContactS": robot.first_contact_s,
            "enteredRestricted": robot.entered_restricted,
            "mechanism": mechanism,
            "parts": list(part_transforms.get(robot.body.id) or []),
            "lastVerb": robot.last_verb,
            "actuators": {
                ident: {
                    "rpm": float(payload.get("rpm") or 0.0),
                    "currentA": float(payload.get("currentA") or 0.0),
                    "command": float(payload.get("command") or 0.0),
                    "position": float(payload.get("position") or 0.0),
                }
                for ident, payload in actuators.items()
                if isinstance(payload, dict)
            },
            "batteryVoltageV": (
                float(mechanism.get("batteryVoltageV") or power.get("openCircuitVoltageV") or 12.0)
                if mechanism is not None
                else None
            ),
            "batteryCurrentA": (
                float(mechanism.get("batteryCurrentA") or 0.0) if mechanism is not None else None
            ),
        }

    def snapshot(self) -> dict[str, Any]:
        rs = self.actor()
        read_part_transforms = getattr(
            self.backend,
            "robot_mechanism_transforms",
            None,
        )
        part_transforms = (
            read_part_transforms() if callable(read_part_transforms) else {}
        )
        physical = bool(
            self.robot.get("actuators")
            or self.robot.get("piecePath")
            or self.robot.get("rigidParts")
        )
        stored_slot = {
            pid: index
            for r in self.robots.values()
            for index, pid in enumerate(r.held)
        }
        actor = rs
        actor_mech = actor.mechanism.state_dict() if actor.mechanism is not None else {}
        actor_actuators = actor_mech.get("actuators") if isinstance(actor_mech.get("actuators"), dict) else {}
        return {
            "t": self.time_s,
            "phase": self.phase,
            "trueScore": self.true_score,
            "physicalPieces": physical,
            "mechanismCommands": {
                ident: float(payload.get("command") or 0.0)
                for ident, payload in actor_actuators.items()
                if isinstance(payload, dict)
            },
            "robots": [
                self._snapshot_robot(r, part_transforms)
                for r in self.robots.values()
            ],
            "robotDesign": {
                "chassis": dict(self.robot.get("chassis") or {}),
                "intakes": list(self.intakes),
                "launchers": list(self.launchers),
                "rigidParts": list(self.robot.get("rigidParts") or []),
                "joints": list(self.robot.get("joints") or []),
                "actuators": list(self.robot.get("actuators") or []),
                "mechanismSensors": list(
                    self.robot.get("mechanismSensors") or []
                ),
                "powerSystem": dict(self.robot.get("powerSystem") or {}),
                "piecePath": dict(self.robot.get("piecePath") or {}),
                "policyInterfaceVersion": self.robot.get(
                    "policyInterfaceVersion"
                ),
                "visualAsset": self.robot.get("visualAsset"),
                "visualOffset": dict(self.robot.get("visualOffset") or {}),
                "collisionShape": (self.robot.get("chassis") or {}).get("collisionShape"),
            },
            "pieces": [
                {
                    "id": p.id,
                    "typeId": p.type_id,
                    "x": p.x,
                    "y": p.y,
                    "z": p.z,
                    "quat": [p.qw, p.qx, p.qy, p.qz],
                    "qw": p.qw,
                    "qx": p.qx,
                    "qy": p.qy,
                    "qz": p.qz,
                    "headingDeg": rad_to_deg(math.atan2(2.0 * (p.qw * p.qy - p.qz * p.qx), 1.0 - 2.0 * (p.qy * p.qy + p.qz * p.qz))),
                    "radius": p.radius,
                    "visualAsset": p.visual_asset,
                    "collisionAsset": p.collision_asset,
                    "color": p.attrs.get("color"),
                    "heldBy": p.held_by,
                    "storedSlot": stored_slot.get(p.id),
                    "inFlight": p.in_flight,
                    "scored": p.scored,
                }
                for p in self.pieces.values()
            ],
            "queues": dict(self.queues),
            "gate": dict(self.gate_state),
            "matchVarsPrivileged": dict(self.match_vars),
            "observedMatchVars": dict(self.observed_vars),
            "vision": self.vision_hits,
            "events": [e.explain if hasattr(e, "explain") else e.kind for e in self.last_events[-8:]],
            "explains": list(self.explains[-6:]),
            "stepExplains": list(self.step_explains),
            "penalties": [e for e in self.explains if float(e.get("points") or 0) < 0],
            "collision": {
                "wall": self.wall_hit,
                "robot": self.robot_hit,
                "piece": self.piece_hit,
                "collisionTimeS": rs.collision_time_s,
                "firstContactS": rs.first_contact_s,
                "enteredRestricted": rs.entered_restricted,
            },
            "fieldSizeIn": self.field["fieldSizeIn"],
            "fieldId": self.field.get("id"),
            "backgroundAsset": self.field.get("backgroundAsset"),
            "collisionAsset": self.field.get("collisionAsset"),
            "cadManifest": self.field.get("cadManifest"),
            "cadSourceSha256": self._cad_source_sha256,
            "cadAssetVersion": self._cad_asset_version,
            "gamePieces": [
                {
                    "typeId": spec["typeId"],
                    "displayName": spec.get("displayName"),
                    "shape": dict(spec.get("shape") or {}),
                    "color": spec.get("color"),
                    "visualAsset": spec.get("visualAsset"),
                    "collisionAsset": spec.get("collisionAsset"),
                }
                for spec in (self.field.get("gamePieces") or [])
                if spec.get("typeId")
            ],
            "cadCollision": getattr(self.backend, "cad_stats", None),
            "fieldMechanisms": [
                {"id": mechanism_id, "angleRad": angle}
                for mechanism_id, angle in self.field_mechanisms.items()
            ],
            "physicsBackend": getattr(self.backend, "name", None),
            "elements": [
                {
                    "id": el["id"],
                    "type": el.get("type"),
                    "alliance": el.get("alliance"),
                    "pose": el.get("pose") or {"x": 0, "y": 0, "headingDeg": 0},
                    "shape": el.get("shape"),
                    "tags": el.get("tags") or [],
                    "isOccluder": bool(el.get("isOccluder")),
                }
                for el in self.elements
            ],
            "aprilTags": [
                {
                    "tagId": t.get("tagId"),
                    "role": t.get("role"),
                    "pose": t.get("pose") or {"x": 0, "y": 0, "z": 12, "headingDeg": 0},
                    "sizeIn": t.get("sizeIn") or 8,
                    "mapsTo": t.get("mapsTo"),
                }
                for t in (self.field.get("aprilTags") or [])
            ],
            "noisyPose": [
                rs.body.x + float(self.rng.normal(0, self.pos_noise)),
                rs.body.y + float(self.rng.normal(0, self.pos_noise)),
                wrap_angle(rs.body.heading + float(self.rng.normal(0, self.heading_noise))),
            ],
        }
