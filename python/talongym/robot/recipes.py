"""Load guided drivebase recipes and instantiate schema 1.2 catalog assemblies."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from talongym.paths import PRESETS_DIR
from talongym.presets.loader import validate_document
from talongym.robot.assembly import CompiledAssembly, compile_assembly_to_preset, solve_assembly_poses
from talongym.robot.catalog import get_part
from talongym.robot.contract import ASSEMBLY_SCHEMA_VERSION, AssemblyError
from talongym.robot.graph import validate_assembly_graph
from talongym.robot.mounts import part_mount
from talongym.robot.transmissions import part_tags, round_measure, wheel_radius_in

RECIPE_SCHEMA_VERSION = "1.0.0"
RECIPE_DIR = PRESETS_DIR / "robot_recipes"
RECIPE_IDS = ("gobilda_mecanum", "gobilda_tank", "rev_mecanum", "rev_tank")
_CORNERS = ("fl", "fr", "rl", "rr")
_GOBILDA_BRACKET = "1203-0001-0001"
_GOBILDA_PLATE = "1202-0001-0001"
_GOBILDA_HUB = "1310-0016-0008"
_GOBILDA_SENSOR = "3110-0001-0001"
_GOBILDA_SERVO_MOUNT = "1201-0043-0002"
_GOBILDA_SERVO = "2000-0025-0002"
_REV_CORNER = "REV-41-1480"
_REV_PLASTIC = "REV-41-1305"
_REV_UP_BRACKET = "REV-41-1621"
_REV_CARTRIDGE = "REV-41-1603"
_REV_HUB = "REV-31-1595"
_REV_SPARK = "REV-11-1271"


class RecipeError(RuntimeError):
    pass


def _read_recipe_path(recipe_id: str):
    path = RECIPE_DIR / f"{recipe_id}.json"
    if not path.is_file():
        raise RecipeError(f"unknown drivebase recipe {recipe_id}")
    return path


@lru_cache(maxsize=1)
def load_recipes() -> dict[str, dict[str, Any]]:
    recipes: dict[str, dict[str, Any]] = {}
    for recipe_id in RECIPE_IDS:
        document = json.loads(_read_recipe_path(recipe_id).read_text(encoding="utf-8"))
        _validate_recipe_document(document, recipe_id)
        recipes[recipe_id] = document
    return recipes


def reload_recipes() -> dict[str, dict[str, Any]]:
    load_recipes.cache_clear()
    return load_recipes()


def recipe_public(recipe: dict[str, Any]) -> dict[str, Any]:
    parameters = recipe["parameters"]
    defaults = {name: spec.get("default") for name, spec in parameters.items()}
    return {
        "id": recipe["id"],
        "displayName": recipe["displayName"],
        "manufacturer": recipe["manufacturer"],
        "drivetrain": recipe["drivetrainType"],
        "description": recipe.get("description"),
        "defaultParameters": defaults,
        "lengthSkus": list((parameters.get("lengthSku") or {}).get("choices") or []),
        "widthSkus": list((parameters.get("widthSku") or {}).get("choices") or []),
        "motorSkus": list((parameters.get("motorSku") or {}).get("choices") or []),
        "wheelSkus": list((parameters.get("wheelSku") or {}).get("choices") or []),
        "cartridgeSkus": list((parameters.get("cartridgeSku") or {}).get("choices") or []),
        "parameters": parameters,
    }


def list_public_recipes() -> list[dict[str, Any]]:
    return [recipe_public(load_recipe(recipe_id)) for recipe_id in RECIPE_IDS]


def list_recipes() -> list[dict[str, Any]]:
    return [load_recipe(recipe_id) for recipe_id in RECIPE_IDS]


def load_recipe(recipe_id: str) -> dict[str, Any]:
    recipes = load_recipes()
    if recipe_id not in recipes:
        raise RecipeError(f"unknown drivebase recipe {recipe_id}")
    return dict(recipes[recipe_id])


def _validate_recipe_document(document: dict[str, Any], recipe_id: str) -> None:
    if str(document.get("schemaVersion")) != RECIPE_SCHEMA_VERSION:
        raise RecipeError(f"{recipe_id} must use recipe schema {RECIPE_SCHEMA_VERSION}")
    if str(document.get("id")) != recipe_id:
        raise RecipeError(f"{recipe_id} id mismatch")
    if document.get("manufacturer") not in {"gobilda", "rev"}:
        raise RecipeError(f"{recipe_id} has unknown manufacturer")
    if document.get("drivetrainType") not in {"mecanum", "tank"}:
        raise RecipeError(f"{recipe_id} has unknown drivetrainType")
    parameters = document.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise RecipeError(f"{recipe_id} is missing parameters")
    for name in ("lengthSku", "widthSku", "motorSku", "wheelSku", "includeElectronics"):
        if name not in parameters:
            raise RecipeError(f"{recipe_id} is missing parameter {name}")
    if document.get("manufacturer") == "rev" and "cartridgeSku" not in parameters:
        raise RecipeError(f"{recipe_id} is missing parameter cartridgeSku")


def _pattern_counts(sku: str, mount_id: str) -> tuple[int, int]:
    pattern = part_mount(get_part(sku), mount_id).get("pattern") or {}
    return int(pattern.get("countU") or 1), int(pattern.get("countV") or 1)


def _choice(spec: dict[str, Any], name: str, raw: dict[str, Any]) -> Any:
    if spec.get("type") == "boolean":
        default = bool(spec.get("default"))
        if name not in raw:
            return default
        value = raw[name]
        if not isinstance(value, bool):
            raise RecipeError(f"parameter {name} must be a boolean")
        return value
    choices = list(spec.get("choices") or [])
    default = spec.get("default")
    value = raw.get(name, default)
    if value not in choices:
        raise RecipeError(f"parameter {name} value {value!r} is not a recipe choice")
    return value


def _resolved_parameters(recipe: dict[str, Any], overlay: dict[str, Any] | None) -> dict[str, Any]:
    specs = recipe["parameters"]
    raw = dict(overlay or {})
    unknown = sorted(key for key in raw if key not in specs)
    if unknown:
        raise RecipeError(f"unknown recipe parameters: {unknown}")
    resolved = {name: _choice(spec, name, raw) for name, spec in specs.items()}
    manufacturer = str(recipe["manufacturer"])
    for name, value in resolved.items():
        if not name.lower().endswith("sku"):
            continue
        part = get_part(str(value))
        if str(part.get("manufacturer")) != manufacturer:
            raise RecipeError(f"{value} is not a {manufacturer} catalog sku")
    wheel = get_part(str(resolved["wheelSku"]))
    tags = part_tags(wheel)
    wanted = str(recipe["drivetrainType"])
    if wanted == "mecanum" and "wheel_mecanum" not in tags:
        raise RecipeError(f"{resolved['wheelSku']} is not a mecanum wheel")
    if wanted == "tank" and not tags.intersection({"wheel_traction", "wheel_omni"}):
        raise RecipeError(f"{resolved['wheelSku']} is not a tank wheel")
    motor = get_part(str(resolved["motorSku"]))
    if not part_tags(motor).intersection({"motor", "gearbox"}):
        raise RecipeError(f"{resolved['motorSku']} is not a drive motor")
    return resolved


def _ref(instance_id: str, mount_id: str, index: tuple[int, int] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"instanceId": instance_id, "mountId": mount_id}
    if index is not None:
        row["patternIndex"] = [int(index[0]), int(index[1])]
    return row


def _conn(
    ident: str,
    parent_id: str,
    parent_mount: str,
    parent_index: tuple[int, int] | None,
    child_id: str,
    child_mount: str,
    child_index: tuple[int, int] | None,
    secondary: tuple[tuple[int, int], tuple[int, int]] | None = None,
    spin_deg: float = 0.0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": ident,
        "parent": _ref(parent_id, parent_mount, parent_index),
        "child": _ref(child_id, child_mount, child_index),
    }
    if secondary:
        row["secondary"] = {
            "parent": _ref(parent_id, parent_mount, secondary[0]),
            "child": _ref(child_id, child_mount, secondary[1]),
        }
    if abs(float(spin_deg)) > 1e-12:
        row["spinDeg"] = float(spin_deg)
    return row


def _instance(ident: str, sku: str, pose: dict[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"id": ident, "sku": sku}
    if pose:
        row["pose"] = pose
    return row


def _wheel_diameter(sku: str) -> float | None:
    radius = wheel_radius_in(get_part(sku))
    return None if radius is None else 2.0 * float(radius)


def _document(
    recipe: dict[str, Any],
    parameters: dict[str, Any],
    root_id: str,
    instances: list[dict[str, Any]],
    connections: list[dict[str, Any]],
    *,
    warnings: list[dict[str, Any]] | None = None,
    track_width_in: float | None = None,
    wheelbase_in: float | None = None,
) -> dict[str, Any]:
    drivetrain: dict[str, Any] = {"type": recipe["drivetrainType"]}
    if recipe["drivetrainType"] == "mecanum":
        drivetrain["strafeMultiplier"] = 1.0
    diameter = _wheel_diameter(str(parameters["wheelSku"]))
    if diameter:
        drivetrain["wheelDiameterIn"] = round_measure(diameter)
    if track_width_in and track_width_in > 0:
        drivetrain["trackWidthIn"] = round_measure(track_width_in)
    if wheelbase_in and wheelbase_in > 0:
        drivetrain["wheelbaseIn"] = round_measure(wheelbase_in)
    document: dict[str, Any] = {
        "schemaVersion": ASSEMBLY_SCHEMA_VERSION,
        "id": str(recipe["id"]),
        "displayName": str(recipe["displayName"]),
        "assembly": {
            "rootInstanceId": root_id,
            "recipe": {"id": str(recipe["id"]), "parameters": parameters},
            "instances": instances,
            "connections": connections,
        },
        "functionalBindings": {"confirmed": False, "drivetrain": drivetrain},
    }
    if warnings:
        document["warnings"] = warnings
    errors = validate_document("robot", document)
    if errors:
        raise RecipeError("instantiated recipe failed schema 1.2: " + "; ".join(errors[:8]))
    return document


def _geometry_from_poses(instances: list[dict[str, Any]], connections: list[dict[str, Any]], root_id: str) -> tuple[float, float]:
    assembly = {"rootInstanceId": root_id, "instances": instances, "connections": connections}
    root, order, by_child = validate_assembly_graph(assembly)
    indexed = {row["id"]: row for row in instances}
    poses = solve_assembly_poses(indexed, {ident: get_part(str(row["sku"])) for ident, row in indexed.items()}, root, order, by_child)
    wheels = [ident for ident in indexed if ident.startswith("wheel_")]
    rails = [ident for ident in ("left_rail", "right_rail") if ident in poses]
    left = [ident for ident in wheels if ident.endswith(("_fl", "_rl"))]
    right = [ident for ident in wheels if ident.endswith(("_fr", "_rr"))]
    if left and right:
        left_y = sum(float(poses[ident][1, 3]) for ident in left) / len(left)
        right_y = sum(float(poses[ident][1, 3]) for ident in right) / len(right)
        xs = [float(poses[ident][0, 3]) for ident in left + right]
        return abs(left_y - right_y), abs(max(xs) - min(xs))
    if len(wheels) >= 2:
        points = [poses[ident][:3, 3] for ident in wheels]
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        return abs(max(ys) - min(ys)), abs(max(xs) - min(xs))
    if len(rails) == 2:
        delta = poses[rails[1]][:2, 3] - poses[rails[0]][:2, 3]
        span = float((delta * delta).sum() ** 0.5)
        return span, span
    return 0.0, 0.0


def _instantiate_gobilda(recipe: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    length_sku = str(parameters["lengthSku"])
    width_sku = str(parameters["widthSku"])
    motor_sku = str(parameters["motorSku"])
    wheel_sku = str(parameters["wheelSku"])
    count_u, count_v = _pattern_counts(length_sku, "web")
    width_u, width_v = _pattern_counts(width_sku, "web")
    flange_u, _ = _pattern_counts(length_sku, "flange_a")
    if count_v < 3 or width_v < 3:
        raise RecipeError(f"{length_sku}/{width_sku} web pattern is too small for a drivebase recipe")
    if count_u < 12 or flange_u < 12 or width_u < 4:
        raise RecipeError(f"{length_sku}/{width_sku} does not have enough holes for four drive modules")
    rear_motor = count_u - 6
    instances = [
        _instance("left_rail", length_sku),
        _instance("front_rail", width_sku),
        _instance("right_rail", length_sku),
        _instance("rear_rail", width_sku),
        _instance("plate_center", _GOBILDA_PLATE),
        _instance("hub_center", _GOBILDA_HUB),
        _instance("bracket_fl", _GOBILDA_BRACKET),
        _instance("bracket_fr", _GOBILDA_BRACKET),
    ]
    connections = [
        _conn("front_rail", "left_rail", "web", (0, 1), "front_rail", "web", (0, 1), spin_deg=90.0),
        _conn("right_rail", "front_rail", "web", (width_u - 1, 1), "right_rail", "web", (count_u - 1, 1), spin_deg=180.0),
        _conn("rear_rail", "left_rail", "web", (count_u - 1, 1), "rear_rail", "web", (0, 1), spin_deg=90.0),
        _conn("motor_fl", "left_rail", "flange_a", (3, 0), "motor_fl", "face", (0, 0), ((5, 0), (1, 0))),
        _conn("motor_rl", "left_rail", "flange_a", (rear_motor, 0), "motor_rl", "face", (0, 0), ((rear_motor + 2, 0), (1, 0))),
        _conn("motor_fr", "right_rail", "flange_a", (3, 0), "motor_fr", "face", (0, 0), ((5, 0), (1, 0))),
        _conn("motor_rr", "right_rail", "flange_a", (rear_motor, 0), "motor_rr", "face", (0, 0), ((rear_motor + 2, 0), (1, 0))),
        _conn("plate_center", "front_rail", "web", (max(width_u // 2, 1), 1), "plate_center", "pattern", (0, 0), ((max(width_u // 2, 1) + 1, 1), (1, 0))),
        _conn("hub_center", "plate_center", "pattern", (2, 0), "hub_center", "pattern", (0, 0)),
        _conn("bracket_fl", "left_rail", "web", (8, 1), "bracket_fl", "face_a", (0, 0), ((9, 1), (1, 0))),
        _conn("bracket_fr", "right_rail", "web", (8, 1), "bracket_fr", "face_a", (0, 0), ((9, 1), (1, 0))),
    ]
    for corner in _CORNERS:
        instances.append(_instance(f"motor_{corner}", motor_sku))
        instances.append(_instance(f"wheel_{corner}", wheel_sku))
        connections.append(_conn(f"wheel_{corner}", f"motor_{corner}", "output", None, f"wheel_{corner}", "bore", None))
    if parameters["includeElectronics"]:
        servo_u = 11 if count_u > 14 else 6
        instances.extend(
            [
                _instance("sensor_imu", _GOBILDA_SENSOR),
                _instance("servo_mount", _GOBILDA_SERVO_MOUNT),
                _instance("servo_front", _GOBILDA_SERVO),
            ]
        )
        connections.extend(
            [
                _conn("sensor_imu", "plate_center", "pattern", (2, 2), "sensor_imu", "channel", (0, 0), ((3, 2), (1, 0))),
                _conn("servo_mount", "left_rail", "web", (servo_u, 0), "servo_mount", "pattern", (0, 0), ((servo_u + 1, 0), (1, 0))),
                _conn("servo_front", "servo_mount", "pattern", (2, 0), "servo_front", "tabs", (0, 0)),
            ]
        )
    track, wheelbase = _geometry_from_poses(instances, connections, "left_rail")
    return _document(
        recipe,
        parameters,
        "left_rail",
        instances,
        connections,
        track_width_in=track,
        wheelbase_in=wheelbase,
    )


def _instantiate_rev(recipe: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    length_sku = str(parameters["lengthSku"])
    width_sku = str(parameters["widthSku"])
    motor_sku = str(parameters["motorSku"])
    wheel_sku = str(parameters["wheelSku"])
    cartridge_sku = str(parameters.get("cartridgeSku") or _REV_CARTRIDGE)
    opposite = str(parameters.get("wheelSkuOpposite") or wheel_sku)
    length_u, _ = _pattern_counts(length_sku, "slot")
    width_u, _ = _pattern_counts(width_sku, "slot")
    if length_u < 8 or width_u < 4:
        raise RecipeError(f"{length_sku}/{width_sku} slot pattern is too small for a drivebase recipe")
    front_motor = 3
    rear_motor = max(length_u - 5, front_motor + 2)
    instances = [
        _instance("left_rail", length_sku),
        _instance("front_rail", width_sku),
        _instance("right_rail", length_sku),
        _instance("rear_rail", width_sku),
        _instance("corner_fl", _REV_CORNER),
        _instance("corner_fr", _REV_CORNER),
        _instance("corner_rl", _REV_CORNER),
        _instance("corner_rr", _REV_CORNER),
        _instance("up_fl", _REV_UP_BRACKET),
        _instance("up_fr", _REV_UP_BRACKET),
        _instance("up_rl", _REV_UP_BRACKET),
        _instance("up_rr", _REV_UP_BRACKET),
        _instance("cartridge_fl", cartridge_sku),
        _instance("cartridge_fr", cartridge_sku),
        _instance("cartridge_rl", cartridge_sku),
        _instance("cartridge_rr", cartridge_sku),
        _instance("bracket_fl", _REV_PLASTIC),
        _instance("bracket_fr", _REV_PLASTIC),
    ]
    connections = [
        _conn("corner_fl", "left_rail", "slot", (0, 0), "corner_fl", "face_a", (0, 0), ((1, 0), (1, 0))),
        _conn("front_rail", "corner_fl", "face_b", (0, 0), "front_rail", "slot", (0, 0), ((1, 0), (1, 0))),
        _conn("corner_fr", "front_rail", "slot", (width_u - 1, 0), "corner_fr", "face_a", (0, 0), ((width_u - 2, 0), (1, 0))),
        _conn("right_rail", "corner_fr", "face_b", (0, 0), "right_rail", "slot", (0, 0)),
        _conn("corner_rl", "left_rail", "slot", (length_u - 1, 0), "corner_rl", "face_a", (0, 0), ((length_u - 2, 0), (1, 0))),
        _conn("rear_rail", "corner_rl", "face_b", (0, 0), "rear_rail", "slot", (0, 0), spin_deg=180.0),
        _conn("corner_rr", "right_rail", "slot", (length_u - 1, 0), "corner_rr", "face_a", (0, 0), ((length_u - 2, 0), (1, 0))),
        _conn("up_fl", "left_rail", "slot", (front_motor, 0), "up_fl", "extrusion", (0, 0), ((front_motor + 1, 0), (1, 0))),
        _conn("up_rl", "left_rail", "slot", (rear_motor, 0), "up_rl", "extrusion", (0, 0), ((rear_motor + 1, 0), (1, 0))),
        _conn("up_fr", "right_rail", "slot", (front_motor, 0), "up_fr", "extrusion", (0, 0), ((front_motor + 1, 0), (1, 0))),
        _conn("up_rr", "right_rail", "slot", (rear_motor, 0), "up_rr", "extrusion", (0, 0), ((rear_motor + 1, 0), (1, 0))),
        _conn("bracket_fl", "corner_fl", "face_a", (0, 1), "bracket_fl", "face_a", (0, 0), ((1, 1), (1, 0))),
        _conn("bracket_fr", "corner_fr", "face_a", (0, 1), "bracket_fr", "face_a", (0, 0), ((1, 1), (1, 0))),
    ]
    wheels = {
        "fl": wheel_sku,
        "rl": wheel_sku,
        "fr": opposite,
        "rr": opposite,
    }
    for corner in _CORNERS:
        instances.append(_instance(f"motor_{corner}", motor_sku))
        instances.append(_instance(f"wheel_{corner}", wheels[corner]))
        connections.append(
            _conn(
                f"motor_{corner}",
                f"up_{corner}",
                "motor_face",
                (0, 0),
                f"motor_{corner}",
                "face",
                (0, 0),
                ((1, 0), (1, 0)),
            )
        )
        connections.append(
            _conn(f"cartridge_{corner}", f"motor_{corner}", "output", None, f"cartridge_{corner}", "input", None)
        )
        connections.append(
            _conn(f"wheel_{corner}", f"cartridge_{corner}", "output", None, f"wheel_{corner}", "bore", None)
        )
    if parameters["includeElectronics"]:
        instances.extend([_instance("control_hub", _REV_HUB), _instance("spark_mini", _REV_SPARK)])
        connections.extend(
            [
                _conn("control_hub", "up_rl", "extrusion", (2, 0), "control_hub", "case", (0, 0), ((3, 0), (1, 0))),
                _conn("spark_mini", "up_rr", "extrusion", (2, 0), "spark_mini", "case", (0, 0)),
            ]
        )
    track, wheelbase = _geometry_from_poses(instances, connections, "left_rail")
    return _document(
        recipe,
        parameters,
        "left_rail",
        instances,
        connections,
        track_width_in=track,
        wheelbase_in=wheelbase,
    )


def instantiate_recipe(recipe_id: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Expand a guided drivebase recipe into a schema 1.2 assembly document."""
    recipe = load_recipe(recipe_id)
    resolved = _resolved_parameters(recipe, parameters)
    if recipe["manufacturer"] == "gobilda":
        return _instantiate_gobilda(recipe, resolved)
    return _instantiate_rev(recipe, resolved)


def instantiate_and_compile(
    recipe_id: str,
    parameters: dict[str, Any] | None = None,
    *,
    competitive: bool = True,
) -> CompiledAssembly:
    document = instantiate_recipe(recipe_id, parameters)
    try:
        return compile_assembly_to_preset(document, competitive=competitive)
    except AssemblyError as exc:
        raise RecipeError(f"{recipe_id} failed assembly compile: {exc}") from exc
