import type { CatalogPart, FunctionalBindings, RobotAssembly } from "../api";
import { validateAssemblyGraph } from "./assemblyMath";
import { inferredJointType, partMount } from "./mounts";

export const TOPOLOGY_NOTE = "mechanism topology from mecanum_biobuzz_4cap (unconfirmed)";

const WHEEL_TAGS = new Set(["wheel_mecanum", "wheel_traction", "wheel_omni", "wheel_compliant"]);
const STRUCTURE_TAGS = new Set(["channel", "extrusion", "plate", "bracket"]);
const DRIVE_TAGS = new Set(["motor", "gearbox"]);

export type InferenceJoint = {
  connectionId: string;
  parentId: string;
  childId: string;
  type: string;
};

export type InferenceReport = {
  confirmed: boolean;
  roles: Record<string, string>;
  joints: InferenceJoint[];
  notes: string[];
  weldedInstanceIds: string[];
  articulatedInstanceIds: string[];
};

export function inferRole(part: CatalogPart | undefined, bindings?: FunctionalBindings, instanceId = ""): string {
  const bound = bindings?.instanceRoles || {};
  if (instanceId && bound[instanceId]) return bound[instanceId];
  if (!part) return "structure";
  const tags = new Set(part.tags || []);
  if (tags.has("intake_roller")) return "intake";
  if (tags.has("flywheel")) return "flywheel";
  if ([...tags].some((tag) => WHEEL_TAGS.has(tag))) return "drive_wheel";
  if (tags.has("servo")) return "servo";
  if (tags.has("sensor")) return "sensor";
  if ([...tags].some((tag) => DRIVE_TAGS.has(tag))) return "drive_motor";
  if ([...tags].some((tag) => STRUCTURE_TAGS.has(tag))) return "structure";
  return "structure";
}

export function buildInferenceReport(
  assembly: RobotAssembly,
  catalog: Record<string, CatalogPart>,
  bindings?: FunctionalBindings,
): InferenceReport {
  const { byChild } = validateAssemblyGraph(assembly);
  const roles: Record<string, string> = {};
  for (const instance of assembly.instances) {
    roles[instance.id] = inferRole(catalog[instance.id], bindings, instance.id);
  }
  const joints: InferenceJoint[] = [];
  const welded: string[] = [];
  const articulated: string[] = [];
  for (const connection of byChild.values()) {
    const childId = connection.child.instanceId;
    const parentId = connection.parent.instanceId;
    const parentPart = catalog[parentId];
    const childPart = catalog[childId];
    if (!parentPart || !childPart) continue;
    const explicit = connection.jointType;
    const jointType =
      explicit ||
      inferredJointType(partMount(parentPart, connection.parent.mountId), partMount(childPart, connection.child.mountId));
    joints.push({ connectionId: connection.id, parentId, childId, type: jointType });
    if (jointType === "fixed") welded.push(childId);
    else articulated.push(childId);
  }
  const notes = [TOPOLOGY_NOTE];
  if (!bindings?.confirmed) notes.push("functional bindings are inferred and not confirmed");
  return {
    confirmed: competitiveSaveBlocked(bindings, assembly.instances.length) === null,
    roles,
    joints,
    notes,
    weldedInstanceIds: welded,
    articulatedInstanceIds: articulated,
  };
}

export function confirmBindings(report: InferenceReport, drivetrainType: string, trackWidthIn: number, wheelbaseIn?: number): FunctionalBindings {
  return {
    confirmed: true,
    drivetrain: {
      type: drivetrainType,
      trackWidthIn: Number(trackWidthIn.toFixed(3)),
      ...(wheelbaseIn != null ? { wheelbaseIn: Number(wheelbaseIn.toFixed(3)) } : {}),
    },
    instanceRoles: { ...report.roles },
  };
}

export function competitiveSaveBlocked(bindings?: FunctionalBindings, assemblyPartCount = 0): string | null {
  if (!assemblyPartCount) return null;
  if (!bindings?.confirmed) {
    return "Confirm inferred drivetrain, transmissions, and part roles before saving a competitive robot.";
  }
  if (!bindings.drivetrain?.type) {
    return "Confirmed bindings must include a drivetrain type.";
  }
  if (!bindings.instanceRoles || Object.keys(bindings.instanceRoles).length < assemblyPartCount) {
    return "Confirmed bindings must include a role for every catalog instance.";
  }
  return null;
}
