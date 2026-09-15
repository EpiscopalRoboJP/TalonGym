import { Component, Suspense, useEffect, useMemo, useState, type ReactNode } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { Html, OrbitControls, Grid, Line, useGLTF } from "@react-three/drei";
import { DoubleSide, FrontSide, Mesh, Object3D } from "three";
import type {
  ActuatorSpec,
  Frame,
  FramePiece,
  FrameRobot,
  IntakeSpec,
  LauncherSpec,
  PiecePathSpec,
  RigidPartSpec,
  RobotDesign,
  RobotPartTransform,
} from "../api";
import { API } from "../api";
import { theme } from "../theme";
import {
  buildPieceCatalog,
  cadCacheTokenFromManifest,
  explicitCadCacheToken,
  fetchCadManifest,
  fieldAssetUrl,
  ftcToThreePosition,
  hasYupQuaternion,
  pieceAssetUrl,
  pieceThreePose,
  resolveBackgroundAsset,
  resolveCadManifestPath,
  resolvePieceVisualAsset,
  visualOffsetYawDeg,
  type CadManifest,
  type PieceCatalog,
} from "./cadAssets";

export { resolveBackgroundAsset } from "./cadAssets";

export type SceneView = "threeQuarter" | "top";

type El = Frame["elements"][number];

function inch(n: number) {
  return n;
}

const GRAVITY_IN_S2 = 386.088;

export function usesPhysicalStorage(frame: Pick<Frame, "physicalPieces" | "robots" | "pieces">): boolean {
  if (frame.physicalPieces === true) return true;
  if ((frame.robots || []).some((robot) => (robot.parts && robot.parts.length > 0) || robot.batteryVoltageV != null)) {
    return true;
  }
  return (frame.pieces || []).some((piece) => Boolean(piece.heldBy) && typeof piece.storedSlot === "number");
}

export function launchArcPoints(
  path: PiecePathSpec,
  flywheelFrac: number,
  hoodFrac: number,
  actuators?: ActuatorSpec[],
): [number, number, number][] {
  const muzzle = path.muzzlePose || {};
  const hood = actuators?.find((row) => row.id === path.hoodActuatorId);
  const flywheel = actuators?.find((row) => row.id === path.flywheelActuatorId);
  const travel = hood?.travelLimit || [25, 70];
  const pitchDeg = muzzle.pitchDeg ?? travel[0] + hoodFrac * (travel[1] - travel[0]);
  const yawDeg = muzzle.yawDeg ?? 0;
  const rpm = (flywheel?.targetRpm || 0) * flywheelFrac;
  const radius = path.wheelRadiusIn || 2;
  const efficiency = path.launchEfficiency ?? 0.235;
  const speed = Math.max(8, radius * (rpm * (2 * Math.PI) / 60) * efficiency);
  const pitch = (pitchDeg * Math.PI) / 180;
  const yaw = (yawDeg * Math.PI) / 180;
  const vx = speed * Math.cos(pitch) * Math.cos(yaw);
  const vy = speed * Math.cos(pitch) * Math.sin(yaw);
  const vz = speed * Math.sin(pitch);
  const x0 = muzzle.x || 0;
  const y0 = muzzle.y || 0;
  const z0 = muzzle.z || 12;
  const points: [number, number, number][] = [];
  for (let i = 0; i <= 24; i += 1) {
    const t = i * 0.02;
    const z = z0 + vz * t - 0.5 * GRAVITY_IN_S2 * t * t;
    if (z < 0 && i > 0) break;
    points.push([x0 + vx * t, z, -(y0 + vy * t)]);
  }
  return points.length >= 2 ? points : [[x0, z0, -y0], [x0 + 8, z0, -y0]];
}

function partQuaternion(part: RobotPartTransform): [number, number, number, number] | undefined {
  if (!hasYupQuaternion(part)) return undefined;
  return [part.qx as number, part.qy as number, part.qz as number, part.qw as number];
}

function prepareCadScene(scene: Object3D) {
  scene.traverse((obj) => {
    if (!(obj instanceof Mesh) || !obj.material) return;
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    for (const mat of mats) {
      mat.side = DoubleSide;
      if ("metalness" in mat) mat.metalness = Math.min(Number(mat.metalness ?? 0), 0.12);
      if ("roughness" in mat) mat.roughness = Math.max(Number(mat.roughness ?? 0.7), 0.55);
      if ("color" in mat && mat.color && typeof mat.color.getHex === "function" && mat.color.getHex() === 0) {
        mat.color.set(theme.goldLight);
      }
    }
  });
}

function prepareFieldCadScene(scene: Object3D) {
  prepareCadScene(scene);
  scene.traverse((obj) => {
    if (!(obj instanceof Mesh) || !obj.material) return;
    const name = obj.name.toLowerCase();
    const source = Array.isArray(obj.material) ? obj.material : [obj.material];
    const mats = source.map((material) => material.clone());
    obj.material = Array.isArray(obj.material) ? mats : mats[0];
    for (const mat of mats) {
      if (!("color" in mat) || !mat.color) continue;
      if (name.includes("soft_tiles")) mat.color.set(theme.field);
      else if (name.includes("gaffer_tape_red")) mat.color.set(theme.maroon);
      else if (name.includes("gaffer_tape_blue")) mat.color.set(theme.allianceBlue);
      else if (name.includes("flower")) mat.color.set(theme.gold);
      else if (name.includes("blue_goal")) mat.color.set(theme.allianceBlueBright);
      else if (name.includes("red_goal")) mat.color.set(theme.maroon);
      else if (name.includes("field_side_glass")) {
        mat.color.set("#b9d7df");
        mat.transparent = true;
        mat.opacity = 0.2;
        mat.side = FrontSide;
        mat.forceSinglePass = true;
        mat.depthWrite = true;
        mat.polygonOffset = true;
        mat.polygonOffsetFactor = -1;
        mat.polygonOffsetUnits = -1;
      } else mat.color.set("#a9adb0");
    }
  });
}

function isFlatOverlay(el: El) {
  const tags = el.tags || [];
  const type = el.type || "";
  if (type === "tape" || type === "zone" || type === "spike_mark" || type === "net_zone" || type === "observation_zone") {
    return true;
  }
  return tags.some((t) => t.includes("restricted") || t.includes("start"));
}

function overlayColor(el: El) {
  const tags = el.tags || [];
  if (tags.some((t) => t.includes("restricted"))) return theme.restricted;
  if (el.type === "tape" || tags.some((t) => t.includes("launch"))) return theme.gold;
  if (tags.some((t) => t.includes("start"))) return el.alliance === "blue" ? theme.allianceBlue : theme.maroon;
  return el.alliance === "blue" ? theme.allianceBlue : theme.maroon;
}

function overlayKey(el: El) {
  const p = el.pose || { x: 0, y: 0 };
  return `${el.id}|${p.x}|${p.y}`;
}

function dedupe(els: El[]) {
  const seen = new Set<string>();
  const out: El[] = [];
  for (const el of els) {
    const k = overlayKey(el);
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(el);
  }
  return out;
}

function hideSchematicSolid(el: El, hasCad: boolean) {
  if (!hasCad) return false;
  const tags = el.tags || [];
  const type = el.type || "";
  if (type === "wall" || type === "hive_frame" || type === "goal" || type === "cell" || type === "flower") return true;
  return tags.some((t) => t === "hive" || t === "frame" || t === "cell" || t === "flower" || t === "perimeter");
}

function CameraRig({ view, w, d }: { view: SceneView; w: number; d: number }) {
  const { camera } = useThree();
  useEffect(() => {
    const span = Math.max(w, d, 72);
    if (view === "top") {
      camera.position.set(0, span * 1.65, 0.01);
    } else {
      camera.position.set(0, span * 1.45, span * 0.95);
    }
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [view, w, d, camera]);
  return null;
}

function ZoneOverlay({ el, fieldArea, active }: { el: El; fieldArea: number; active?: boolean }) {
  const pose = el.pose || { x: 0, y: 0 };
  const sh = el.shape || { kind: "aabb" };
  const width = sh.width || sh.radius || 24;
  const depth = sh.depth || sh.radius || 4;
  const large = width * depth > 0.15 * fieldArea;
  const color = overlayColor(el);
  const y = el.type === "tape" ? 0.12 : large ? 0.04 : 0.08;
  const hw = width / 2;
  const hd = depth / 2;
  const x = inch(pose.x);
  const z = inch(-pose.y);
  const loop: [number, number, number][] = [
    [x - hw, y, z - hd],
    [x + hw, y, z - hd],
    [x + hw, y, z + hd],
    [x - hw, y, z + hd],
    [x - hw, y, z - hd],
  ];
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[x, y, z]}>
        <planeGeometry args={[width, depth]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={el.type === "tape" ? 0.85 : large ? (active ? 0.22 : 0.06) : active ? 0.45 : 0.22}
          depthWrite={false}
        />
      </mesh>
      {large && <Line points={loop} color={color} lineWidth={active ? 2.5 : 1.5} />}
    </group>
  );
}

class CadErrorBoundary extends Component<{ onError?: () => void; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch() {
    this.props.onError?.();
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

class SceneErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div
        role="status"
        style={{
          width: "100%",
          height: "100%",
          display: "grid",
          placeItems: "center",
          padding: 24,
          color: theme.muted,
          background: theme.scene,
          textAlign: "center",
        }}
      >
        3D preview unavailable. Enable WebGL to view the field.
      </div>
    );
  }
}

function CadBackground({ url, onReady }: { url: string; onReady?: () => void }) {
  const gltf = useGLTF(url);
  const scene = useMemo(() => {
    const cloned = gltf.scene.clone(true);
    prepareFieldCadScene(cloned);
    return cloned;
  }, [gltf.scene]);
  useEffect(() => {
    onReady?.();
    // Load completion is tied to the cloned scene, not the parent callback identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene]);
  return <primitive object={scene} />;
}

function FieldMechanismActor({
  visualAsset,
  pivot,
  axis,
  angle,
  cacheToken,
}: {
  visualAsset: string;
  pivot: [number, number, number];
  axis: [number, number, number];
  angle: number;
  cacheToken?: string | null;
}) {
  const url = fieldAssetUrl(visualAsset, cacheToken);
  const length = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  const halfSin = Math.sin(angle / 2);
  const quaternion: [number, number, number, number] = [
    (axis[0] / length) * halfSin,
    (axis[1] / length) * halfSin,
    (axis[2] / length) * halfSin,
    Math.cos(angle / 2),
  ];
  return (
    <group position={pivot} quaternion={quaternion}>
      <Suspense fallback={null}>
        <CadBackground url={url} />
      </Suspense>
    </group>
  );
}

function RobotCad({ url, onReady }: { url: string; onReady?: () => void }) {
  const gltf = useGLTF(url);
  const scene = useMemo(() => {
    const cloned = gltf.scene.clone(true);
    prepareCadScene(cloned);
    return cloned;
  }, [gltf.scene]);
  useEffect(() => {
    onReady?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene]);
  return <primitive object={scene} />;
}

function pieceColor(color?: string) {
  if (color === "Y" || color === "yellow") return "#e2c44a";
  if (color === "R" || color === "red") return "#c4453c";
  if (color === "B" || color === "blue") return theme.allianceBlue;
  if (color === "G") return "#3dbf6a";
  return "#7a4ad6";
}

function IntakeGizmo({ intake, chassisHeight }: { intake: IntakeSpec; chassisHeight: number }) {
  const pose = intake.poseOnRobot || {};
  const heading = ((pose.headingDeg || 0) * Math.PI) / 180;
  const reach = intake.reachIn || 5;
  const width = intake.widthIn || 12;
  const height = intake.heightIn || 4;
  const z = (pose.z ?? 2) - chassisHeight / 2;
  return (
    <group position={[pose.x || 0, z, -(pose.y || 0)]} rotation={[0, heading, 0]}>
      <mesh position={[reach / 2, 0, 0]}>
        <boxGeometry args={[reach, height, width]} />
        <meshStandardMaterial color={theme.intake} transparent opacity={0.45} />
      </mesh>
    </group>
  );
}

function LauncherGizmo({ launcher, chassisHeight }: { launcher: LauncherSpec; chassisHeight: number }) {
  const pose = launcher.poseOnRobot || {};
  const heading = ((pose.headingDeg || 0) * Math.PI) / 180;
  const pitch = ((pose.pitchDeg || 0) * Math.PI) / 180;
  const speed = launcher.muzzleSpeedInPerS || 180;
  const len = Math.min(28, 8 + speed / 20);
  const z = (pose.z ?? 12) - chassisHeight / 2;
  const points: [number, number, number][] = [
    [0, 0, 0],
    [len * Math.cos(pitch), len * Math.sin(pitch), 0],
  ];
  return (
    <group position={[pose.x || 0, z, -(pose.y || 0)]} rotation={[0, heading, 0]}>
      <mesh>
        <boxGeometry args={[2.2, 2.2, 2.2]} />
        <meshStandardMaterial color={theme.gold} />
      </mesh>
      <Line points={points} color={theme.goldBright} lineWidth={2} />
    </group>
  );
}

function CollisionPrimitive({ part }: { part: RigidPartSpec }) {
  const pose = part.pose || {};
  const collision = part.collision[0];
  if (!collision) return null;
  const roll = ((collision.pose?.rollDeg || pose.rollDeg || 0) * Math.PI) / 180;
  const pitch = ((collision.pose?.pitchDeg || pose.pitchDeg || 0) * Math.PI) / 180;
  const yaw = ((collision.pose?.yawDeg || pose.yawDeg || 0) * Math.PI) / 180;
  const color = part.id === "chassis" ? theme.chassis : part.id.includes("fly") ? theme.gold : theme.intake;
  if (collision.kind === "box") {
    return (
      <mesh rotation={[pitch, yaw, roll]}>
        <boxGeometry args={collision.sizeIn} />
        <meshStandardMaterial color={color} transparent opacity={0.72} />
      </mesh>
    );
  }
  if (collision.kind === "sphere") {
    return (
      <mesh>
        <sphereGeometry args={[collision.radiusIn, 16, 16]} />
        <meshStandardMaterial color={color} transparent opacity={0.72} />
      </mesh>
    );
  }
  if (collision.kind === "convex_mesh") {
    return (
      <mesh>
        <boxGeometry args={[4, 2, 4]} />
        <meshStandardMaterial color={color} transparent opacity={0.55} />
      </mesh>
    );
  }
  const length = collision.lengthIn || collision.radiusIn * 2;
  return (
    <mesh rotation={[pitch + (collision.kind === "cylinder" ? Math.PI / 2 : 0), yaw, roll]}>
      <cylinderGeometry args={[collision.radiusIn, collision.radiusIn, length, 16]} />
      <meshStandardMaterial color={color} transparent opacity={0.72} />
    </mesh>
  );
}

function RobotPartActor({
  part,
  design,
}: {
  part: RobotPartTransform;
  design?: RobotDesign;
}) {
  const spec = (design?.rigidParts || []).find((row) => row.id === part.id);
  const visualAsset = part.visualAsset || spec?.visualAsset;
  const cadUrl = visualAsset ? `${API}/robot-assets/${visualAsset}` : null;
  const [cadReady, setCadReady] = useState(false);
  useEffect(() => {
    setCadReady(false);
    if (cadUrl) useGLTF.preload(cadUrl);
  }, [cadUrl]);
  const orient = partQuaternion(part);
  return (
    <group position={ftcToThreePosition(part.x, part.y, part.z)} quaternion={orient}>
      {cadUrl && (
        <CadErrorBoundary onError={() => setCadReady(false)}>
          <Suspense fallback={null}>
            <RobotCad url={cadUrl} onReady={() => setCadReady(true)} />
          </Suspense>
        </CadErrorBoundary>
      )}
      {(!cadUrl || !cadReady) && spec && <CollisionPrimitive part={spec} />}
      {(!cadUrl || !cadReady) && !spec && (
        <mesh>
          <boxGeometry args={[3, 2, 3]} />
          <meshStandardMaterial color={theme.chassis} transparent opacity={0.6} />
        </mesh>
      )}
    </group>
  );
}

function RobotTelemetry({ robot, chassisHeight }: { robot: Pick<FrameRobot, "actuators" | "batteryVoltageV" | "batteryCurrentA">; chassisHeight: number }) {
  const flywheel = robot.actuators?.flywheel || Object.values(robot.actuators || {})[0];
  if (robot.batteryVoltageV == null && !flywheel) return null;
  const rpm = flywheel?.rpm;
  const current = flywheel?.currentA;
  const bits = [
    robot.batteryVoltageV != null ? `${robot.batteryVoltageV.toFixed(1)} V` : null,
    robot.batteryCurrentA != null ? `${robot.batteryCurrentA.toFixed(1)} A` : null,
    rpm != null ? `${rpm.toFixed(0)} rpm` : null,
    current != null ? `${current.toFixed(1)} A motor` : null,
  ].filter((row): row is string => Boolean(row));
  if (bits.length === 0) return null;
  return (
    <Html position={[0, chassisHeight / 2 + 3, 0]} center style={{ pointerEvents: "none", color: theme.cream, fontSize: "11px", whiteSpace: "nowrap" }}>
      {bits.join(" · ")}
    </Html>
  );
}

function LaunchPreviewArc({
  path,
  flywheelFrac,
  hoodFrac,
  actuators,
}: {
  path: PiecePathSpec;
  flywheelFrac: number;
  hoodFrac: number;
  actuators?: ActuatorSpec[];
}) {
  const points = useMemo(
    () => launchArcPoints(path, flywheelFrac, hoodFrac, actuators),
    [path, flywheelFrac, hoodFrac, actuators],
  );
  return <Line points={points} color={theme.goldBright} lineWidth={2} />;
}

function StorageSlotMarkers({ path }: { path: PiecePathSpec }) {
  return (
    <>
      {(path.storageSlots || []).map((slot, index) => (
        <mesh key={`slot-${index}`} position={[slot.x || 0, slot.z || 4, -(slot.y || 0)]}>
          <sphereGeometry args={[1.2, 12, 12]} />
          <meshStandardMaterial color={theme.gold} transparent opacity={0.35} />
        </mesh>
      ))}
    </>
  );
}

export function RobotActor({
  x = 0,
  y = 0,
  headingDeg = 0,
  dynamic = true,
  design,
  showFov = false,
  showHull = false,
  highlight = null,
  hideBody = false,
  telemetry,
  launchPreview,
}: {
  x?: number;
  y?: number;
  headingDeg?: number;
  dynamic?: boolean;
  design?: RobotDesign;
  showFov?: boolean;
  showHull?: boolean;
  highlight?: "foul" | "contact" | null;
  hideBody?: boolean;
  telemetry?: Pick<FrameRobot, "actuators" | "batteryVoltageV" | "batteryCurrentA">;
  launchPreview?: { flywheelFrac: number; hoodFrac: number };
}) {
  const length = design?.chassis?.lengthIn ?? 18;
  const width = design?.chassis?.widthIn ?? 18;
  const height = design?.chassis?.heightIn ?? 10;
  const visualAsset = design?.visualAsset;
  const offset = design?.visualOffset || {};
  const cadUrl = visualAsset ? `${API}/robot-assets/${visualAsset}` : null;
  const [cadReady, setCadReady] = useState(false);
  useEffect(() => {
    setCadReady(false);
    if (cadUrl) useGLTF.preload(cadUrl);
  }, [cadUrl]);
  const showBox = !hideBody && (!cadUrl || !cadReady || showHull || Boolean(highlight));
  const chassisColor = highlight ? theme.restricted : dynamic ? theme.chassis : theme.chassisIdle;
  const emissiveIntensity = highlight === "foul" ? 0.55 : highlight === "contact" ? 0.35 : 0;
  const previewParts = hideBody ? [] : design?.rigidParts || [];
  return (
    <group position={[inch(x), height / 2, inch(-y)]} rotation={[0, (headingDeg * Math.PI) / 180, 0]}>
      {cadUrl && !hideBody && (
        <group position={[offset.x || 0, (offset.z || 0) - height / 2, -(offset.y || 0)]} rotation={[0, (visualOffsetYawDeg(offset) * Math.PI) / 180, 0]}>
          <CadErrorBoundary onError={() => setCadReady(false)}>
            <Suspense fallback={null}>
              <RobotCad url={cadUrl} onReady={() => setCadReady(true)} />
            </Suspense>
          </CadErrorBoundary>
        </group>
      )}
      {showBox && (
        <>
          <mesh>
            <boxGeometry args={[length, height, width]} />
            <meshStandardMaterial
              color={chassisColor}
              transparent={Boolean(cadUrl) || Boolean(highlight) || previewParts.length > 0}
              opacity={highlight ? (cadUrl ? 0.45 : 0.92) : cadUrl || previewParts.length > 0 ? 0.28 : 1}
              emissive={highlight ? theme.restricted : "#000000"}
              emissiveIntensity={emissiveIntensity}
            />
          </mesh>
          <mesh position={[length / 2 - 1.5, 1, 0]}>
            <boxGeometry args={[3, 2.5, Math.min(6, width * 0.4)]} />
            <meshStandardMaterial color="#222" />
          </mesh>
        </>
      )}
      {!hideBody &&
        previewParts.map((part) => (
          <group key={part.id} position={[part.pose?.x || 0, (part.pose?.z || 0) - height / 2, -(part.pose?.y || 0)]}>
            <CollisionPrimitive part={part} />
          </group>
        ))}
      {(design?.intakes || []).map((intake) => (
        <IntakeGizmo key={intake.id} intake={intake} chassisHeight={height} />
      ))}
      {(design?.launchers || []).map((launcher) => (
        <LauncherGizmo key={launcher.id} launcher={launcher} chassisHeight={height} />
      ))}
      {design?.piecePath && launchPreview && (
        <>
          <StorageSlotMarkers path={design.piecePath} />
          <LaunchPreviewArc
            path={design.piecePath}
            flywheelFrac={launchPreview.flywheelFrac}
            hoodFrac={launchPreview.hoodFrac}
            actuators={design.actuators}
          />
        </>
      )}
      {telemetry && <RobotTelemetry robot={telemetry} chassisHeight={height} />}
      {showFov && dynamic && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[20, 0.2, 0]}>
          <circleGeometry args={[40, 24, -0.6, 1.2]} />
          <meshStandardMaterial color={theme.gold} transparent opacity={0.18} />
        </mesh>
      )}
    </group>
  );
}

export function RobotPreview({
  design,
  showFov = false,
  showHull = false,
  launchPreview,
}: {
  design: RobotDesign;
  showFov?: boolean;
  showHull?: boolean;
  launchPreview?: { flywheelFrac: number; hoodFrac: number };
}) {
  const span = Math.max(design.chassis?.lengthIn || 18, design.chassis?.widthIn || 18, 24);
  return (
    <Canvas camera={{ position: [span * 0.9, span * 1.1, span * 0.9], fov: 40 }} style={{ width: "100%", height: "100%" }}>
      <color attach="background" args={[theme.scene]} />
      <ambientLight intensity={0.7} />
      <directionalLight position={[40, 80, 30]} intensity={1} />
      <Grid args={[span * 2, span * 2]} cellSize={6} sectionSize={18} cellColor={theme.grid} sectionColor={theme.gridSection} fadeDistance={120} />
      <RobotActor design={design} showFov={showFov} showHull={showHull} launchPreview={launchPreview} />
      <OrbitControls makeDefault />
    </Canvas>
  );
}

function useCadRuntime(frame: Frame, cadFallback?: string | null) {
  const cadPath = resolveBackgroundAsset(frame, cadFallback);
  const manifestPath = resolveCadManifestPath(frame);
  const explicitToken = explicitCadCacheToken(frame);
  const [manifest, setManifest] = useState<CadManifest | null>(null);
  const [manifestSettled, setManifestSettled] = useState(!manifestPath);
  useEffect(() => {
    if (!manifestPath) {
      setManifest(null);
      setManifestSettled(true);
      return;
    }
    let cancelled = false;
    setManifestSettled(Boolean(explicitToken));
    void fetchCadManifest(manifestPath).then((doc) => {
      if (cancelled) return;
      setManifest(doc);
      setManifestSettled(true);
    });
    return () => {
      cancelled = true;
    };
  }, [manifestPath, explicitToken]);
  const token = explicitToken || cadCacheTokenFromManifest(manifest);
  const catalog = useMemo(() => buildPieceCatalog(frame, manifest), [frame, manifest]);
  const cadUrl = cadPath && (token || manifestSettled) ? fieldAssetUrl(cadPath, token) : null;
  return { cadPath, cadUrl, token, catalog, manifest };
}

function PieceCad({ url, color, onReady }: { url: string; color: string; onReady?: () => void }) {
  const gltf = useGLTF(url);
  const scene = useMemo(() => {
    const cloned = gltf.scene.clone(true);
    prepareCadScene(cloned);
    cloned.traverse((obj) => {
      if (!(obj instanceof Mesh) || !obj.material) return;
      const source = Array.isArray(obj.material) ? obj.material : [obj.material];
      const mats = source.map((material) => material.clone());
      obj.material = Array.isArray(obj.material) ? mats : mats[0];
      for (const mat of mats) {
        if ("color" in mat && mat.color) mat.color.set(color);
      }
    });
    return cloned;
  }, [gltf.scene, color]);
  useEffect(() => {
    onReady?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene]);
  return <primitive object={scene} />;
}

function PieceActor({
  piece,
  catalog,
  backgroundAsset,
  cacheToken,
}: {
  piece: FramePiece;
  catalog: PieceCatalog;
  backgroundAsset?: string | null;
  cacheToken?: string | null;
}) {
  const visualAsset = resolvePieceVisualAsset(piece, catalog, backgroundAsset);
  const pose = pieceThreePose(piece, catalog);
  const cadUrl = visualAsset ? pieceAssetUrl(visualAsset, catalog, piece.typeId, cacheToken) : null;
  const color = pieceColor(piece.color);
  const [cadReady, setCadReady] = useState(false);
  useEffect(() => {
    setCadReady(false);
    if (cadUrl) useGLTF.preload(cadUrl);
  }, [cadUrl]);
  const orient = pose.quaternion ? { quaternion: pose.quaternion } : { rotation: pose.euler };
  return (
    <group position={pose.position} {...orient}>
      {cadUrl && (
        <CadErrorBoundary onError={() => setCadReady(false)}>
          <Suspense fallback={null}>
            <PieceCad url={cadUrl} color={color} onReady={() => setCadReady(true)} />
          </Suspense>
        </CadErrorBoundary>
      )}
      {(!cadUrl || !cadReady) && (
        <mesh>
          <sphereGeometry args={[pose.radius, 16, 16]} />
          <meshStandardMaterial color={color} />
        </mesh>
      )}
    </group>
  );
}

function FieldMeshes({
  frame,
  showFov,
  path,
  cadFallback,
}: {
  frame: Frame;
  showFov: boolean;
  path: { x: number; y: number }[];
  cadFallback?: string | null;
}) {
  const w = frame.fieldSizeIn.width;
  const d = frame.fieldSizeIn.depth;
  const fieldArea = w * d;
  const elements = dedupe(frame.elements || []);
  const { cadUrl, token, catalog, cadPath, manifest } = useCadRuntime(frame, cadFallback);
  const [cadReady, setCadReady] = useState(false);
  const enteredRestricted =
    (frame.robots || []).some((r) => r.enteredRestricted) || Boolean(frame.collision?.enteredRestricted);
  const foulStep = (frame.stepExplains || []).some((e) => e.points < 0);
  const contact = Boolean(frame.collision?.wall || frame.collision?.robot);
  const physicalStorage = usesPhysicalStorage(frame);
  useEffect(() => {
    setCadReady(false);
    if (cadUrl) useGLTF.preload(cadUrl);
  }, [cadUrl]);
  return (
    <group>
      {!cadReady && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.2, 0]}>
          <planeGeometry args={[w, d]} />
          <meshStandardMaterial color={theme.field} />
        </mesh>
      )}
      {cadUrl && (
        <CadErrorBoundary onError={() => setCadReady(false)}>
          <Suspense fallback={null}>
            <CadBackground url={cadUrl} onReady={() => setCadReady(true)} />
          </Suspense>
        </CadErrorBoundary>
      )}
      {(manifest?.field?.mechanisms || []).map((mechanism) => {
        const angle = frame.fieldMechanisms?.find((state) => state.id === mechanism.id)?.angleRad || 0;
        return (
          <CadErrorBoundary key={mechanism.id}>
            <FieldMechanismActor
              visualAsset={mechanism.visualAsset}
              pivot={mechanism.pivotIn}
              axis={mechanism.axis}
              angle={angle}
              cacheToken={token}
            />
          </CadErrorBoundary>
        );
      })}
      {!cadReady &&
        (
          [
            [0, d / 2, w + 2, 2],
            [0, -d / 2, w + 2, 2],
            [-w / 2, 0, 2, d],
            [w / 2, 0, 2, d],
          ] as [number, number, number, number][]
        ).map(([x, y, bw, bd], i) => (
          <mesh key={`wall-${i}`} position={[x, 6, -y]}>
            <boxGeometry args={[bw, 12, bd]} />
            <meshStandardMaterial color={theme.muted} />
          </mesh>
        ))}
      {elements.map((el, idx) => {
        const pose = el.pose || { x: 0, y: 0 };
        const sh = el.shape || { kind: "aabb" };
        if (el.type === "wall") return null;
        if (hideSchematicSolid(el, cadReady)) return null;
        if (isFlatOverlay(el)) {
          const restricted = (el.tags || []).some((t) => t.includes("restricted"));
          return (
            <ZoneOverlay
              key={`${overlayKey(el)}-${idx}`}
              el={el}
              fieldArea={fieldArea}
              active={restricted && enteredRestricted}
            />
          );
        }
        if (sh.kind === "circle") {
          return (
            <mesh key={`${el.id}-${idx}`} position={[inch(pose.x), 2, inch(-pose.y)]}>
              <cylinderGeometry args={[sh.radius || 4, sh.radius || 4, 4, 20]} />
              <meshStandardMaterial color={theme.allianceBlue} transparent opacity={0.5} />
            </mesh>
          );
        }
        const width = sh.width || 8;
        const depth = sh.depth || 8;
        const h = el.type === "goal" ? 16 : 8;
        const color = (el.tags || []).includes("gate")
          ? theme.gold
          : el.alliance === "blue"
            ? theme.allianceBlueBright
            : el.isOccluder
              ? theme.occluder
              : theme.maroon;
        return (
          <mesh key={`${el.id}-${idx}`} position={[inch(pose.x), h / 2, inch(-pose.y)]}>
            <boxGeometry args={[width, h, depth]} />
            <meshStandardMaterial color={color} transparent opacity={0.9} />
          </mesh>
        );
      })}
      {(frame.pieces || [])
        .filter((p) => !p.scored && (physicalStorage || !p.heldBy))
        .map((p) => (
          <PieceActor
            key={p.id}
            piece={p}
            catalog={catalog}
            backgroundAsset={cadPath}
            cacheToken={token}
          />
        ))}
      {!physicalStorage &&
        (frame.pieces || [])
          .filter((piece) => !piece.scored && Boolean(piece.heldBy))
          .map((piece) => {
            const robot = (frame.robots || []).find((candidate) => candidate.id === piece.heldBy);
            if (!robot) return null;
            const held = (frame.pieces || []).filter(
              (candidate) => candidate.heldBy === piece.heldBy && !candidate.scored,
            );
            const index = held.findIndex((candidate) => candidate.id === piece.id);
            const radius = pieceThreePose(piece, catalog).radius;
            const lateral = (index - (held.length - 1) / 2) * radius * 2.1;
            const heading = (robot.headingDeg * Math.PI) / 180;
            const stagedPiece: FramePiece = {
              ...piece,
              x: robot.x - Math.sin(heading) * lateral,
              y: robot.y + Math.cos(heading) * lateral,
              z: (frame.robotDesign?.chassis?.heightIn ?? 10) + radius,
              heldBy: undefined,
            };
            return (
              <PieceActor
                key={`held-${piece.id}`}
                piece={stagedPiece}
                catalog={catalog}
                backgroundAsset={cadPath}
                cacheToken={token}
              />
            );
          })}
      {(frame.robots || []).map((r) => {
        const parts = r.parts || [];
        return (
          <group key={r.id}>
            {parts.map((part) => (
              <RobotPartActor key={`${r.id}-${part.id}`} part={part} design={frame.robotDesign} />
            ))}
            <RobotActor
              x={r.x}
              y={r.y}
              headingDeg={r.headingDeg}
              dynamic={r.dynamic}
              design={frame.robotDesign}
              hideBody={parts.length > 0}
              showHull={frame.robotDesign?.collisionShape === "mesh" || frame.robotDesign?.chassis?.collisionShape === "mesh"}
              showFov={showFov}
              highlight={r.dynamic ? (foulStep || r.enteredRestricted ? "foul" : contact ? "contact" : null) : null}
              telemetry={r}
            />
          </group>
        );
      })}
      {path.length > 1 && (
        <Line
          points={path.map((p) => [inch(p.x), 0.5, inch(-p.y)] as [number, number, number])}
          color={theme.gold}
          lineWidth={2}
        />
      )}
    </group>
  );
}

export function FieldScene({
  frame,
  showFov,
  path = [],
  view = "threeQuarter",
  cadAsset,
}: {
  frame: Frame | null;
  showFov: boolean;
  path?: { x: number; y: number }[];
  view?: SceneView;
  cadAsset?: string | null;
}) {
  const w = frame?.fieldSizeIn.width || 144;
  const d = frame?.fieldSizeIn.depth || 144;
  const span = Math.max(w, d);
  return (
    <SceneErrorBoundary>
      <Canvas camera={{ position: [0, span * 1.45, span * 0.95], fov: 40 }} style={{ width: "100%", height: "100%" }}>
        <color attach="background" args={[theme.scene]} />
        <ambientLight intensity={0.6} />
        <directionalLight position={[80, 200, 60]} intensity={1.1} />
        <Grid
          args={[w, d]}
          position={[0, -0.15, 0]}
          cellSize={24}
          sectionSize={72}
          cellColor={theme.grid}
          sectionColor={theme.gridSection}
          fadeDistance={400}
        />
        <CameraRig view={view} w={w} d={d} />
        {frame && <FieldMeshes frame={frame} showFov={showFov} path={path} cadFallback={cadAsset} />}
        <OrbitControls makeDefault />
      </Canvas>
    </SceneErrorBoundary>
  );
}
