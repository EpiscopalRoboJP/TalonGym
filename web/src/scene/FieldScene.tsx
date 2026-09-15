import { Component, Suspense, useEffect, useMemo, useState, type ReactNode } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Grid, Line, useGLTF } from "@react-three/drei";
import { DoubleSide, Mesh, Object3D } from "three";
import type { Frame, IntakeSpec, LauncherSpec, RobotDesign } from "../api";
import { API } from "../api";
import { theme } from "../theme";

const BIOBUZZ_GLB = "seasons/biobuzz_2026/field.glb";
const CAD_ASSET_VERSION = "am-5850e";

export function resolveBackgroundAsset(frame: Frame | null | undefined, fallback?: string | null) {
  if (frame?.backgroundAsset) return frame.backgroundAsset;
  if (fallback) return fallback;
  const els = frame?.elements || [];
  if (els.some((el) => el.id === "red_cell_up" || el.type === "hive_frame" || (el.tags || []).includes("hive"))) {
    return BIOBUZZ_GLB;
  }
  return null;
}

export type SceneView = "threeQuarter" | "top";

type El = Frame["elements"][number];

function inch(n: number) {
  return n;
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
  if (tags.some((t) => t.includes("start"))) return el.alliance === "blue" ? theme.allianceBlue : theme.allianceRed;
  return el.alliance === "blue" ? theme.allianceBlue : theme.allianceRed;
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

function CadBackground({ url, onReady }: { url: string; onReady?: () => void }) {
  const gltf = useGLTF(url);
  const scene = useMemo(() => {
    const cloned = gltf.scene.clone(true);
    prepareCadScene(cloned);
    return cloned;
  }, [gltf.scene]);
  useEffect(() => {
    onReady?.();
    // Load completion is tied to the cloned scene, not the parent callback identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene]);
  return <primitive object={scene} />;
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

export function RobotActor({
  x = 0,
  y = 0,
  headingDeg = 0,
  dynamic = true,
  design,
  showFov = false,
  showHull = false,
  highlight = null,
}: {
  x?: number;
  y?: number;
  headingDeg?: number;
  dynamic?: boolean;
  design?: RobotDesign;
  showFov?: boolean;
  showHull?: boolean;
  highlight?: "foul" | "contact" | null;
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
  const showBox = !cadUrl || !cadReady || showHull || Boolean(highlight);
  const chassisColor = highlight ? theme.restricted : dynamic ? theme.chassis : theme.chassisIdle;
  const emissiveIntensity = highlight === "foul" ? 0.55 : highlight === "contact" ? 0.35 : 0;
  return (
    <group position={[inch(x), height / 2, inch(-y)]} rotation={[0, (headingDeg * Math.PI) / 180, 0]}>
      {cadUrl && (
        <group position={[offset.x || 0, (offset.z || 0) - height / 2, -(offset.y || 0)]} rotation={[0, ((offset.yawDeg || 0) * Math.PI) / 180, 0]}>
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
              transparent={Boolean(cadUrl) || Boolean(highlight)}
              opacity={highlight ? (cadUrl ? 0.45 : 0.92) : cadUrl ? 0.28 : 1}
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
      {(design?.intakes || []).map((intake) => (
        <IntakeGizmo key={intake.id} intake={intake} chassisHeight={height} />
      ))}
      {(design?.launchers || []).map((launcher) => (
        <LauncherGizmo key={launcher.id} launcher={launcher} chassisHeight={height} />
      ))}
      {showFov && dynamic && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[20, 0.2, 0]}>
          <circleGeometry args={[40, 24, -0.6, 1.2]} />
          <meshStandardMaterial color={theme.gold} transparent opacity={0.18} />
        </mesh>
      )}
    </group>
  );
}

export function RobotPreview({ design, showFov = false, showHull = false }: { design: RobotDesign; showFov?: boolean; showHull?: boolean }) {
  const span = Math.max(design.chassis?.lengthIn || 18, design.chassis?.widthIn || 18, 24);
  return (
    <Canvas camera={{ position: [span * 2.1, span * 1.8, span * 2.1], fov: 40 }} style={{ width: "100%", height: "100%" }}>
      <color attach="background" args={[theme.scene]} />
      <ambientLight intensity={0.7} />
      <directionalLight position={[40, 80, 30]} intensity={1} />
      <Grid args={[span * 2, span * 2]} cellSize={6} sectionSize={18} cellColor={theme.grid} sectionColor={theme.gridSection} fadeDistance={120} />
      <RobotActor design={design} showFov={showFov} showHull={showHull} />
      <OrbitControls makeDefault />
    </Canvas>
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
  const cadPath = resolveBackgroundAsset(frame, cadFallback);
  const cadUrl = cadPath ? `${API}/field-assets/${cadPath}?v=${CAD_ASSET_VERSION}` : null;
  const [cadReady, setCadReady] = useState(false);
  const enteredRestricted =
    (frame.robots || []).some((r) => r.enteredRestricted) || Boolean(frame.collision?.enteredRestricted);
  const foulStep = (frame.stepExplains || []).some((e) => e.points < 0);
  const contact = Boolean(frame.collision?.wall || frame.collision?.robot);
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
            <meshStandardMaterial color={theme.mapLabel} />
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
              : theme.allianceRed;
        return (
          <mesh key={`${el.id}-${idx}`} position={[inch(pose.x), h / 2, inch(-pose.y)]}>
            <boxGeometry args={[width, h, depth]} />
            <meshStandardMaterial color={color} transparent opacity={0.9} />
          </mesh>
        );
      })}
      {(frame.pieces || [])
        .filter((p) => !p.scored)
        .map((p) => (
          <mesh key={p.id} position={[inch(p.x), p.z ?? 2.6, inch(-p.y)]}>
            <sphereGeometry args={[2.5, 16, 16]} />
            <meshStandardMaterial color={pieceColor(p.color)} />
          </mesh>
        ))}
      {(frame.robots || []).map((r) => (
        <RobotActor
          key={r.id}
          x={r.x}
          y={r.y}
          headingDeg={r.headingDeg}
          dynamic={r.dynamic}
          design={frame.robotDesign}
          showHull={frame.robotDesign?.collisionShape === "mesh" || frame.robotDesign?.chassis?.collisionShape === "mesh"}
          showFov={showFov}
          highlight={r.dynamic ? (foulStep || r.enteredRestricted ? "foul" : contact ? "contact" : null) : null}
        />
      ))}
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
    <Canvas camera={{ position: [0, span * 1.45, span * 0.95], fov: 40 }} style={{ width: "100%", height: "100%" }}>
      <color attach="background" args={[theme.scene]} />
      <ambientLight intensity={0.6} />
      <directionalLight position={[80, 200, 60]} intensity={1.1} />
      <Grid args={[w, d]} cellSize={24} sectionSize={72} cellColor={theme.grid} sectionColor={theme.gridSection} fadeDistance={400} />
      <CameraRig view={view} w={w} d={d} />
      {frame && <FieldMeshes frame={frame} showFov={showFov} path={path} cadFallback={cadAsset} />}
      <OrbitControls makeDefault />
    </Canvas>
  );
}
