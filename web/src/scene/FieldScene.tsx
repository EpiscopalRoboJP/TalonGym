import { useEffect } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Grid, Line } from "@react-three/drei";
import type { Frame } from "../api";

export type SceneView = "threeQuarter" | "top";

type El = Frame["elements"][number];

function inch(n: number) {
  return n;
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
  if (tags.some((t) => t.includes("restricted"))) return "#d16b6b";
  if (el.type === "tape" || tags.some((t) => t.includes("launch"))) return "#e0c36a";
  if (tags.some((t) => t.includes("start"))) return el.alliance === "blue" ? "#3d6b8a" : "#2f6f55";
  return el.alliance === "blue" ? "#3d6b8a" : "#2f6f55";
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

function ZoneOverlay({ el, fieldArea }: { el: El; fieldArea: number }) {
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
          opacity={el.type === "tape" ? 0.85 : large ? 0.06 : 0.22}
          depthWrite={false}
        />
      </mesh>
      {large && <Line points={loop} color={color} lineWidth={1.5} />}
    </group>
  );
}

function FieldMeshes({
  frame,
  showFov,
  path,
}: {
  frame: Frame;
  showFov: boolean;
  path: { x: number; y: number }[];
}) {
  const w = frame.fieldSizeIn.width;
  const d = frame.fieldSizeIn.depth;
  const fieldArea = w * d;
  const elements = dedupe(frame.elements || []);
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.2, 0]}>
        <planeGeometry args={[w, d]} />
        <meshStandardMaterial color="#1c4a38" />
      </mesh>
      {(
        [
          [0, d / 2, w + 2, 2],
          [0, -d / 2, w + 2, 2],
          [-w / 2, 0, 2, d],
          [w / 2, 0, 2, d],
        ] as [number, number, number, number][]
      ).map(([x, y, bw, bd], i) => (
        <mesh key={`wall-${i}`} position={[x, 6, -y]}>
          <boxGeometry args={[bw, 12, bd]} />
          <meshStandardMaterial color="#8aa0ae" />
        </mesh>
      ))}
      {elements.map((el, idx) => {
        const pose = el.pose || { x: 0, y: 0 };
        const sh = el.shape || { kind: "aabb" };
        if (el.type === "wall") return null;
        if (isFlatOverlay(el)) {
          return <ZoneOverlay key={`${overlayKey(el)}-${idx}`} el={el} fieldArea={fieldArea} />;
        }
        if (sh.kind === "circle") {
          return (
            <mesh key={`${el.id}-${idx}`} position={[inch(pose.x), 2, inch(-pose.y)]}>
              <cylinderGeometry args={[sh.radius || 4, sh.radius || 4, 4, 20]} />
              <meshStandardMaterial color="#3d6b8a" transparent opacity={0.5} />
            </mesh>
          );
        }
        const width = sh.width || 8;
        const depth = sh.depth || 8;
        const h = el.type === "goal" ? 16 : 8;
        const color = (el.tags || []).includes("gate")
          ? "#d4a574"
          : el.alliance === "blue"
            ? "#4a7aa8"
            : el.isOccluder
              ? "#3d5a4c"
              : "#2f6f55";
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
          <mesh key={p.id} position={[inch(p.x), 2.6, inch(-p.y)]}>
            <sphereGeometry args={[2.5, 16, 16]} />
            <meshStandardMaterial color={p.color === "G" ? "#3dbf6a" : "#7a4ad6"} />
          </mesh>
        ))}
      {(frame.robots || []).map((r) => (
        <group key={r.id} position={[inch(r.x), 5, inch(-r.y)]} rotation={[0, (r.headingDeg * Math.PI) / 180, 0]}>
          <mesh>
            <boxGeometry args={[18, 10, 18]} />
            <meshStandardMaterial color={r.dynamic ? "#e8c9a3" : "#7d8b94"} />
          </mesh>
          <mesh position={[8, 2, 0]}>
            <boxGeometry args={[4, 3, 6]} />
            <meshStandardMaterial color="#222" />
          </mesh>
          {showFov && r.dynamic && (
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[20, 0.2, 0]}>
              <circleGeometry args={[40, 24, -0.6, 1.2]} />
              <meshStandardMaterial color="#d4a574" transparent opacity={0.18} />
            </mesh>
          )}
        </group>
      ))}
      {path.length > 1 && (
        <Line
          points={path.map((p) => [inch(p.x), 0.5, inch(-p.y)] as [number, number, number])}
          color="#d4a574"
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
}: {
  frame: Frame | null;
  showFov: boolean;
  path?: { x: number; y: number }[];
  view?: SceneView;
}) {
  const w = frame?.fieldSizeIn.width || 144;
  const d = frame?.fieldSizeIn.depth || 144;
  const span = Math.max(w, d);
  return (
    <Canvas camera={{ position: [0, span * 1.45, span * 0.95], fov: 40 }} style={{ width: "100%", height: "100%" }}>
      <color attach="background" args={["#070b0e"]} />
      <ambientLight intensity={0.6} />
      <directionalLight position={[80, 200, 60]} intensity={1.1} />
      <Grid args={[w, d]} cellSize={24} sectionSize={72} cellColor="#2a4a3c" sectionColor="#3d6a52" fadeDistance={400} />
      <CameraRig view={view} w={w} d={d} />
      {frame && <FieldMeshes frame={frame} showFov={showFov} path={path} />}
      <OrbitControls makeDefault />
    </Canvas>
  );
}
