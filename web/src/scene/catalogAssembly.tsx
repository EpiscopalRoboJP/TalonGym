import { Component, Suspense, useEffect, useMemo, useState, type ReactNode } from "react";
import { Html, useGLTF } from "@react-three/drei";
import { Box3, BufferGeometry, DoubleSide, Group, Matrix4, Mesh, Object3D, Vector3 } from "three";
import type { CatalogPart, CollisionShapeSpec, RigidPartSpec, RobotDesign, RobotPartTransform, Transform3 } from "../api";
import { API } from "../api";
import { theme } from "../theme";
import { ftcToThreePosition, hasYupQuaternion } from "./cadAssets";
import { cadShouldShowMesh, type CadCollisionFit, type Vec3 } from "./cadFit";
import { cadPlaceholderLabel } from "../robotBuilder/cadVisual";
import { catalogPartToRigid, collisionSpan, ftcSizeToThree, proxyColor } from "../robotBuilder/partVisual";
import { poseEulerRad } from "../robotBuilder/transforms";

function cloneMaterials(obj: Mesh) {
  const source = Array.isArray(obj.material) ? obj.material : [obj.material];
  const mats = source.map((material) => material.clone());
  obj.material = Array.isArray(obj.material) ? mats : mats[0];
  return mats;
}

export function prepareCadScene(scene: Object3D) {
  scene.traverse((obj) => {
    if (!(obj instanceof Mesh) || !obj.material) return;
    const geo = obj.geometry as BufferGeometry | undefined;
    if (geo && !geo.getAttribute("normal")) geo.computeVertexNormals();
    const mats = cloneMaterials(obj);
    for (const mat of mats) {
      mat.side = DoubleSide;
      mat.transparent = false;
      mat.opacity = 1;
      mat.depthWrite = true;
      if ("transmission" in mat) {
        (mat as { transmission?: number }).transmission = 0;
      }
      if ("metalness" in mat) mat.metalness = Math.min(Number(mat.metalness ?? 0), 0.12);
      if ("roughness" in mat) mat.roughness = Math.max(Number(mat.roughness ?? 0.7), 0.55);
      if ("color" in mat && mat.color && typeof (mat.color as { getHex?: () => number }).getHex === "function") {
        const color = mat.color as { getHex: () => number; set: (value: string) => void };
        if (color.getHex() === 0) color.set(theme.goldLight);
      }
    }
  });
}

export function measureCadScene(scene: Object3D): { triangleCount: number; span: number; size: Vec3; center: Vec3 } {
  let triangles = 0;
  scene.traverse((obj) => {
    if (!(obj instanceof Mesh) || !obj.geometry) return;
    const geo = obj.geometry as BufferGeometry;
    const index = geo.index;
    const pos = geo.getAttribute("position");
    if (index) triangles += Math.floor(index.count / 3);
    else if (pos) triangles += Math.floor(pos.count / 3);
  });
  const box = new Box3().setFromObject(scene);
  const sizeVec = box.getSize(new Vector3());
  const centerVec = box.getCenter(new Vector3());
  const size: Vec3 = [sizeVec.x, sizeVec.y, sizeVec.z];
  const center: Vec3 = [centerVec.x, centerVec.y, centerVec.z];
  return { triangleCount: triangles, span: Math.max(size[0], size[1], size[2], 0), size, center };
}

function applyCadCollisionFit(scene: Object3D, fit: CadCollisionFit) {
  const wrapper = new Group();
  wrapper.add(scene);
  wrapper.scale.setScalar(fit.scale);
  const m = new Matrix4().set(
    fit.rotation[0],
    fit.rotation[1],
    fit.rotation[2],
    0,
    fit.rotation[3],
    fit.rotation[4],
    fit.rotation[5],
    0,
    fit.rotation[6],
    fit.rotation[7],
    fit.rotation[8],
    0,
    0,
    0,
    0,
    1,
  );
  wrapper.quaternion.setFromRotationMatrix(m);
  wrapper.position.set(fit.translation[0], fit.translation[1], fit.translation[2]);
  return wrapper;
}

function applyCadOpacity(scene: Object3D, opacity: number) {
  if (!(opacity < 1)) return;
  scene.traverse((obj) => {
    if (!(obj instanceof Mesh) || !obj.material) return;
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    for (const mat of mats) {
      mat.transparent = true;
      mat.opacity = opacity;
      mat.depthWrite = false;
    }
  });
}

export class CadErrorBoundary extends Component<{ onError?: () => void; fallback?: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch() {
    this.props.onError?.();
  }
  render() {
    if (this.state.failed) return this.props.fallback ?? null;
    return this.props.children;
  }
}

export function RobotCad({
  url,
  onReady,
  opacity = 1,
  targetSpan,
  targetSize,
}: {
  url: string;
  onReady?: () => void;
  opacity?: number;
  targetSpan?: number;
  targetSize?: Vec3;
}) {
  const gltf = useGLTF(url);
  const prepared = useMemo(() => {
    const cloned = gltf.scene.clone(true);
    prepareCadScene(cloned);
    const stats = measureCadScene(cloned);
    const size = targetSize || (targetSpan != null ? ([targetSpan, targetSpan, targetSpan] as Vec3) : undefined);
    const verdict = cadShouldShowMesh(stats.triangleCount, stats.span, targetSpan ?? (size ? Math.max(...size) : undefined), stats.size, stats.center, size);
    const fitted = verdict.ready ? applyCadCollisionFit(cloned, verdict.fit) : cloned;
    applyCadOpacity(cloned, opacity);
    return { scene: fitted, ready: verdict.ready };
  }, [gltf.scene, opacity, targetSpan, targetSize?.[0], targetSize?.[1], targetSize?.[2]]);
  useEffect(() => {
    if (prepared.ready) onReady?.();
    // Load completion is tied to the cloned scene, not the parent callback identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prepared]);
  return <primitive object={prepared.scene} />;
}

function partQuaternion(part: RobotPartTransform): [number, number, number, number] | undefined {
  if (!hasYupQuaternion(part)) return undefined;
  return [part.qx as number, part.qy as number, part.qz as number, part.qw as number];
}

const CAD_PRELOAD_LIMIT = 4;
const cadPreloadQueued = new Set<string>();
let cadPreloadInFlight = 0;
const cadPreloadWait: string[] = [];

function queueCadPreload(url: string) {
  if (!url || cadPreloadQueued.has(url)) return;
  cadPreloadQueued.add(url);
  cadPreloadWait.push(url);
  pumpCadPreload();
}

function pumpCadPreload() {
  while (cadPreloadInFlight < CAD_PRELOAD_LIMIT && cadPreloadWait.length) {
    const url = cadPreloadWait.shift();
    if (!url) return;
    cadPreloadInFlight += 1;
    Promise.resolve()
      .then(() => useGLTF.preload(url))
      .finally(() => {
        cadPreloadInFlight -= 1;
        pumpCadPreload();
      });
  }
}

function Placeholder({ label, color = theme.cream }: { label: string; color?: string }) {
  return (
    <Html center style={{ pointerEvents: "none", color, fontSize: "10px", whiteSpace: "nowrap" }}>
      {label}
    </Html>
  );
}

function CollisionMaterial({ color, opacity }: { color: string; opacity: number }) {
  const transparent = opacity < 0.999;
  return (
    <meshStandardMaterial
      color={color}
      transparent={transparent}
      opacity={opacity}
      metalness={0.22}
      roughness={0.42}
      depthWrite={!transparent}
    />
  );
}

function CollisionLocal({ collision, color, opacity }: { collision: CollisionShapeSpec; color: string; opacity: number }) {
  const pose = collision.pose || {};
  const position = ftcToThreePosition(pose.x || 0, pose.y || 0, pose.z || 0);
  const rotation = poseEulerRad(pose);
  if (collision.kind === "convex_mesh") {
    const url = `${API}/robot-assets/${collision.asset}`;
    return (
      <group position={position} rotation={rotation}>
        <CadErrorBoundary fallback={<Placeholder label="CAD failed" color={theme.restricted} />}>
          <Suspense fallback={<Placeholder label="Loading CAD…" />}>
            <RobotCad url={url} opacity={opacity} />
          </Suspense>
        </CadErrorBoundary>
      </group>
    );
  }
  if (collision.kind === "box") {
    return (
      <mesh position={position} rotation={rotation} userData={{ proxy: true }}>
        <boxGeometry args={ftcSizeToThree(collision.sizeIn)} />
        <CollisionMaterial color={color} opacity={opacity} />
      </mesh>
    );
  }
  if (collision.kind === "sphere") {
    return (
      <mesh position={position} rotation={rotation} userData={{ proxy: true }}>
        <sphereGeometry args={[collision.radiusIn || 0.5, 16, 16]} />
        <CollisionMaterial color={color} opacity={opacity} />
      </mesh>
    );
  }
  const length = collision.lengthIn || (collision.radiusIn || 0.5) * 2;
  return (
    <mesh position={position} rotation={[rotation[0] + Math.PI / 2, rotation[1], rotation[2]]} userData={{ proxy: true }}>
      <cylinderGeometry args={[collision.radiusIn || 0.5, collision.radiusIn || 0.5, length, 20]} />
      <CollisionMaterial color={color} opacity={opacity} />
    </mesh>
  );
}

export function CollisionPrimitive({ part, opacity = 0.94 }: { part: RigidPartSpec; opacity?: number }) {
  const collisions = (part.collision || []) as CollisionShapeSpec[];
  if (!collisions.length) return <Placeholder label="No collision proxy" />;
  const color = proxyColor(part.tags, part.id);
  return (
    <group userData={{ proxy: true, instanceId: part.id }}>
      {collisions.map((collision, index) => (
        <CollisionLocal key={`${part.id}-col-${index}`} collision={collision} color={color} opacity={opacity} />
      ))}
    </group>
  );
}

export function PartVisual({
  part,
  opacity = 1,
  showLabel = false,
  preferCad = true,
}: {
  part: RigidPartSpec;
  opacity?: number;
  showLabel?: boolean;
  preferCad?: boolean;
}) {
  const cadUrl = preferCad && part.visualAsset ? `${API}/robot-assets/${part.visualAsset}` : null;
  const [cadReady, setCadReady] = useState(false);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    setCadReady(false);
    setFailed(false);
    if (cadUrl) queueCadPreload(cadUrl);
  }, [cadUrl]);
  const placeholder = cadPlaceholderLabel({
    visualAsset: part.visualAsset,
    cacheState: part.cacheState,
    cacheReason: part.cacheReason,
    failed,
    ready: cadReady,
  });
  const cadMissing = !cadUrl || failed || !cadReady;
  const proxySpan = collisionSpan(part);
  const targetSpan = Math.max(proxySpan[0], proxySpan[1], proxySpan[2]);
  return (
    <>
      {cadUrl && !failed && (
        <CadErrorBoundary
          onError={() => {
            setCadReady(false);
            setFailed(true);
          }}
          fallback={null}
        >
          <Suspense fallback={null}>
            <RobotCad url={cadUrl} opacity={opacity} targetSpan={targetSpan} targetSize={proxySpan} onReady={() => setCadReady(true)} />
          </Suspense>
        </CadErrorBoundary>
      )}
      {cadMissing && <CollisionPrimitive part={part} opacity={Math.min(opacity, 0.94)} />}
      {showLabel && placeholder && <Placeholder label={placeholder} color={theme.restricted} />}
    </>
  );
}

export function RobotPartActor({
  part,
  design,
  opacity = 1,
  selected = false,
  onSelect,
}: {
  part: RobotPartTransform;
  design?: RobotDesign;
  opacity?: number;
  selected?: boolean;
  onSelect?: (id: string) => void;
}) {
  const spec = (design?.rigidParts || []).find((row) => row.id === part.id);
  const visualAsset = part.visualAsset || spec?.visualAsset;
  const cadUrl = visualAsset ? `${API}/robot-assets/${visualAsset}` : null;
  const [cadReady, setCadReady] = useState(false);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    setCadReady(false);
    setFailed(false);
    if (cadUrl) queueCadPreload(cadUrl);
  }, [cadUrl]);
  const orient = partQuaternion(part);
  const proxySpan = collisionSpan(spec || { collision: [{ kind: "box", sizeIn: [4, 4, 2] }] });
  const targetSpan = Math.max(proxySpan[0], proxySpan[1], proxySpan[2]);
  return (
    <group
      userData={{ instanceId: part.id }}
      position={ftcToThreePosition(part.x, part.y, part.z)}
      quaternion={orient}
      onClick={(event) => {
        event.stopPropagation();
        onSelect?.(part.id);
      }}
    >
      {cadUrl && !failed && (
        <CadErrorBoundary
          onError={() => {
            setCadReady(false);
            setFailed(true);
          }}
          fallback={null}
        >
          <Suspense fallback={null}>
            <RobotCad url={cadUrl} opacity={opacity} targetSpan={targetSpan} targetSize={proxySpan} onReady={() => setCadReady(true)} />
          </Suspense>
        </CadErrorBoundary>
      )}
      {(!cadUrl || !cadReady || failed) && spec && <CollisionPrimitive part={spec} opacity={Math.min(opacity, 0.94)} />}
      {selected && (
        <mesh userData={{ ghost: true }} raycast={() => undefined}>
          <boxGeometry args={collisionSpan(spec || { collision: [{ kind: "box", sizeIn: [4, 4, 2] }] }).map((n) => n + 0.2) as [number, number, number]} />
          <meshStandardMaterial color={theme.selected} wireframe transparent opacity={0.55} depthWrite={false} />
        </mesh>
      )}
    </group>
  );
}

export function CatalogPartInstance({
  id,
  part,
  pose,
  selected = false,
  ghost = false,
  ghostState,
  onSelect,
}: {
  id: string;
  part: CatalogPart;
  pose: Transform3;
  selected?: boolean;
  ghost?: boolean;
  ghostState?: "snap" | "free" | "invalid";
  onSelect?: (id: string) => void;
}) {
  const spec = catalogPartToRigid(id, part, undefined);
  const euler = poseEulerRad(pose);
  const opacity = ghost ? (ghostState === "invalid" ? 0.28 : 0.46) : 1;
  const span = collisionSpan(spec);
  const selectColor = ghostState === "snap" ? "#3dbf6a" : ghostState === "invalid" ? theme.restricted : theme.selected;
  return (
    <group
      userData={{ instanceId: ghost ? "ghost" : id, ghost, kind: spec.visualAsset ? "cad" : "proxy", tags: part.tags }}
      raycast={ghost ? () => undefined : undefined}
      position={ftcToThreePosition(pose.x || 0, pose.y || 0, pose.z || 0)}
      rotation={euler}
      onClick={(event) => {
        if (ghost) return;
        event.stopPropagation();
        onSelect?.(id);
      }}
    >
      <PartVisual part={spec} opacity={opacity} showLabel={Boolean(selected && !ghost)} preferCad />
      {(selected || ghost) && (
        <mesh userData={{ ghost: true }} raycast={() => undefined}>
          <boxGeometry args={[span[0] + 0.18, span[1] + 0.18, span[2] + 0.18]} />
          <meshStandardMaterial color={selectColor} wireframe transparent opacity={ghost ? 0.85 : 0.5} depthWrite={false} />
        </mesh>
      )}
    </group>
  );
}

export function CatalogAssembly({
  instances,
  poses,
  parts,
  selectedId,
  ghost,
  ghostState,
  hiddenIds,
  onSelect,
}: {
  instances: { id: string; sku: string }[];
  poses: Record<string, Transform3>;
  parts: Record<string, CatalogPart>;
  selectedId?: string | null;
  ghost?: { id: string; part: CatalogPart; pose: Transform3 } | null;
  ghostState?: "snap" | "free" | "invalid";
  hiddenIds?: Set<string>;
  onSelect?: (id: string) => void;
}) {
  return (
    <group>
      {instances.map((instance) => {
        if (hiddenIds?.has(instance.id)) return null;
        const part = parts[instance.id];
        const pose = poses[instance.id];
        if (!part || !pose) return null;
        return (
          <CatalogPartInstance
            key={instance.id}
            id={instance.id}
            part={part}
            pose={pose}
            selected={selectedId === instance.id}
            onSelect={onSelect}
          />
        );
      })}
      {ghost && (
        <CatalogPartInstance id={ghost.id} part={ghost.part} pose={ghost.pose} ghost ghostState={ghostState} />
      )}
    </group>
  );
}

export function MountMarker({
  world,
  compatible,
  active = false,
  occupied = false,
  selected = false,
  onSelect,
}: {
  world: [number, number, number];
  compatible: boolean;
  active?: boolean;
  occupied?: boolean;
  selected?: boolean;
  onSelect?: () => void;
}) {
  const radius = selected || active ? 0.38 : compatible ? 0.24 : 0.16;
  const color = occupied ? "#777178" : selected ? "#48a7ff" : compatible ? (active ? "#6dff9d" : "#3dbf6a") : theme.restricted;
  return (
    <group position={ftcToThreePosition(world[0], world[1], world[2])}>
      <mesh
        userData={{ mountMarker: true, occupied }}
        onClick={(event) => {
          event.stopPropagation();
          if (!occupied) onSelect?.();
        }}
      >
        <octahedronGeometry args={[radius, 0]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={occupied ? 0.42 : compatible ? (active || selected ? 1 : 0.88) : 0.28}
          emissive={selected ? "#174d78" : compatible ? "#1f7a3e" : "#000000"}
          emissiveIntensity={active || selected ? 0.7 : compatible ? 0.25 : 0}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}
