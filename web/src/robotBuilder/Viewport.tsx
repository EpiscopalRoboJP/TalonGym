import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Grid, OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { MOUSE, Object3D, Plane, Raycaster, Vector2, Vector3 } from "three";
import type { CatalogPart, RobotPreset, Transform3 } from "../api";
import { RobotActor } from "../scene/FieldScene";
import { CatalogAssembly } from "../scene/catalogAssembly";
import { ftcToThreePosition } from "../scene/cadAssets";
import { theme } from "../theme";
import { ndcFromClientRect, threeHitToFtc } from "./pointer";
import { highlightMounts, ORIENTATION_SPINS, worldMountHoles, type MountTarget, type SnapCandidate } from "./snap";
import { matrixFromPose } from "./transforms";

export type BuilderHit = { x: number; y: number };

declare global {
  interface Window {
    __talonBuilder?: {
      dragging: boolean;
      placing: boolean;
      candidate: string | null;
      poses: Record<string, Transform3>;
      connections: { parent: { instanceId: string }; child: { instanceId: string } }[];
      visibleIds: string[];
      mounts?: { instanceId: string; mountId: string; index: [number, number]; x: number; y: number; occupied?: boolean }[];
      snapCamera?: (view: CameraView) => void;
      project: (id: string) => BuilderHit | null;
    };
  }
}

function instanceIdFromObject(obj: Object3D | null): string | null {
  let cur: Object3D | null = obj;
  while (cur) {
    const id = cur.userData?.instanceId;
    if (typeof id === "string" && id !== "ghost") return id;
    cur = cur.parent;
  }
  return null;
}

function PointerBridge({
  placing,
  candidate,
  poses,
  connections,
  onPointer,
  onRelease,
  onCancel,
  onInstanceDown,
}: {
  placing: boolean;
  candidate: SnapCandidate | null;
  poses: Record<string, Transform3>;
  connections: { parent: { instanceId: string }; child: { instanceId: string } }[];
  onPointer: (point: [number, number, number]) => void;
  onRelease: () => void;
  onCancel: () => void;
  onInstanceDown: (id: string, point: [number, number, number]) => void;
}) {
  const { camera, gl, scene } = useThree();
  const placingRef = useRef(placing);
  const candidateRef = useRef(candidate);
  const posesRef = useRef(poses);
  const connectionsRef = useRef(connections);
  const onPointerRef = useRef(onPointer);
  const onReleaseRef = useRef(onRelease);
  const onCancelRef = useRef(onCancel);
  const onInstanceDownRef = useRef(onInstanceDown);
  placingRef.current = placing;
  candidateRef.current = candidate;
  posesRef.current = poses;
  connectionsRef.current = connections;
  onPointerRef.current = onPointer;
  onReleaseRef.current = onRelease;
  onCancelRef.current = onCancel;
  onInstanceDownRef.current = onInstanceDown;

  useEffect(() => {
    const canvas = gl.domElement;
    canvas.style.touchAction = "none";
    const raycaster = new Raycaster();
    const ndc = new Vector2();
    const floor = new Plane(new Vector3(0, 1, 0), 0);
    const hit = new Vector3();

    function worldFromClient(clientX: number, clientY: number): [number, number, number] | null {
      const rect = canvas.getBoundingClientRect();
      const coords = ndcFromClientRect(clientX, clientY, rect);
      if (!coords) return null;
      raycaster.setFromCamera(new Vector2(coords[0], coords[1]), camera);
      const hits = raycaster.intersectObjects(scene.children, true).filter((row) => !row.object.userData?.ghost && instanceIdFromObject(row.object) !== "ghost");
      if (hits[0]) return threeHitToFtc(hits[0].point.x, hits[0].point.y, hits[0].point.z);
      if (!raycaster.ray.intersectPlane(floor, hit)) return null;
      return threeHitToFtc(hit.x, hit.y, hit.z);
    }

    function projectId(id: string, rect: DOMRect): { x: number; y: number } | null {
      const pose = posesRef.current[id];
      if (!pose || rect.width < 1 || rect.height < 1) return null;
      const v = new Vector3(...ftcToThreePosition(pose.x || 0, pose.y || 0, pose.z || 0));
      v.project(camera);
      if (!Number.isFinite(v.x) || !Number.isFinite(v.y)) return null;
      return {
        x: rect.left + (v.x * 0.5 + 0.5) * rect.width,
        y: rect.top + (-v.y * 0.5 + 0.5) * rect.height,
      };
    }

    function nearestInstance(clientX: number, clientY: number): string | null {
      const rect = canvas.getBoundingClientRect();
      let bestId: string | null = null;
      let best = 42;
      for (const id of Object.keys(posesRef.current)) {
        const projected = projectId(id, rect);
        if (!projected) continue;
        const dist = Math.hypot(projected.x - clientX, projected.y - clientY);
        if (dist < best) {
          best = dist;
          bestId = id;
        }
      }
      return bestId;
    }

    function onMove(ev: PointerEvent) {
      if (!placingRef.current) return;
      const point = worldFromClient(ev.clientX, ev.clientY);
      if (point) onPointerRef.current(point);
    }
    function onDown(ev: PointerEvent) {
      if (ev.button === 2) {
        if (placingRef.current) {
          ev.preventDefault();
          ev.stopImmediatePropagation();
          onCancelRef.current();
        }
        return;
      }
      if (ev.button !== 0) return;
      if (placingRef.current) {
        ev.preventDefault();
        ev.stopImmediatePropagation();
        const point = worldFromClient(ev.clientX, ev.clientY);
        if (point) onPointerRef.current(point);
        onReleaseRef.current();
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const coords = ndcFromClientRect(ev.clientX, ev.clientY, rect);
      if (!coords) return;
      ndc.set(coords[0], coords[1]);
      raycaster.setFromCamera(ndc, camera);
      const hits = raycaster.intersectObjects(scene.children, true).filter((row) => !row.object.userData?.ghost);
      const meshId = hits.map((row) => instanceIdFromObject(row.object)).find((value) => value);
      const id = nearestInstance(ev.clientX, ev.clientY) || meshId;
      if (!id) return;
      const point = worldFromClient(ev.clientX, ev.clientY);
      if (!point) return;
      ev.preventDefault();
      ev.stopImmediatePropagation();
      onInstanceDownRef.current(id, point);
    }
    function onContext(ev: MouseEvent) {
      ev.preventDefault();
    }

    window.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerdown", onDown, true);
    canvas.addEventListener("contextmenu", onContext);
    return () => {
      window.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerdown", onDown, true);
      canvas.removeEventListener("contextmenu", onContext);
    };
  }, [camera, gl, scene]);

  useFrame(() => {
    const rect = gl.domElement.getBoundingClientRect();
    const currentPoses = posesRef.current;
    const visible = new Set<string>();
    scene.traverse((obj) => {
      const id = obj.userData?.instanceId;
      if (typeof id === "string" && id !== "ghost" && !obj.userData?.ghost) visible.add(id);
    });
    window.__talonBuilder = {
      snapCamera: window.__talonBuilder?.snapCamera,
      mounts: window.__talonBuilder?.mounts,
      dragging: placingRef.current,
      placing: placingRef.current,
      candidate: candidateRef.current?.parentInstanceId || null,
      poses: currentPoses,
      connections: connectionsRef.current,
      visibleIds: [...visible],
      project(id: string) {
        const pose = currentPoses[id];
        if (!pose || rect.width < 1 || rect.height < 1) return null;
        const v = new Vector3(...ftcToThreePosition(pose.x || 0, pose.y || 0, pose.z || 0));
        v.project(camera);
        if (!Number.isFinite(v.x) || !Number.isFinite(v.y)) return null;
        return {
          x: rect.left + (v.x * 0.5 + 0.5) * rect.width,
          y: rect.top + (-v.y * 0.5 + 0.5) * rect.height,
        };
      },
    };
  });
  return null;
}

function FramingCamera({ span }: { span: number }) {
  const { camera } = useThree();
  useEffect(() => {
    camera.position.set(span * 0.9, span * 0.75, span * 0.9);
    camera.near = 0.1;
    camera.far = Math.max(240, span * 8);
    camera.updateProjectionMatrix();
  }, [camera, span]);
  return null;
}

function MountDebugProjection({
  markers,
}: {
  markers: { instanceId: string; mountId: string; index: [number, number]; world: [number, number, number]; occupied?: boolean }[];
}) {
  const { camera, gl } = useThree();
  useFrame(() => {
    if (!window.__talonBuilder) return;
    const rect = gl.domElement.getBoundingClientRect();
    window.__talonBuilder.mounts = markers.map((marker) => {
      const point = new Vector3(marker.world[0], marker.world[2], -marker.world[1]).project(camera);
      return {
        instanceId: marker.instanceId,
        mountId: marker.mountId,
        index: marker.index,
        occupied: marker.occupied,
        x: rect.left + ((point.x + 1) / 2) * rect.width,
        y: rect.top + ((1 - point.y) / 2) * rect.height,
      };
    });
  });
  return null;
}

function MountOverlay({
  markers,
  selectedMount,
  candidate,
  onSelect,
}: {
  markers: {
    instanceId: string;
    mountId: string;
    index: [number, number];
    compatible: boolean;
    occupied?: boolean;
  }[];
  selectedMount?: MountTarget | null;
  candidate?: SnapCandidate | null;
  onSelect: (target: MountTarget) => void;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let frame = 0;
    const update = () => {
      const root = rootRef.current;
      if (root) {
        const rect = root.getBoundingClientRect();
        const projected = window.__talonBuilder?.mounts || [];
        root.querySelectorAll<HTMLButtonElement>(".mount-overlay-node").forEach((button, index) => {
          const point = projected[index];
          if (!point) {
            button.hidden = true;
            return;
          }
          button.hidden = false;
          button.style.transform = `translate(${point.x - rect.left}px, ${point.y - rect.top}px) translate(-50%, -50%)`;
        });
      }
      frame = requestAnimationFrame(update);
    };
    frame = requestAnimationFrame(update);
    return () => cancelAnimationFrame(frame);
  }, [markers]);
  return (
    <div ref={rootRef} className="mount-overlay" aria-label="Selected part mounts">
      {markers.map((marker) => {
        const selected =
          selectedMount?.instanceId === marker.instanceId &&
          selectedMount.mountId === marker.mountId &&
          selectedMount.patternIndex[0] === marker.index[0] &&
          selectedMount.patternIndex[1] === marker.index[1];
        const active =
          candidate?.parentMountId === marker.mountId &&
          candidate.parentIndex?.[0] === marker.index[0] &&
          candidate.parentIndex?.[1] === marker.index[1];
        return (
          <button
            key={`${marker.instanceId}-${marker.mountId}-${marker.index.join("-")}`}
            type="button"
            className={`mount-overlay-node${selected ? " selected" : ""}${active ? " active" : ""}${marker.occupied ? " occupied" : ""}`}
            aria-label={`${marker.occupied ? "Occupied" : "Available"} mount ${marker.mountId} ${marker.index.join(",")}`}
            disabled={marker.occupied}
            onClick={() => onSelect({ instanceId: marker.instanceId, mountId: marker.mountId, patternIndex: marker.index })}
          />
        );
      })}
    </div>
  );
}

type CameraView = "iso" | "front" | "right" | "top";

function CameraActions({ span }: { span: number }) {
  const { camera } = useThree();
  const snapRef = useRef<(view: CameraView) => void>(() => undefined);
  snapRef.current = (view: CameraView) => {
    const distance = Math.max(18, span * 1.45);
    const positions: Record<CameraView, [number, number, number]> = {
      iso: [distance, distance * 0.78, distance],
      front: [0, distance * 0.18, distance],
      right: [distance, distance * 0.18, 0],
      top: [0, distance, distance * 0.35],
    };
    camera.position.set(...positions[view]);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  };
  const bridge = useRef((view: CameraView) => snapRef.current(view));
  useFrame(() => {
    if (window.__talonBuilder) window.__talonBuilder.snapCamera = bridge.current;
  });
  return null;
}

function CameraPersistence({ draftId }: { draftId: string }) {
  const { camera } = useThree();
  const frames = useRef(0);
  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(`talongym:builder-camera:${draftId}`) || "null") as
        | { x: number; y: number; z: number }
        | null;
      if (saved && [saved.x, saved.y, saved.z].every(Number.isFinite)) {
        camera.position.set(saved.x, saved.y, saved.z);
        camera.lookAt(0, 0, 0);
      }
    } catch {
      localStorage.removeItem(`talongym:builder-camera:${draftId}`);
    }
  }, [camera, draftId]);
  useFrame(() => {
    frames.current += 1;
    if (frames.current % 30 !== 0) return;
    localStorage.setItem(
      `talongym:builder-camera:${draftId}`,
      JSON.stringify({ x: camera.position.x, y: camera.position.y, z: camera.position.z }),
    );
  });
  return null;
}

export function BuilderViewport({
  doc,
  parts,
  poses,
  selectedId,
  ghost,
  pendingPart,
  hiddenIds,
  dragging,
  candidate,
  spinIndex,
  selectedMount,
  highlights,
  previewFlywheel,
  previewHood,
  onSelect,
  onPointer,
  onRelease,
  onCancel,
  onInstanceDown,
  onMountSelect,
  onRotate,
  onFreePose,
}: {
  doc: RobotPreset;
  parts: Record<string, CatalogPart>;
  poses: Record<string, Transform3>;
  selectedId?: string | null;
  ghost?: { id: string; part: CatalogPart; pose: Transform3 } | null;
  pendingPart?: CatalogPart | null;
  hiddenIds?: Set<string>;
  dragging: boolean;
  candidate: SnapCandidate | null;
  spinIndex: number;
  selectedMount?: MountTarget | null;
  highlights?: { mountId: string; index: [number, number]; world: [number, number, number]; compatible: boolean }[];
  previewFlywheel: number;
  previewHood: number;
  onSelect: (id: string | null) => void;
  onPointer: (point: [number, number, number]) => void;
  onRelease: () => void;
  onCancel: () => void;
  onInstanceDown: (id: string, point: [number, number, number]) => void;
  onMountSelect: (target: MountTarget) => void;
  onRotate: (delta: number) => void;
  onFreePose: (instanceId: string, pose: Transform3) => void;
}) {
  const assembly = doc.assembly;
  const span = useMemo(() => {
    let max = Math.max(doc.chassis?.lengthIn || 18, doc.chassis?.widthIn || 18, 18);
    for (const pose of Object.values(poses)) {
      max = Math.max(max, Math.abs(pose.x || 0) + 6, Math.abs(pose.y || 0) + 6, Math.abs(pose.z || 0) + 6);
    }
    return max;
  }, [doc.chassis?.lengthIn, doc.chassis?.widthIn, poses]);
  const [gizmoMode, setGizmoMode] = useState<"translate" | "rotate">("translate");
  const [helpOpen, setHelpOpen] = useState(false);
  const movableIds = useMemo(() => {
    if (!assembly) return new Set<string>();
    const connected = new Set(assembly.connections.map((connection) => connection.child.instanceId));
    return new Set(
      assembly.instances
        .filter((instance) => instance.id === assembly.rootInstanceId || !connected.has(instance.id))
        .map((instance) => instance.id),
    );
  }, [assembly]);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement | null)?.closest("input, textarea, select, [contenteditable='true']")) return;
      if (event.key.toLowerCase() === "f") {
        event.preventDefault();
        window.__talonBuilder?.snapCamera?.("iso");
      } else if (event.key === "Enter" && dragging) {
        event.preventDefault();
        onRelease();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [dragging, onRelease]);
  const markers = useMemo(() => {
    if (highlights) return highlights.map((row) => ({ ...row, instanceId: selectedId || "" }));
    if (!assembly || !selectedId) return [];
    const part = parts[selectedId];
    const pose = poses[selectedId];
    if (!part || !pose) return [];
    const rows = pendingPart
      ? highlightMounts(part, matrixFromPose(pose), pendingPart)
      : worldMountHoles(part, matrixFromPose(pose)).map((row) => ({ ...row, mountId: row.mount.id, compatible: true }));
    const stride = Math.max(1, Math.ceil(rows.length / 6));
    return rows.filter((_row, index) => index % stride === 0).map((row) => ({
      ...row,
      instanceId: selectedId,
      occupied: assembly.connections.some(
        (connection) =>
          connection.parent.instanceId === selectedId &&
          connection.parent.mountId === row.mountId &&
          (connection.parent.patternIndex?.[0] || 0) === row.index[0] &&
          (connection.parent.patternIndex?.[1] || 0) === row.index[1],
      ),
    }));
  }, [assembly, highlights, parts, pendingPart, poses, selectedId]);
  const yaw = ORIENTATION_SPINS[((spinIndex % ORIENTATION_SPINS.length) + ORIENTATION_SPINS.length) % ORIENTATION_SPINS.length];
  const ghostState = !dragging ? undefined : candidate ? "snap" : pendingPart ? "invalid" : "free";
  const status = !dragging
    ? selectedMount
      ? `${selectedMount.instanceId} · ${selectedMount.mountId} selected · choose a compatible part`
      : selectedId
      ? `${selectedId} selected · click a mount node to add a part · RMB orbits`
      : "Select a catalog part to place · click a placed part to pick up · RMB orbits"
    : candidate
      ? `Snap to ${candidate.parentInstanceId} · ${candidate.parentMountId} · ${yaw}°`
      : pendingPart
        ? "No compatible mount in range"
        : "Loading part…";
  const hudHint = dragging
    ? "Enter / LMB confirm · Esc cancel · Q/E rotate"
    : selectedId
      ? "Green: available · gray: occupied · click a mount · F frame"
      : "LMB select · RMB orbit · wheel zoom";
  return (
    <div
      className={`builder-viewport${dragging ? " is-placing" : ""}${selectedId ? " has-selection" : ""}`}
      data-testid="builder-viewport"
      data-dragging={dragging ? "true" : "false"}
      data-placing={dragging ? "true" : "false"}
      data-selected={selectedId || ""}
    >
      <Canvas
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: false }}
        resize={{ scroll: true, debounce: 0 }}
        style={{ width: "100%", height: "100%", display: "block", touchAction: "none" }}
        tabIndex={-1}
        onPointerMissed={(event) => {
          if (dragging) return;
          if (event.button === 0) onSelect(null);
        }}
      >
        <PerspectiveCamera makeDefault fov={40} position={[span * 0.9, span * 0.75, span * 0.9]} />
        <FramingCamera span={span} />
        <CameraActions span={span} />
        <CameraPersistence draftId={doc.id} />
        <color attach="background" args={[theme.scene]} />
        <ambientLight intensity={0.78} />
        <directionalLight position={[40, 80, 30]} intensity={1.15} />
        <Grid args={[span * 2, span * 2]} cellSize={6} sectionSize={18} cellColor={theme.grid} sectionColor={theme.gridSection} fadeDistance={Math.max(80, span * 4)} raycast={() => undefined} />
        <PointerBridge
          placing={dragging}
          candidate={candidate}
          poses={poses}
          connections={doc.assembly?.connections || []}
          onPointer={onPointer}
          onRelease={onRelease}
          onCancel={onCancel}
          onInstanceDown={onInstanceDown}
        />
        {assembly?.instances.length ? (
          <CatalogAssembly
            instances={assembly.instances}
            poses={poses}
            parts={parts}
            selectedId={selectedId}
            ghost={ghost}
            ghostState={ghostState}
            hiddenIds={hiddenIds}
            onSelect={(id) => {
              if (!dragging) onSelect(id);
            }}
          />
        ) : (
          <RobotActor
            design={{
              chassis: doc.chassis,
              intakes: doc.intakes,
              launchers: doc.launchers,
              visualAsset: doc.visualAsset,
              visualOffset: doc.visualOffset,
              collisionShape: doc.chassis?.collisionShape,
              sensors: doc.sensors,
              piecePath: doc.piecePath,
              rigidParts: doc.rigidParts,
              joints: doc.joints,
              actuators: doc.actuators,
            }}
            showFov
            launchPreview={doc.piecePath ? { flywheelFrac: previewFlywheel, hoodFrac: previewHood } : undefined}
          />
        )}
        {assembly?.instances.length && doc.piecePath && (
          <RobotActor
            design={{ chassis: doc.chassis, sensors: doc.sensors, piecePath: doc.piecePath, actuators: doc.actuators }}
            hideBody
            showFov
            launchPreview={{ flywheelFrac: previewFlywheel, hoodFrac: previewHood }}
          />
        )}
        <MountDebugProjection markers={markers} />
        <OrbitControls
          makeDefault
          enabled={!dragging}
          enableDamping={false}
          enablePan
          mouseButtons={{ LEFT: -1 as unknown as (typeof MOUSE)[keyof typeof MOUSE], MIDDLE: MOUSE.PAN, RIGHT: MOUSE.ROTATE }}
        />
      </Canvas>
      <MountOverlay markers={markers} selectedMount={selectedMount} candidate={candidate} onSelect={onMountSelect} />
      {dragging && (
        <div className="placement-hud" data-testid="placement-hud">
          <span className="placement-hud-title">{pendingPart?.displayName || "Placing part"}</span>
          <span className="placement-hud-keys">Q/E rotate · R cycle · LMB place · Esc cancel</span>
        </div>
      )}
      {selectedId && !dragging && (
        <div className="placement-hud select" data-testid="selection-hud">
          <span className="placement-hud-title">{selectedId}</span>
          <span className="placement-hud-keys">Click to pick up · G grab · RMB orbit</span>
        </div>
      )}
      <div
        className={`viewport-status${dragging ? (candidate ? " snap" : " none") : ""}`}
        data-testid="drag-status"
        data-has-snap={candidate ? "true" : "false"}
        data-spin={String(yaw)}
      >
        {status}
      </div>
      <div className="viewport-hint" data-testid="placement-hint">
        {hudHint}
      </div>
      <div className="orientation-cube" data-testid="orientation-cube" aria-label="View and placement orientation">
        <button type="button" className="cube-face cube-top" title="Top view" onClick={() => window.__talonBuilder?.snapCamera?.("top")}>Top</button>
        <button type="button" className="cube-face cube-front" title="Front view" onClick={() => window.__talonBuilder?.snapCamera?.("front")}>Front</button>
        <button type="button" className="cube-face cube-right" title="Right view" onClick={() => window.__talonBuilder?.snapCamera?.("right")}>Right</button>
        <button type="button" className="cube-face cube-iso" title="Isometric view" onClick={() => window.__talonBuilder?.snapCamera?.("iso")}>Iso</button>
        {dragging && (
          <div className="cube-rotation" data-testid="mate-rotation-controls">
            <button type="button" aria-label="Rotate mate counterclockwise" onClick={() => onRotate(-1)}>−90°</button>
            <output>{yaw}°</output>
            <button type="button" aria-label="Rotate mate clockwise" onClick={() => onRotate(1)}>+90°</button>
          </div>
        )}
      </div>
      <div className="keybind-help">
        <button type="button" className="keybind-summary" data-testid="keybind-toggle" onClick={() => setHelpOpen((open) => !open)}>
          Controls
        </button>
        {helpOpen && (
          <>
            <div><b>LMB</b> select/confirm · <b>RMB</b> orbit · <b>Wheel</b> zoom</div>
            <div><b>M</b> choose mount · <b>Q/E</b> rotate · <b>Enter</b> confirm · <b>Esc</b> cancel</div>
            <div><b>F</b> frame · <b>Delete</b> remove · <b>Ctrl Z</b> undo</div>
          </>
        )}
      </div>
      {selectedId && movableIds.has(selectedId) && !dragging && (
        <div className="gizmo-mode" data-testid="gizmo-mode">
          <button type="button" className={gizmoMode === "translate" ? "active" : ""} onClick={() => setGizmoMode("translate")}>Move</button>
          <button type="button" className={gizmoMode === "rotate" ? "active" : ""} onClick={() => setGizmoMode("rotate")}>Rotate</button>
          {gizmoMode === "translate" ? (
            <>
              <button type="button" aria-label="Move negative X" onClick={() => onFreePose(selectedId, { ...poses[selectedId], x: (poses[selectedId]?.x || 0) - 0.25 })}>−X</button>
              <button type="button" aria-label="Move positive X" onClick={() => onFreePose(selectedId, { ...poses[selectedId], x: (poses[selectedId]?.x || 0) + 0.25 })}>+X</button>
              <button type="button" aria-label="Move negative Y" onClick={() => onFreePose(selectedId, { ...poses[selectedId], y: (poses[selectedId]?.y || 0) - 0.25 })}>−Y</button>
              <button type="button" aria-label="Move positive Y" onClick={() => onFreePose(selectedId, { ...poses[selectedId], y: (poses[selectedId]?.y || 0) + 0.25 })}>+Y</button>
              <button type="button" aria-label="Move negative Z" onClick={() => onFreePose(selectedId, { ...poses[selectedId], z: (poses[selectedId]?.z || 0) - 0.25 })}>−Z</button>
              <button type="button" aria-label="Move positive Z" onClick={() => onFreePose(selectedId, { ...poses[selectedId], z: (poses[selectedId]?.z || 0) + 0.25 })}>+Z</button>
            </>
          ) : (
            <>
              <button type="button" aria-label="Rotate counterclockwise" onClick={() => onFreePose(selectedId, { ...poses[selectedId], yawDeg: (poses[selectedId]?.yawDeg || 0) - 15 })}>−15°</button>
              <button type="button" aria-label="Rotate clockwise" onClick={() => onFreePose(selectedId, { ...poses[selectedId], yawDeg: (poses[selectedId]?.yawDeg || 0) + 15 })}>+15°</button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
