import type { ReactNode } from "react";
import type { CatalogPart, RobotPreset, Transform3 } from "../api";
import { Field, NumberField, Panel } from "../ui";
import type { BuilderSel } from "./store";

function withChassis(doc: RobotPreset, patch: { lengthIn?: number; widthIn?: number; massKg?: number }): RobotPreset {
  const fallback = { lengthIn: 18, widthIn: 14, massKg: 6 };
  return { ...doc, chassis: { ...fallback, ...doc.chassis, ...patch } };
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="section">
      <h3 className="section-title">{title}</h3>
      <div className="stack">{children}</div>
    </div>
  );
}

export function Inspector({
  doc,
  sel,
  part,
  pose,
  onPatchDoc,
  onPatchPose,
  onDetach,
  onPickUp,
  onReplace,
  onRemove,
}: {
  doc: RobotPreset;
  sel: BuilderSel;
  part?: CatalogPart;
  pose?: Transform3;
  onPatchDoc: (doc: RobotPreset) => void;
  onPatchPose: (pose: Transform3) => void;
  onDetach: () => void;
  onPickUp: () => void;
  onReplace: () => void;
  onRemove: () => void;
}) {
  const camera = doc.sensors?.find((s) => s.kind === "apriltag_camera") || doc.sensors?.[0];
  const instance = sel.kind === "instance" ? doc.assembly?.instances.find((row) => row.id === sel.id) : undefined;
  return (
    <Panel
      className="grow"
      title={sel.kind === "instance" ? sel.id : sel.kind === "camera" ? "Camera" : "Inspector"}
      bodyClass="panel-body scroll"
      actions={
        sel.kind === "instance" && (
          <div className="row">
            <button type="button" className="btn sm" onClick={onDetach}>
              Detach
            </button>
            <button type="button" className="btn sm" data-testid="pick-up-part" onClick={onPickUp}>
              Pick up
            </button>
            <button type="button" className="btn sm" onClick={onReplace}>
              Replace
            </button>
            <button type="button" className="btn sm danger" onClick={onRemove}>
              Remove
            </button>
          </div>
        )
      }
    >
      {instance && (
        <Section title="Catalog instance">
          <p className="note">
            {part?.displayName || instance.sku} · {part?.manufacturer} · {(part?.massKg || 0).toFixed(3)} kg
          </p>
          <Field id="inst-sku" label="SKU">
            <input id="inst-sku" value={instance.sku} readOnly />
          </Field>
          {pose && (
            <div className="fields three">
              <NumberField id="inst-x" label="X" unit="in" value={pose.x || 0} onChange={(n) => onPatchPose({ ...pose, x: n })} />
              <NumberField id="inst-y" label="Y" unit="in" value={pose.y || 0} onChange={(n) => onPatchPose({ ...pose, y: n })} />
              <NumberField id="inst-z" label="Z" unit="in" value={pose.z || 0} onChange={(n) => onPatchPose({ ...pose, z: n })} />
              <NumberField id="inst-yaw" label="Yaw" unit="°" value={pose.yawDeg || 0} onChange={(n) => onPatchPose({ ...pose, yawDeg: n })} />
            </div>
          )}
          <p className="note">Loose-root poses are numeric. Connected child poses are solved from exact hole and shaft snaps. Drag in the viewport to re-parent.</p>
        </Section>
      )}
      {(sel.kind === "chassis" || sel.kind === "camera") && (
        <>
          <Section title="General">
            <Field id="display-name" label="Display name">
              <input id="display-name" value={doc.displayName} onChange={(e) => onPatchDoc({ ...doc, displayName: e.target.value })} />
            </Field>
            <div className="fields two">
              <NumberField
                id="chassis-l"
                label="Length"
                unit="in"
                value={doc.chassis?.lengthIn ?? 18}
                onChange={(n) => onPatchDoc(withChassis(doc, { lengthIn: n }))}
              />
              <NumberField
                id="chassis-w"
                label="Width"
                unit="in"
                value={doc.chassis?.widthIn ?? 14}
                onChange={(n) => onPatchDoc(withChassis(doc, { widthIn: n }))}
              />
              <NumberField id="track" label="Track width" unit="in" value={doc.drivetrain?.trackWidthIn ?? 14} onChange={(n) => onPatchDoc({ ...doc, drivetrain: { ...(doc.drivetrain || { type: "mecanum" }), trackWidthIn: n } })} />
              {doc.drivetrain?.wheelbaseIn != null && (
                <NumberField id="wheelbase" label="Wheelbase" unit="in" value={doc.drivetrain.wheelbaseIn} onChange={(n) => onPatchDoc({ ...doc, drivetrain: { ...doc.drivetrain, wheelbaseIn: n } })} />
              )}
              {Number.isFinite(doc.motors?.gearRatio) && (
                <NumberField id="gear" label="Gear ratio" value={doc.motors?.gearRatio} onChange={(n) => onPatchDoc({ ...doc, motors: { ...(doc.motors || { drive: 4 }), gearRatio: n } })} />
              )}
            </div>
          </Section>
          <Section title="AprilTag camera">
            {camera ? (
              <>
                <NumberField
                  id="fov"
                  label="Field of view"
                  unit="°"
                  value={camera.fovDeg ?? 70}
                  onChange={(n) =>
                    onPatchDoc({
                      ...doc,
                      sensors: (doc.sensors || []).map((s) => (s.id === camera.id ? { ...s, fovDeg: n } : s)),
                    })
                  }
                />
                <NumberField
                  id="range"
                  label="Detection range"
                  unit="in"
                  value={camera.rangeIn ?? 96}
                  onChange={(n) =>
                    onPatchDoc({
                      ...doc,
                      sensors: (doc.sensors || []).map((s) => (s.id === camera.id ? { ...s, rangeIn: n } : s)),
                    })
                  }
                />
              </>
            ) : (
              <p className="note">This preset has no camera sensor.</p>
            )}
          </Section>
        </>
      )}
    </Panel>
  );
}
