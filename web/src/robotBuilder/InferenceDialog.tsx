import { useEffect, useRef } from "react";
import type { InferenceReport } from "./inference";
import { Panel } from "../ui";

export function InferenceDialog({
  report,
  onCancel,
  onConfirm,
}: {
  report: InferenceReport;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const node = dialogRef.current;
    if (!node) return;
    if (!node.open) node.showModal();
    return () => {
      if (node.open) node.close();
    };
  }, []);
  return (
    <dialog ref={dialogRef} className="modal" data-testid="inference-dialog" onCancel={onCancel}>
      <Panel title="Confirm inferred mechanisms" sub="Joints, wheels, and roles are inferred from connectivity. Confirm before competitive save.">
        <div className="stack" style={{ padding: "1rem" }}>
          {report.notes.map((note) => (
            <p key={note} className="note">
              {note}
            </p>
          ))}
          <ul className="list">
            {Object.entries(report.roles).map(([id, role]) => (
              <li key={id} className="list-item">
                <span className="title">{id}</span>
                <span className="meta">{role}</span>
              </li>
            ))}
          </ul>
          <p className="note">
            {report.joints.length} joints · {report.weldedInstanceIds.length} welded · {report.articulatedInstanceIds.length} articulated
          </p>
          <div className="row end">
            <button type="button" className="btn" onClick={onCancel}>
              Not yet
            </button>
            <button type="button" className="btn primary" data-testid="confirm-inference-submit" onClick={onConfirm}>
              Confirm inference
            </button>
          </div>
        </div>
      </Panel>
    </dialog>
  );
}
