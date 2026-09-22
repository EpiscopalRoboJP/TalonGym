import type { CatalogPart, RobotAssembly } from "../api";
import { theme } from "../theme";
import { Panel } from "../ui";
import { sameSel, type BuilderSel } from "./store";

export function AssemblyTree({
  assembly,
  parts,
  sel,
  onSelect,
}: {
  assembly: RobotAssembly;
  parts: Record<string, CatalogPart>;
  sel: BuilderSel;
  onSelect: (sel: BuilderSel) => void;
}) {
  const children = new Map<string, string[]>();
  for (const connection of assembly.connections) {
    const list = children.get(connection.parent.instanceId) || [];
    list.push(connection.child.instanceId);
    children.set(connection.parent.instanceId, list);
  }
  const childIds = new Set(assembly.connections.map((row) => row.child.instanceId));
  const roots = assembly.instances.map((row) => row.id).filter((id) => !childIds.has(id));
  const primaryRoot = roots.includes(assembly.rootInstanceId || "") ? assembly.rootInstanceId : roots[0];
  function Node({ id, depth }: { id: string; depth: number }) {
    const instance = assembly.instances.find((row) => row.id === id);
    const part = parts[id];
    const on = sel.kind === "instance" && sel.id === id;
    return (
      <li>
        <button type="button" className={`list-item ${on ? "on" : ""}`} data-testid={`assembly-instance-${id}`} style={{ paddingLeft: `${0.75 + depth * 0.75}rem` }} onClick={() => onSelect({ kind: "instance", id })}>
          <span className="dot" style={{ background: on ? theme.selected : theme.gold }} />
          <span className="grow">
            <span className="title" style={{ display: "block" }}>
              {id}
            </span>
            <span className="meta" style={{ display: "block" }}>
              {part?.displayName || instance?.sku || "Loading catalog…"}
            </span>
          </span>
        </button>
        {(children.get(id) || []).length > 0 && (
          <ul className="list">
            {(children.get(id) || []).map((child) => (
              <Node key={child} id={child} depth={depth + 1} />
            ))}
          </ul>
        )}
      </li>
    );
  }
  return (
    <Panel className="assembly-tree" title="Assembly tree" bodyClass="panel-body flush scroll">
      <ul className="list" data-testid="assembly-tree">
        {primaryRoot ? <Node id={primaryRoot} depth={0} /> : <li className="note" style={{ padding: "0.75rem 1rem" }}>No catalog instances yet.</li>}
        {roots.filter((id) => id !== primaryRoot).map((id) => (
          <li key={`loose-${id}`}>
            <div className="assembly-loose-label">Loose subassembly</div>
            <Node id={id} depth={0} />
          </li>
        ))}
        <li>
          <button type="button" className={`list-item ${sameSel(sel, { kind: "chassis" }) ? "on" : ""}`} data-testid="assembly-chassis" onClick={() => onSelect({ kind: "chassis" })}>
            <span className="dot" style={{ background: theme.chassis }} />
            <span className="grow">
              <span className="title" style={{ display: "block" }}>
                Chassis & camera
              </span>
            </span>
          </button>
        </li>
      </ul>
    </Panel>
  );
}
