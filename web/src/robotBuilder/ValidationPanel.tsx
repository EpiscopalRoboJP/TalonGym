import { Alert, Panel } from "../ui";
import type { ValidationResult } from "./validation";

export function ValidationPanel({ result }: { result: ValidationResult }) {
  return (
    <Panel title="Validation" sub={result.blocking.length ? "blocking" : result.warnings.length ? "warnings" : "ok"} bodyClass="panel-body scroll">
      {result.blocking.length === 0 && result.warnings.length === 0 && <p className="note">No blocking mount or collision issues.</p>}
      {result.blocking.map((issue) => (
        <Alert key={`${issue.code}-${issue.message}`} kind="bad">
          <b>{issue.code}</b>
          <div>{issue.message}</div>
        </Alert>
      ))}
      {result.warnings.map((issue) => (
        <Alert key={`${issue.code}-${issue.message}`} kind="warn">
          <b>{issue.code}</b>
          <div>{issue.message}</div>
        </Alert>
      ))}
    </Panel>
  );
}
