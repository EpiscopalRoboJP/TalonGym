export const CREDIT_LINE =
  "Built by FTC Team 17986 — 904 Robo Eagles and Team 27268 — Talon Strike.";

export const LICENSE_NOTICE =
  "TalonGym — Copyright (C) 2026 TalonGym contributors. " +
  `${CREDIT_LINE} ` +
  "Licensed under GNU GPL v3 or later; see LICENSE in the repository.";

export function logLicenseNotice(): void {
  console.info(LICENSE_NOTICE);
}
