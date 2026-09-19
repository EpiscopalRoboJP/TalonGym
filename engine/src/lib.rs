//! Batched Rapier2D ABI. Not wired until this crate meets the Phase 0
//! throughput gate. Python `Planar2DBackend` is the documented fallback
//! and already runs chassis / wall / floor-piece contacts.
//!
//! Built by FTC Team 17986 — 904 Robo Eagles and Team 27268 — Talon Strike.

pub fn engine_name() -> &'static str {
    "rapier2d-stub"
}
