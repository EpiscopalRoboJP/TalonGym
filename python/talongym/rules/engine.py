from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


MECHANISM_VERBS = ("idle", "intake", "score", "open_gate", "stow")


@dataclass
class TickEvent:
    kind: str
    volume_id: str | None = None
    piece_id: str | None = None
    robot_id: str | None = None
    fsm_id: str | None = None
    from_state: str | None = None
    to_state: str | None = None
    phase: str | None = None
    piece_type: str | None = None
    piece_attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleContext:
    phase: str
    events: list[TickEvent]
    volume_occupancy: dict[str, set[str]]
    prev_occupancy: dict[str, set[str]]
    accumulators: dict[str, Any]
    match_vars: dict[str, Any]
    gate_state: dict[str, str]
    queues: dict[str, list[str]]
    queue_caps: dict[str, int]
    actor_id: str
    actor_volumes: set[str]
    fire_counts: dict[str, int]
    ident_types: dict[str, str] = field(default_factory=dict)
    piece_ops: list[tuple[str, str | None, str | None]] = field(default_factory=list)


class RuleEngine:
    def __init__(self, scoring: dict[str, Any]) -> None:
        self.scoring = scoring
        self.nodes: list[dict[str, Any]] = list(scoring.get("nodes") or [])
        self.acc_defs = {a["id"]: a for a in scoring.get("accumulators") or []}
        channels = scoring.get("scoreChannels") or {}
        self.true_score_id = channels.get("trueScore", "true_score")

    def init_accumulators(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for acc in self.scoring.get("accumulators") or []:
            init = acc.get("init", 0)
            out[acc["id"]] = deepcopy(init)
        return out

    def init_match_vars(self, rng) -> dict[str, Any]:
        values: dict[str, Any] = {}
        acc_seed: dict[str, Any] = {}
        for var in self.scoring.get("matchVariables") or []:
            domain = var.get("domain") or {}
            if "enum" in domain:
                choices = list(domain["enum"])
                values[var["id"]] = choices[int(rng.integers(0, len(choices)))]
            elif "intRange" in domain:
                lo = int(domain["intRange"]["min"])
                hi = int(domain["intRange"]["max"])
                values[var["id"]] = int(rng.integers(lo, hi + 1))
            repeat = var.get("repeat")
            if repeat and isinstance(values.get(var["id"]), str):
                seq = list(values[var["id"]]) * int(repeat.get("times", 1))
                acc_seed[repeat["intoAccumulator"]] = seq
        return values, acc_seed

    def evaluate(self, ctx: RuleContext) -> list[dict[str, Any]]:
        explains: list[dict[str, Any]] = []
        true_delta = 0.0
        for node in self.nodes:
            phases = node.get("enabledPhases") or ["AUTO"]
            if ctx.phase not in phases:
                continue
            matches = self._trigger_matches(node.get("trigger") or {}, ctx)
            if not matches:
                continue
            for match in matches:
                if not self._conditions_ok(node.get("conditions") or [], ctx, match):
                    continue
                key = f"{node['id']}:{node.get('fireScope', 'episode')}:{self._scope_id(node, ctx, match)}"
                max_fires = node.get("maxFires")
                if max_fires is not None and ctx.fire_counts.get(key, 0) >= int(max_fires):
                    continue
                pts = self._apply_actions(node, ctx, match)
                true_delta += pts
                ctx.fire_counts[key] = ctx.fire_counts.get(key, 0) + 1
                if node.get("explain"):
                    explains.append({"id": node["id"], "explain": node["explain"], "points": pts})
        return explains, true_delta

    def _scope_id(self, node: dict[str, Any], ctx: RuleContext, match: TickEvent) -> str:
        scope = node.get("fireScope", "episode")
        if scope == "actor":
            return match.robot_id or ctx.actor_id
        if scope == "piece":
            return match.piece_id or "none"
        if scope == "phase":
            return ctx.phase
        return "episode"

    def _trigger_matches(self, trigger: dict[str, Any], ctx: RuleContext) -> list[TickEvent]:
        kind = trigger.get("kind")
        if kind == "alwaysTick":
            return [TickEvent(kind="alwaysTick", robot_id=ctx.actor_id)]
        if kind == "phaseEnd":
            return [e for e in ctx.events if e.kind == "phaseEnd" and (not trigger.get("phase") or e.phase == trigger["phase"])]
        wanted = trigger.get("volumeId")
        piece_filter = trigger.get("filterPieceType")
        types = {t.strip() for t in str(piece_filter).split(",")} if piece_filter else None
        out: list[TickEvent] = []
        if kind in {"volumeEnter", "volumeExit", "contactStart", "contactEnd", "fsmTransition"}:
            for e in ctx.events:
                if e.kind != kind:
                    continue
                if wanted and e.volume_id != wanted and e.fsm_id != wanted:
                    continue
                if types and e.piece_type and e.piece_type not in types:
                    continue
                out.append(e)
        return out

    def _conditions_ok(self, conditions: list[dict[str, Any]], ctx: RuleContext, match: TickEvent) -> bool:
        return all(self._cond(c, ctx, match) for c in conditions)

    def _cond(self, c: dict[str, Any], ctx: RuleContext, match: TickEvent) -> bool:
        op = c.get("op")
        if op == "and":
            return all(self._cond(x, ctx, match) for x in c.get("allOf") or [])
        if op == "or":
            return any(self._cond(x, ctx, match) for x in c.get("anyOf") or [])
        if op == "not":
            inner = c.get("not")
            return not self._cond(inner, ctx, match) if inner else True
        if op == "phaseIs":
            return ctx.phase == c.get("equals")
        if op == "pieceTypeIs":
            return match.piece_type == c.get("pieceType")
        if op == "pieceAttrEquals":
            return match.piece_attrs.get(c.get("attr")) == c.get("equals")
        if op == "actorInVolume":
            return c.get("volumeId") in ctx.actor_volumes
        if op == "actorNotInVolume":
            return c.get("volumeId") not in ctx.actor_volumes
        if op == "accumulatorGte":
            return float(ctx.accumulators.get(c.get("accumulator"), 0) or 0) >= float(c.get("value", 0))
        if op == "accumulatorEq":
            return ctx.accumulators.get(c.get("accumulator")) == c.get("value")
        if op == "matchVarEquals":
            return ctx.match_vars.get(c.get("matchVar")) == c.get("equals")
        if op == "gateRetaining":
            eid = c.get("elementId") or ""
            return ctx.gate_state.get(eid, "closed") == "closed"
        if op == "queueHasSlot":
            acc = c.get("accumulator")
            eid = c.get("elementId") or ""
            if acc:
                q = ctx.accumulators.get(acc) or []
            else:
                q = ctx.queues.get(eid, [])
            cap = 9
            if acc and acc in ctx.queue_caps:
                cap = ctx.queue_caps[acc]
            elif eid and eid in ctx.queue_caps:
                cap = ctx.queue_caps[eid]
            return len(q) < cap
        if op == "volumeHasPieceType":
            vol = c.get("volumeId") or ""
            ptype = c.get("pieceType")
            for ident in ctx.volume_occupancy.get(vol) or set():
                if ctx.ident_types.get(ident) == ptype:
                    return True
            return False
        if op == "sequencePrefix":
            return True  # match count applied in addScore; presence of sequences is enough
        if op == "sequenceEquals":
            return ctx.accumulators.get(c.get("accumulator")) == c.get("value")
        return True

    def _pattern_matches(self, ctx: RuleContext, node: dict[str, Any]) -> int:
        cond = next((c for c in (node.get("conditions") or []) if c.get("op") == "sequencePrefix"), {})
        seq = list(ctx.accumulators.get(cond.get("accumulator") or "") or [])
        target = list(ctx.accumulators.get(cond.get("targetAccumulator") or "") or [])
        n = 0
        for i, color in enumerate(seq):
            if i < len(target) and color == target[i]:
                n += 1
        return n

    def _apply_actions(self, node: dict[str, Any], ctx: RuleContext, match: TickEvent) -> float:
        gained = 0.0
        per_index = any(c.get("op") == "sequencePrefix" for c in (node.get("conditions") or []))
        for action in node.get("actions") or []:
            kind = action.get("kind")
            acc = action.get("accumulator")
            if kind == "addScore":
                pts = float(action.get("points") or 0.0)
                if per_index:
                    n = self._pattern_matches(ctx, node)
                    pts *= n
                    ctx.accumulators["pattern_points"] = int(ctx.accumulators.get("pattern_points") or 0) + int(pts)
                ctx.accumulators[acc or self.true_score_id] = float(
                    ctx.accumulators.get(acc or self.true_score_id) or 0
                ) + pts
                if action.get("channel") == "trueScore":
                    gained += pts
            elif kind == "setFlag" and acc:
                ctx.accumulators[acc] = True
            elif kind == "incAccumulator" and acc:
                ctx.accumulators[acc] = (ctx.accumulators.get(acc) or 0) + (action.get("delta") or 1)
            elif kind == "pushSequence" and acc:
                seq = list(ctx.accumulators.get(acc) or [])
                color = match.piece_attrs.get(action.get("pieceAttr") or "color")
                if color is not None:
                    seq.append(color)
                    cap = self.acc_defs.get(acc, {}).get("maxLen")
                    if cap:
                        seq = seq[: int(cap)]
                    ctx.accumulators[acc] = seq
                    ctx.queues[acc] = seq
            elif kind == "openGate":
                fsm = action.get("fsmId") or next(iter(ctx.gate_state), "gate")
                ctx.gate_state[fsm] = "open"
                ctx.piece_ops.append(("release_queue", None, None))
            elif kind == "closeGate":
                fsm = action.get("fsmId") or next(iter(ctx.gate_state), "gate")
                ctx.gate_state[fsm] = "closed"
            elif kind == "transferPiece":
                if match.piece_id:
                    ctx.piece_ops.append(("transfer", match.piece_id, action.get("toVolume")))
            elif kind == "despawnPiece":
                if match.piece_id:
                    ctx.piece_ops.append(("despawn", match.piece_id, None))
            elif kind == "emitExplain":
                pass
        return gained
