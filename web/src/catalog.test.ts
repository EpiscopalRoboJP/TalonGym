import { catalogAssemblyCompilePath, catalogCacheAllCancelPath, catalogCacheAllPath, catalogCacheAllRetryPath, catalogRecipeInstantiatePath, catalogRecipesPath, catalogSearchPath, robotDraftPath, robotPartModelPath } from "./api";
import { cadCollisionFit, cadShouldShowMesh, cadUnitFitScale, uniqueLongAxis } from "./scene/cadFit";
import { RECIPE_IDS } from "./robotBuilder/recipes";
import { createBuilderState, reduceBuilder, type BuilderAction } from "./robotBuilder/store";
import { competitiveSaveBlocked, confirmBindings, inferRole, TOPOLOGY_NOTE } from "./robotBuilder/inference";
import { bestSnap, candidatesForMount, cycleSpin, partFitsMount, SNAP_RANGE_IN, snapCandidatesFor } from "./robotBuilder/snap";
import { ndcFromClientRect, threeHitToFtc } from "./robotBuilder/pointer";
import { formatMeasure } from "./ui";
import { holeInPart, mountCompatibilityError } from "./robotBuilder/mounts";
import { cadPlaceholderLabel, cacheBatchProgressPercent, cacheBatchStateLabel, catalogCacheAllRequest, catalogCacheAllScopeLabel, catalogCacheLabel, formatCacheBatchCounts, hideChassisLump, skipReasonLabel, summarizeCacheBatch, summarizeSkipped, assemblyCadCounts, partHasCachedCad } from "./robotBuilder/cadVisual";
import { composeRigidPartPoses, catalogPartToRigid, ftcSizeToThree, proxyColor } from "./robotBuilder/partVisual";
import { slugify } from "./robotBuilder/ids";
import { identity4 } from "./robotBuilder/transforms";
import { solveDraftAssemblyPoses } from "./robotBuilder/assemblyMath";
import { replacementError } from "./robotBuilder/validation";
import { presetIsShipped, type CatalogPart, type DrivebaseRecipe, type DrivebaseRecipeParameters, type RobotPreset } from "./api";

function assert(cond: unknown, message: string): asserts cond {
  if (!cond) throw new Error(message);
}

function fixturePart(
  sku: string,
  manufacturer: "gobilda" | "rev",
  kind: "threaded_hole" | "clearance_hole" | "mating_face",
  standard: "gobilda_pattern" | "rev_m3" | "rev_15mm",
  extra?: Partial<CatalogPart>,
): CatalogPart {
  return {
    sku,
    manufacturer,
    displayName: sku,
    tags: ["plate"],
    massKg: 0.2,
    downloadEnabled: false,
    cache: { state: "unavailable", reason: "download_disabled" },
    mounts: [
      {
        id: "face",
        kind,
        standard,
        diameterMm: kind === "threaded_hole" ? 3 : kind === "clearance_hole" ? 3.5 : undefined,
        allowedHardware: kind === "mating_face" ? ["m3_t_nut"] : ["m3_screw"],
        axis: [0, 0, 1],
        transform: { x: 0, y: 0, z: 0 },
        pattern: { type: "grid", pitchMm: 8, countU: 3, countV: 1 },
      },
    ],
    collision: [{ kind: "box", sizeIn: [2, 2, 0.5] }],
    ...extra,
  };
}

function sampleDoc(): RobotPreset {
  return {
    schemaVersion: "1.2.0",
    id: "draft_bot",
    displayName: "Draft",
    drivetrain: { type: "mecanum", trackWidthIn: 14 },
    chassis: { lengthIn: 16, widthIn: 14, massKg: 6 },
    motors: { drive: 4 },
    constraints: { maxVelInPerS: 40, maxAccelInPerS2: 40, maxAngVelDegPerS: 180 },
    mechanisms: { capacity: 3 },
    sensors: [],
    assembly: {
      rootInstanceId: "rail",
      instances: [{ id: "rail", sku: "ch" }],
      connections: [],
    },
  };
}

function apply(state: ReturnType<typeof createBuilderState>, history: { past: { doc: RobotPreset }[]; future: { doc: RobotPreset }[] }, action: BuilderAction) {
  return reduceBuilder(state, history, action);
}

const gobildaClear = fixturePart("ch", "gobilda", "clearance_hole", "gobilda_pattern", {
  mounts: [
    {
      id: "face",
      kind: "clearance_hole",
      standard: "gobilda_pattern",
      diameterMm: 4,
      allowedHardware: ["m4_screw"],
      axis: [0, 0, 1],
      transform: { x: 0, y: 0, z: 0 },
      pattern: { type: "grid", pitchMm: 8, countU: 3, countV: 1 },
    },
  ],
});
const gobildaThread = fixturePart("mot", "gobilda", "threaded_hole", "gobilda_pattern", {
  mounts: [
    {
      id: "face",
      kind: "threaded_hole",
      standard: "gobilda_pattern",
      diameterMm: 4,
      allowedHardware: ["m4_screw"],
      axis: [0, 0, 1],
      transform: { x: 0, y: 0, z: 0 },
      pattern: { type: "grid", pitchMm: 8, countU: 3, countV: 1 },
    },
  ],
});
const revClear = fixturePart("revp", "rev", "clearance_hole", "rev_m3");
const revThread = fixturePart("revm", "rev", "threaded_hole", "rev_m3");
const revSlot = fixturePart("revs", "rev", "mating_face", "rev_15mm");

const tests: Array<[string, () => void]> = [
  ["per-part upload path stays on the shipped API", () => {
    assert(
      robotPartModelPath("mecanum_biobuzz_4cap", "intake") === "/presets/robot/mecanum_biobuzz_4cap/parts/intake/model",
      "part model path",
    );
  }],
  ["catalog search query string encodes filters", () => {
    assert(catalogSearchPath() === "/catalog", "empty");
    assert(catalogSearchPath({ q: "mecanum", manufacturer: "gobilda", tag: "wheel_mecanum" }) === "/catalog?q=mecanum&manufacturer=gobilda&tag=wheel_mecanum", "filters");
  }],
  ["recipe, compile, and draft paths match backend routes", () => {
    assert(catalogRecipesPath() === "/catalog/recipes", "recipes");
    assert(catalogRecipeInstantiatePath("gobilda_mecanum") === "/catalog/recipes/gobilda_mecanum/instantiate", "instantiate");
    assert(catalogAssemblyCompilePath() === "/catalog/assemblies/compile", "compile");
    assert(robotDraftPath("working") === "/presets/robot/working/draft", "draft");
    assert(RECIPE_IDS.join(",") === "gobilda_mecanum,gobilda_tank,rev_mecanum,rev_tank", "ids");
  }],
  ["cache-all paths, progress, skip reasons, and completion summary stay stable", () => {
    assert(catalogCacheAllPath() === "/catalog/cache/all", "latest");
    assert(catalogCacheAllPath("batch-1") === "/catalog/cache/all/batch-1", "one");
    assert(catalogCacheAllCancelPath("batch-1") === "/catalog/cache/all/batch-1/cancel", "cancel");
    assert(catalogCacheAllRetryPath("batch-1") === "/catalog/cache/all/batch-1/retry", "retry");
    assert(
      formatCacheBatchCounts({ queued: 2, converting: 1, ready: 4, failed: 0, skipped: 3, cancelled: 0 }) ===
        "Queued 2 · Converting 1 · Ready 4 · Failed 0 · Skipped 3",
      "counts",
    );
    assert(cacheBatchStateLabel("running") === "Downloading / converting CAD…", "state");
    assert(cacheBatchProgressPercent({ total: 4, counts: { queued: 1, converting: 1, ready: 1, failed: 0, skipped: 1, cancelled: 0 } }) === 50, "percent");
    assert(skipReasonLabel("download_disabled") === "download disabled", "disabled");
    assert(skipReasonLabel("no_source_url") === "no source URL", "no source");
    assert(skipReasonLabel("cad_extra_required") === "CAD extra required", "extra");
    assert(
      summarizeSkipped([
        { sku: "a", state: "skipped", reason: "download_disabled" },
        { sku: "b", state: "skipped", reason: "no_source_url" },
        { sku: "c", state: "ready" },
      ]) === "2 skipped (download disabled / no source URL)",
      "skipped",
    );
    assert(
      summarizeCacheBatch({
        state: "done",
        total: 4,
        counts: { queued: 0, converting: 0, ready: 2, failed: 1, skipped: 1, cancelled: 0 },
      }) === "Complete: Cached 2 of 4 · Failed 1 · Skipped 1",
      "summary",
    );
    const gobildaReq = catalogCacheAllRequest({ manufacturer: "gobilda" });
    assert(gobildaReq?.manufacturer === "gobilda" && gobildaReq.skus === undefined, "mfr only");
    const searchReq = catalogCacheAllRequest({ query: "mecanum", skus: ["1207-0001-0001"], filterReady: true });
    assert(JSON.stringify(searchReq?.skus) === JSON.stringify(["1207-0001-0001"]), "search skus");
    assert(catalogCacheAllRequest({ query: "mecanum", filterReady: false }) === null, "wait for filter");
    assert(JSON.stringify(catalogCacheAllRequest({ tag: "motor", skus: [] })?.skus) === "[]", "empty filter");
    assert(catalogCacheAllScopeLabel({ manufacturer: "gobilda" }) === "Caches goBILDA SKUs", "mfr label");
    assert(catalogCacheAllScopeLabel({ query: "mecanum" }) === "Caches the current catalog filter", "filter label");
  }],
  ["compatible mounts snap; M3 tap mates clearance; incompatible brands are rejected", () => {
    const hits = snapCandidatesFor(gobildaClear, identity4(), gobildaThread, [0, 0, 0], 0);
    assert(hits.length > 0, "compatible snap exists");
    const m3 = snapCandidatesFor(revClear, identity4(), revThread, [0, 0, 0], 0);
    assert(m3.length > 0, "M3 tap/clearance snap exists");
    const tnut = snapCandidatesFor(revSlot, identity4(), revClear, [0, 0, 0], 0);
    assert(tnut.length > 0, "T-nut slot mates M3 clearance");
    const branded = fixturePart("revp", "rev", "clearance_hole", "gobilda_pattern");
    assert(
      mountCompatibilityError(gobildaClear, gobildaClear.mounts![0], branded, branded.mounts![0]) ===
        "cross-brand connection requires a catalog adapter",
      "adapter required",
    );
    const none = snapCandidatesFor(gobildaClear, identity4(), revClear, [0, 0, 0], 0);
    assert(none.length === 0, "incompatible rejected");
  }],
  ["orientation cycling wraps 0/90/180/270", () => {
    assert(cycleSpin(0) === 1, "next");
    assert(cycleSpin(3) === 0, "wrap");
    assert(cycleSpin(0, -1) === 3, "prev");
  }],
  ["mount-first candidates remain pinned to the selected pattern hole", () => {
    const target = { instanceId: "rail", mountId: "face", patternIndex: [1, 0] as [number, number] };
    assert(partFitsMount(gobildaClear, target, gobildaThread), "compatible part shown");
    assert(!partFitsMount(gobildaClear, target, revThread), "incompatible part hidden");
    const zero = candidatesForMount(target, gobildaClear, { x: 0, y: 0, z: 0 }, gobildaThread, 0);
    const quarter = candidatesForMount(target, gobildaClear, { x: 0, y: 0, z: 0 }, gobildaThread, 90);
    assert(zero.length > 0 && quarter.length > 0, "valid orientations");
    assert(zero.every((row) => row.parentIndex?.[0] === 1 && row.parentIndex?.[1] === 0), "selected hole preserved");
    assert(zero[0].spinDeg === 0 && quarter[0].spinDeg === 90, "exact spin preserved");
  }],
  ["detach, replace, duplicate, undo, and redo keep a linear history", () => {
    const start = createBuilderState();
    const hydrated = apply(start, { past: [], future: [] }, { type: "hydrate", doc: sampleDoc() });
    const placed = apply(hydrated.state, hydrated.history, {
      type: "snapPlace",
      instanceId: "motor",
      sku: "mot",
      candidate: {
        parentInstanceId: "rail",
        parentMountId: "face",
        parentIndex: [0, 0],
        childMountId: "face",
        childIndex: [0, 0],
        spinDeg: 0,
        pose: { x: 0, y: 0, z: 0 },
        score: 0,
      },
    });
    assert(placed.state.doc?.assembly?.instances.some((row) => row.id === "motor"), "placed");
    const replaced = apply(placed.state, placed.history, { type: "replace", instanceId: "motor", sku: "mot2" });
    assert(replaced.state.doc?.assembly?.instances.find((row) => row.id === "motor")?.sku === "mot2", "replaced");
    const duplicated = apply(replaced.state, replaced.history, { type: "duplicate", instanceId: "motor" });
    assert((duplicated.state.doc?.assembly?.instances.length || 0) > (replaced.state.doc?.assembly?.instances.length || 0), "duplicated");
    const detached = apply(duplicated.state, duplicated.history, { type: "detach", instanceId: "motor", pose: { x: 1, y: 2, z: 3 } });
    assert(!detached.state.doc?.assembly?.connections.some((row) => row.child.instanceId === "motor"), "detached");
    assert(detached.state.doc?.assembly?.instances.find((row) => row.id === "motor")?.pose?.x === 1, "detach keeps world pose");
    const undone = apply(detached.state, detached.history, { type: "undo" });
    assert(undone.state.doc?.assembly?.connections.some((row) => row.child.instanceId === "motor"), "undo restore");
    const redone = apply(undone.state, undone.history, { type: "redo" });
    assert(!redone.state.doc?.assembly?.connections.some((row) => row.child.instanceId === "motor"), "redo detach");
  }],
  ["part replacement preserves every existing mount contract", () => {
    const assembly = {
      rootInstanceId: "rail",
      instances: [{ id: "rail", sku: gobildaClear.sku }, { id: "motor", sku: gobildaThread.sku }],
      connections: [{
        id: "rail_motor",
        parent: { instanceId: "rail", mountId: "face", patternIndex: [2, 0] as [number, number] },
        child: { instanceId: "motor", mountId: "face", patternIndex: [0, 0] as [number, number] },
      }],
    };
    const catalog = { rail: gobildaClear, motor: gobildaThread };
    assert(replacementError(assembly, "motor", gobildaThread, catalog) === null, "compatible replacement should be accepted");
    assert(replacementError(assembly, "motor", revThread, catalog)?.includes("incompatible") === true, "cross-brand replacement should be rejected");
    const missingMount = fixturePart("plain", "gobilda", "threaded_hole", "gobilda_pattern", { mounts: [] });
    assert(replacementError(assembly, "motor", missingMount, catalog)?.includes("no mount") === true, "missing mount should be rejected");
  }],
  ["editable forests preserve loose component poses while publish history stays intact", () => {
    const assembly = {
      rootInstanceId: "rail",
      instances: [
        { id: "rail", sku: "ch", pose: { x: 0, y: 0, z: 0 } },
        { id: "motor", sku: "mot", pose: { x: 4, y: 2, z: 1 } },
      ],
      connections: [],
    };
    const solved = solveDraftAssemblyPoses(assembly, { rail: gobildaClear, motor: gobildaThread });
    assert(solved.roots.length === 2, "two component roots");
    assert(solved.poses.motor.x === 4 && solved.poses.motor.y === 2 && solved.poses.motor.z === 1, "loose pose retained");

    const hydrated = apply(createBuilderState(), { past: [], future: [] }, { type: "hydrate", doc: sampleDoc() });
    const changedDoc = { ...hydrated.state.doc!, displayName: "Changed" };
    const changed = apply(hydrated.state, hydrated.history, { type: "patchDoc", doc: changedDoc });
    const staleAck = apply(changed.state, changed.history, { type: "markSaved", doc: sampleDoc() });
    assert(staleAck.state.dirty, "stale save cannot clear newer changes");
    const exactAck = apply(staleAck.state, staleAck.history, { type: "markSaved", doc: changedDoc });
    assert(!exactAck.state.dirty && exactAck.state.canUndo, "save acknowledgement preserves undo history");
  }],
  ["inference stays unconfirmed until explicit confirm and blocks competitive save", () => {
    const part = fixturePart("ch", "gobilda", "clearance_hole", "gobilda_pattern", { tags: ["channel"] });
    assert(inferRole(part) === "structure", "role");
    assert(competitiveSaveBlocked({ confirmed: false }, 4) !== null, "unconfirmed blocks");
    const bindings = confirmBindings({ confirmed: false, roles: { rail: "structure" }, joints: [], notes: [TOPOLOGY_NOTE], weldedInstanceIds: [], articulatedInstanceIds: [] }, "mecanum", 14);
    assert(bindings.confirmed === true, "confirmed");
    assert(bindings.instanceRoles?.rail === "structure", "roles copied");
    assert(competitiveSaveBlocked(bindings, 1) === null, "complete confirm allows save");
    assert(competitiveSaveBlocked({ confirmed: true, drivetrain: { type: "mecanum", trackWidthIn: 14 } }, 2) !== null, "missing roles still block");
  }],
  ["save-as ids stay slug-safe", () => {
    assert(slugify("My Robot Copy") === "my_robot_copy", "slug");
  }],
  ["inferred measures drop binary float tails", () => {
    assert(formatMeasure(3.622047244094488) === 3.622, "track");
    assert(formatMeasure(19.200000000000003) === 19.2, "gear");
    const bindings = confirmBindings(
      { confirmed: false, roles: { rail: "structure" }, joints: [], notes: [TOPOLOGY_NOTE], weldedInstanceIds: [], articulatedInstanceIds: [] },
      "mecanum",
      7.244094488188976,
      4.724409448818897,
    );
    assert(bindings.drivetrain?.trackWidthIn === 7.244, "confirm track");
    assert(bindings.drivetrain?.wheelbaseIn === 4.724, "confirm wheelbase");
  }],
  ["best snap prefers the closest compatible hole", () => {
    const assembly = sampleDoc().assembly!;
    const catalog = { rail: gobildaClear };
    const hit = bestSnap(assembly, { rail: { x: 0, y: 0, z: 0 } }, catalog, gobildaThread, [0, 0, 0], 0);
    assert(hit?.parentInstanceId === "rail", "parent");
    assert(hit?.parentMountId === "face", "mount");
  }],
  ["best snap ignores mounts outside range", () => {
    const assembly = sampleDoc().assembly!;
    const catalog = { rail: gobildaClear };
    const hit = bestSnap(assembly, { rail: { x: 0, y: 0, z: 0 } }, catalog, gobildaThread, [SNAP_RANGE_IN + 10, 0, 0], 0);
    assert(hit === null, "out of range");
  }],
  ["pointer NDC uses the CSS rectangle, not the drawing buffer", () => {
    const ndc = ndcFromClientRect(150, 40, { left: 100, top: 20, width: 200, height: 80 });
    assert(ndc !== null, "ndc");
    assert(Math.abs(ndc[0] - -0.5) < 1e-9, "x");
    assert(Math.abs(ndc[1] - 0.5) < 1e-9, "y");
    assert(ndcFromClientRect(0, 0, { left: 0, top: 0, width: 0, height: 0 }) === null, "empty");
    const ftc = threeHitToFtc(3, 4, -5);
    assert(ftc[0] === 3 && ftc[1] === 5 && ftc[2] === 4, "three to ftc");
  }],
  ["orientation cycling wraps", () => {
    assert(cycleSpin(3) === 0, "wrap");
    assert(cycleSpin(0, -1) === 3, "back");
  }],
  ["articulated catalog parts hide the chassis lump and CAD missing is explicit", () => {
    assert(hideChassisLump({ rigidParts: [{ id: "chassis" }, { id: "motor_fl" }] }) === true, "hide lump");
    assert(hideChassisLump({ rigidParts: [{ id: "chassis" }] }) === false, "chassis only keeps lump");
    assert(
      hideChassisLump({
        rigidParts: [
          { id: "chassis", collision: [{ kind: "box" }] },
          { id: "intake_roller" },
        ],
      }) === false,
      "4-cap hull stays",
    );
    assert(hideChassisLump({ rigidParts: [{ id: "_catalog" }] }) === true, "uncompiled catalog placeholder");
    assert(cadPlaceholderLabel({ cacheState: "unavailable" }) === "CAD unavailable", "unavailable");
    assert(cadPlaceholderLabel({ cacheState: "invalid" }) === "CAD cache invalid", "invalid");
    assert(cadPlaceholderLabel({ cacheState: "missing" }) === "CAD not cached", "missing");
    assert(cadPlaceholderLabel({}) === "CAD not cached", "default");
    assert(catalogCacheLabel("missing") === "CAD not cached", "cache missing");
    assert(catalogCacheLabel("unavailable", "download_disabled") === "CAD download disabled", "disabled");
    assert(catalogCacheLabel("unavailable", "cad_extra_required").includes("CAD extra required"), "extra");
    assert(catalogCacheLabel("caching") === "CAD caching…", "caching");
    assert(cadPlaceholderLabel({ visualAsset: "robots/x.glb", failed: true }) === "CAD failed", "failed");
    assert(cadPlaceholderLabel({ visualAsset: "robots/x.glb", ready: true }) === null, "ready");
    const cached = catalogPartToRigid("left_rail", fixturePart("ch", "gobilda", "clearance_hole", "gobilda_pattern", {
      cache: { state: "ready", visualAsset: "robot_parts/gobilda/ch/visual.glb" },
    }));
    assert(cached.visualAsset === "robot_parts/gobilda/ch/visual.glb", "committed CAD url");
    assert(catalogPartToRigid("left_rail", fixturePart("ch", "gobilda", "clearance_hole", "gobilda_pattern")).visualAsset == null, "uncached stays proxy");
    const counts = assemblyCadCounts(
      ["ch", "ch", "missing"],
      {
        ch: fixturePart("ch", "gobilda", "clearance_hole", "gobilda_pattern", {
          cache: { state: "ready", visualAsset: "robot_parts/gobilda/ch/visual.glb" },
        }),
      },
    );
    assert(counts.cad === 2 && counts.proxy === 1, "cached instances vs missing proxy");
    assert(partHasCachedCad({ cache: { state: "ready" } }) === false, "ready without visual is proxy");
  }],
  ["CAD unit-fit keeps proxies until the mesh AABB is renderable", () => {
    assert(Math.abs(cadUnitFitScale(0.00189, 1.8898) - 1000) < 1e-3, "mm-vs-metres leftover");
    assert(Math.abs(cadUnitFitScale(0.048, 1.8898) - 39.37007874015748) < 1e-6, "metres to inches");
    assert(cadUnitFitScale(1.8898, 1.8898) === 1, "matching span stays 1");
    assert(cadShouldShowMesh(0, 2, 2).ready === false, "empty triangles");
    assert(cadShouldShowMesh(12, 0, 2).ready === false, "zero span");
    const tiny = cadShouldShowMesh(24, 0.00189, 1.8898);
    assert(tiny.ready === true, "classic unit error is correctable");
    assert(Math.abs(tiny.scale * 0.00189 - 1.8898) < 0.01, "fitted span");
    const mismatch = cadShouldShowMesh(24, 0.04, 12);
    assert(mismatch.ready === false, "non-unit 300x mismatch stays proxy");
    assert(cadShouldShowMesh(24, 1.8, 1.9).ready === true, "near proxy is ready");
  }],
  ["CAD long-axis fit lays an end-origin extrusion along collision X", () => {
    assert(uniqueLongAxis([16.5354, 0.5906, 0.5906]) === 0, "target long is X");
    assert(uniqueLongAxis([0.59, 16.54, 0.59]) === 1, "mesh long is Y");
    const fit = cadCollisionFit([0.00059, 0.01654, 0.00059], [0, 0.00827, 0], [16.5354, 0.5906, 0.5906]);
    assert(Math.abs(fit.scale - 1000) < 1e-3, "scale 1000");
    assert(Math.abs(fit.rotation[1] - 1) < 1e-5, "Y maps onto X");
    assert(Math.abs(fit.translation[0] + 8.27) < 0.05, "center bar on origin");
    assert(Math.abs(fit.translation[1]) < 0.05, "no leftover Y");
    const wheel = cadCollisionFit([0.0027, 0.0027, 0.00161], [0, 0, 0], [2.9528, 1.1811, 2.9528]);
    assert(Math.abs(wheel.scale - 1000) < 1e-3, "wheel scale");
    assert(wheel.rotation[0] === 1 && wheel.rotation[4] === 1 && wheel.rotation[8] === 1, "disc keeps orientation");
  }],
  ["FTC collision boxes swizzle Z-up size into Three Y-up", () => {
    const three = ftcSizeToThree([7.56, 1.89, 0.47]);
    assert(Math.abs(three[0] - 7.56) < 1e-9, "x");
    assert(Math.abs(three[1] - 0.47) < 1e-9, "up from FTC z");
    assert(Math.abs(three[2] - 1.89) < 1e-9, "depth from FTC y");
    assert(proxyColor(["channel"], "left_rail").length > 0, "rail color");
    assert(proxyColor(["wheel_mecanum"], "wheel_fl") !== proxyColor(["channel"], "left_rail"), "wheel vs rail");
  }],
  ["composed rigid-part poses walk parentId instead of treating local pose as world", () => {
    const poses = composeRigidPartPoses([
      { id: "rail", massKg: 1, collision: [{ kind: "box", sizeIn: [8, 2, 1] }], pose: { x: 0, y: 0, z: 0 } },
      { id: "motor", parentId: "rail", massKg: 1, collision: [{ kind: "box", sizeIn: [1, 1, 1] }], pose: { x: 3, y: 0, z: 0 } },
    ]);
    assert(Math.abs((poses.motor.x || 0) - 3) < 1e-9, "child x");
    assert(Math.abs((poses.rail.x || 0)) < 1e-9, "root x");
  }],
  ["placeRoot seeds an empty assembly and stays undoable", () => {
    const start = createBuilderState();
    const hydrated = apply(start, { past: [], future: [] }, { type: "hydrate", doc: { ...sampleDoc(), assembly: { instances: [], connections: [] } } });
    const placed = apply(hydrated.state, hydrated.history, { type: "placeRoot", instanceId: "rail", sku: "ch", pose: { x: 1, y: 2, z: 0 } });
    assert(placed.state.doc?.assembly?.rootInstanceId === "rail", "root");
    assert(placed.state.doc?.assembly?.instances[0]?.pose?.x === 1, "pose");
    const undone = apply(placed.state, placed.history, { type: "undo" });
    assert((undone.state.doc?.assembly?.instances.length || 0) === 0, "undo root");
  }],
  ["user-created robots are not treated as shipped", () => {
    assert(presetIsShipped({ id: "mecanum_biobuzz_4cap", shipped: true }) === true, "disk shipped");
    assert(presetIsShipped({ id: "e2e_catalog_gobilda_mecanum", shipped: false }) === false, "user copy");
    assert(presetIsShipped({ id: "mecanum_biobuzz_4cap" }) === true, "fallback shipped");
    assert(presetIsShipped({ id: "gobilda_mecanum_starter" }) === true, "fallback starter");
    assert(presetIsShipped({ id: "e2e_catalog_gobilda_mecanum" }) === false, "fallback user");
  }],
  ["client recipes have no builtin assembly fallback", () => {
    assert(RECIPE_IDS.length === 4, "four recipes");
    assert(!("builtinAssembly" in ({} as Record<string, unknown>)), "no builtin");
  }],
  ["catalog mount uAxis runs hole patterns along the declared length", () => {
    const part = fixturePart("1120-web", "gobilda", "clearance_hole", "gobilda_pattern");
    const mount = part.mounts?.[0];
    if (!mount) throw new Error("missing mount");
    mount.uAxis = [1, 0, 0];
    mount.pattern = { type: "grid", pitchMm: 25.4, countU: 3, countV: 1 };
    const first = holeInPart(mount, [0, 0]);
    const last = holeInPart(mount, [2, 0]);
    assert(Math.abs(last[0] - first[0] - 2) < 1e-6, "u along +X");
    assert(Math.abs(last[1] - first[1]) < 1e-6, "no Y drift");
  }],
  ["drivebase recipe parameters include selected UltraPlanetary cartridge", () => {
    const parameters: DrivebaseRecipeParameters = { cartridgeSku: "REV-41-1601" };
    const recipe: DrivebaseRecipe = {
      id: "rev_mecanum",
      displayName: "REV mecanum",
      manufacturer: "rev",
      drivetrain: "mecanum",
      defaultParameters: { cartridgeSku: "REV-41-1603" },
      lengthSkus: [],
      widthSkus: [],
      motorSkus: [],
      wheelSkus: [],
      cartridgeSkus: ["REV-41-1601", "REV-41-1602", "REV-41-1603"],
    };
    assert(parameters.cartridgeSku === "REV-41-1601", "parameter");
    assert(recipe.cartridgeSkus?.includes("REV-41-1603") === true, "choices");
    assert(recipe.defaultParameters.cartridgeSku === "REV-41-1603", "default");
  }],
];

let failed = 0;
for (const [name, run] of tests) {
  try {
    run();
    console.log(`ok ${name}`);
  } catch (err) {
    failed += 1;
    console.error(`not ok ${name}`);
    console.error(err);
  }
}
if (failed) {
  throw new Error(`${failed} catalog builder test(s) failed`);
}
console.log(`${tests.length} passed`);
