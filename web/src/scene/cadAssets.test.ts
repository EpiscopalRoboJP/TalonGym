import {
  BIOBUZZ_GLB,
  LEGACY_PIECE_RADIUS_IN,
  buildPieceCatalog,
  cadCacheTokenFromManifest,
  explicitCadCacheToken,
  fieldAssetUrl,
  inferPieceVisualAsset,
  pieceThreeEulerRad,
  pieceThreePose,
  pieceThreeQuaternion,
  resolveBackgroundAsset,
  resolveCadManifestPath,
  resolvePieceRadiusIn,
  resolvePieceVisualAsset,
  siblingCadManifestPath,
  usesCadPieceVisual,
  visualOffsetYawDeg,
  type CadManifest,
} from "./cadAssets";
import type { Frame, FramePiece } from "../api";

function assert(cond: unknown, message: string): asserts cond {
  if (!cond) throw new Error(message);
}

function almostEqual(a: number, b: number, eps = 1e-9) {
  assert(Math.abs(a - b) <= eps, `expected ${a} ≈ ${b}`);
}

function frame(partial: Partial<Frame> = {}): Frame {
  return {
    t: 0,
    trueScore: 0,
    robots: [],
    pieces: [],
    elements: [],
    matchVarsPrivileged: {},
    observedMatchVars: {},
    fieldSizeIn: { width: 144, depth: 144 },
    explains: [],
    queues: {},
    gate: {},
    vision: [],
    ...partial,
  };
}

function pollenCatalog(manifest?: CadManifest) {
  return buildPieceCatalog(
    frame({
      gamePieces: [
        {
          typeId: "pollen",
          shape: { kind: "circle", radius: 1.4 },
          visualAsset: "seasons/biobuzz_2026/pieces/pollen.glb",
        },
        {
          typeId: "nectar_red",
          shape: { kind: "circle", radius: 1.8 },
          visualAsset: "seasons/biobuzz_2026/pieces/nectar_red.glb",
        },
      ],
    }),
    manifest,
  );
}

const tests: Array<[string, () => void]> = [
  ["old replay falls back to 2.5 in primitive", () => {
    const piece: FramePiece = { id: "p1", x: 4, y: -8, color: "Y" };
    const catalog = buildPieceCatalog(frame());
    assert(!usesCadPieceVisual(piece, catalog), "old replay must not bind a CAD GLB");
    assert(resolvePieceVisualAsset(piece, catalog) === null, "old replay has no visualAsset");
    assert(resolvePieceRadiusIn(piece, catalog) === LEGACY_PIECE_RADIUS_IN, "legacy radius is 2.5 in");
  }],
  ["CAD piece uses type catalog radius and GLB", () => {
    const catalog = pollenCatalog();
    const piece: FramePiece = { id: "flower_1", typeId: "pollen", x: -40, y: -64, z: 1.4, color: "Y" };
    assert(resolvePieceVisualAsset(piece, catalog) === "seasons/biobuzz_2026/pieces/pollen.glb", "pollen GLB");
    almostEqual(resolvePieceRadiusIn(piece, catalog), 1.4);
    const nectar: FramePiece = { id: "n1", typeId: "nectar_red", x: 0, y: 0, color: "R" };
    almostEqual(resolvePieceRadiusIn(nectar, catalog), 1.8);
    assert(resolvePieceVisualAsset(nectar, catalog) === "seasons/biobuzz_2026/pieces/nectar_red.glb", "nectar GLB");
  }],
  ["payload radius and visualAsset win over catalog", () => {
    const catalog = pollenCatalog();
    const piece: FramePiece = {
      id: "p1",
      typeId: "pollen",
      x: 1,
      y: 2,
      radius: 1.41,
      visualAsset: "seasons/biobuzz_2026/pieces/pollen.glb",
    };
    almostEqual(resolvePieceRadiusIn(piece, catalog), 1.41);
    assert(resolvePieceVisualAsset(piece, catalog) === piece.visualAsset, "piece visualAsset is preferred");
  }],
  ["manifest diameter fills radius when preset catalog is absent", () => {
    const catalog = buildPieceCatalog(frame(), {
      pieces: { pollen: { visualAsset: "seasons/biobuzz_2026/pieces/pollen.glb", measuredDiameterIn: 2.8 } },
    });
    const piece: FramePiece = { id: "p1", typeId: "pollen", x: 0, y: 0 };
    almostEqual(resolvePieceRadiusIn(piece, catalog), 1.4);
  }],
  ["field cache token comes from runtime manifest/source hash", () => {
    const sha = "05b35961c7df847741031f00fda73ddd068537f809e92a11b1cd59a94bcc8331";
    assert(explicitCadCacheToken(frame({ cadSourceSha256: sha })) === sha, "cadSourceSha256");
    assert(explicitCadCacheToken(frame({ cadAssetVersion: "rebuild-9" })) === "rebuild-9", "cadAssetVersion");
    assert(cadCacheTokenFromManifest({ field: { sha256: sha } }) === sha, "manifest field.sha256");
    assert(
      cadCacheTokenFromManifest({ generatorVersion: "1.1.0", field: { sha256: sha } }) === `${sha}:1.1.0`,
      "derived assets include generator version",
    );
    const url = fieldAssetUrl("seasons/biobuzz_2026/field.glb", sha);
    assert(url.includes(`v=${encodeURIComponent(sha)}`), "cache query uses hash");
    assert(!url.includes("am-5850e"), "hard-coded am-5850e must not appear");
  }],
  ["cad_manifest path is explicit or sibling of the field GLB", () => {
    assert(
      resolveCadManifestPath(frame({ cadManifest: "seasons/biobuzz_2026/cad_manifest.json" })) ===
        "seasons/biobuzz_2026/cad_manifest.json",
      "explicit cadManifest",
    );
    assert(siblingCadManifestPath(BIOBUZZ_GLB) === "seasons/biobuzz_2026/cad_manifest.json", "sibling of field.glb");
    assert(resolveCadManifestPath(frame({ backgroundAsset: BIOBUZZ_GLB })) === "seasons/biobuzz_2026/cad_manifest.json", "from background");
    assert(siblingCadManifestPath("seasons/other/field.png") === null, "non-GLB backgrounds do not guess a manifest");
  }],
  ["hive heuristic still stamps BIOBUZZ background for old frames", () => {
    const old = frame({
      elements: [{ id: "red_cell_up", type: "goal", pose: { x: 0, y: 0 }, shape: { kind: "aabb" } }],
    });
    assert(resolveBackgroundAsset(old) === BIOBUZZ_GLB, "hive heuristic");
  }],
  ["yawDeg wins over headingDeg for robot visual offset", () => {
    almostEqual(visualOffsetYawDeg({ yawDeg: 90, headingDeg: 12 }), 90);
    almostEqual(visualOffsetYawDeg({ headingDeg: 45 }), 45);
    almostEqual(visualOffsetYawDeg({ yawDeg: 0, headingDeg: 45 }), 0);
    almostEqual(visualOffsetYawDeg({}), 0);
  }],
  ["piece pose uses FTC inches and optional Y-up quaternion", () => {
    const catalog = pollenCatalog();
    const posed = pieceThreePose({ id: "p1", typeId: "pollen", x: 10, y: 4, z: 1.4, headingDeg: 90 }, catalog);
    assert(posed.position[0] === 10 && posed.position[1] === 1.4 && posed.position[2] === -4, "FTC to Three");
    almostEqual(posed.euler[1], Math.PI / 2);
    assert(pieceThreeQuaternion({ id: "p", x: 0, y: 0, qw: 1, qx: 0, qy: 0, qz: 0 })?.[3] === 1, "identity wxyz");
    assert(pieceThreeQuaternion({ id: "p", x: 0, y: 0, qw: 0, qx: 0, qy: 0, qz: 0 }) === null, "zero quat ignored");
    almostEqual(pieceThreeEulerRad({ id: "p", x: 0, y: 0, pitchDeg: 30 })[0], (30 * Math.PI) / 180);
  }],
  ["Y-up quaternion is preferred over headingDeg when present", () => {
    const catalog = pollenCatalog();
    const posed = pieceThreePose(
      { id: "p1", typeId: "pollen", x: 0, y: 0, z: 1.4, headingDeg: 90, qw: 1, qx: 0, qy: 0, qz: 0 },
      catalog,
    );
    assert(posed.quaternion?.[3] === 1 && posed.quaternion[0] === 0, "wxyz identity -> xyzw w last");
    almostEqual(posed.euler[1], Math.PI / 2);
  }],
  ["old replay with hive elements still has no typed piece CAD", () => {
    const old = frame({
      backgroundAsset: BIOBUZZ_GLB,
      cadManifest: "seasons/biobuzz_2026/cad_manifest.json",
      cadSourceSha256: "05b35961c7df847741031f00fda73ddd068537f809e92a11b1cd59a94bcc8331",
      pieces: [{ id: "legacy", x: 1, y: 2, color: "Y" }],
    });
    const catalog = buildPieceCatalog(old);
    const piece = old.pieces[0];
    assert(!usesCadPieceVisual(piece, catalog, old.backgroundAsset), "typeless piece stays primitive");
    assert(resolvePieceRadiusIn(piece, catalog) === LEGACY_PIECE_RADIUS_IN, "2.5 in fallback");
  }],
  ["typeId plus field GLB infers official piece path before catalog arrives", () => {
    const inferred = inferPieceVisualAsset("pollen", BIOBUZZ_GLB);
    assert(inferred === "seasons/biobuzz_2026/pieces/pollen.glb", "inferred pollen path");
    const piece: FramePiece = { id: "p1", typeId: "pollen", x: 0, y: 0 };
    assert(resolvePieceVisualAsset(piece, buildPieceCatalog(frame()), BIOBUZZ_GLB) === inferred, "infer from background");
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
  throw new Error(`${failed} cadAssets test(s) failed`);
}
console.log(`${tests.length} passed`);
