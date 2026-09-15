from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn

from talongym import paths
from talongym.eval.harness import run_trials
from talongym.export.roadrunner import export_from_replay
from talongym.presets.loader import load_bundle
from talongym.training.policies import scripted_auto
from talongym.training.ppo import record_policy_episode, train_ppo

app = typer.Typer(help="TalonGym — FTC Autonomous trainer")


@app.command()
def lab(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve the Lab UI and API."""
    uvicorn.run("talongym.api.app:app", host=host, port=port, reload=False)


@app.command()
def train(
    steps: int | None = typer.Option(None, "--steps", help="Total env steps. Default 8192, or the easy budget with --easy"),
    n_envs: int | None = typer.Option(None, "--n-envs"),
    allow_scripted: bool = False,
    algo: str | None = typer.Option(None, "--algo", help="recurrent_ppo (default) or experimental rllib_ppo toy"),
    easy: bool = typer.Option(False, "--easy", help="Autodetect hardware and use the season's easy run config"),
    training: str | None = typer.Option(None, "--training", help="Training preset id for this invocation"),
) -> None:
    from talongym.presets.defaults import get_defaults
    from talongym.presets.loader import load_preset
    from talongym.training.compute import easy_training_id, resolve_training

    training_id = training
    if easy:
        training_id = easy_training_id(training or get_defaults().get("trainingId"))
        doc = load_preset("training", training_id)
        block = doc.get("presets") or {}
        bundle = load_bundle(block.get("fieldId"), block.get("robotId"), block.get("scoringId"), training_id)
    elif training_id:
        bundle = load_bundle(training_id=training_id)
    else:
        bundle = load_bundle()
    resolved = resolve_training(bundle.training, easy=easy)
    if bundle.training is not None:
        bundle.training = {**bundle.training, **resolved}
    n = n_envs if n_envs is not None else int(resolved.get("nEnvs") or 4)
    algo_name = algo or str((resolved.get("algorithm") or {}).get("name") or "recurrent_ppo")
    if easy and algo is None and algo_name == "rllib_ppo":
        algo_name = "recurrent_ppo"
    total = steps if steps is not None else (int((resolved.get("budget") or {}).get("totalEnvSteps") or 8192) if easy else 8192)
    typer.echo(
        f"compute={resolved.get('computeProfile')} nEnvs={n} training={(bundle.training or {}).get('id')} steps={total}"
    )
    if algo_name == "rllib_ppo":
        from talongym.training.rllib import train_rllib

        typer.echo("rllib_ppo is a one-shot toy trainer; not a production scale path")
        result = train_rllib(total_steps=total, log=typer.echo)
    else:
        result = train_ppo(
            bundle=bundle,
            total_steps=total,
            n_envs=n,
            log=typer.echo,
            allow_scripted=allow_scripted,
        )
    typer.echo(f"algo={result.get('algo')} steps={result.get('steps')} ckpt={result.get('checkpoint')}")


@app.command()
def detect() -> None:
    """Print CPU/RAM/CUDA and the recommended compute profile."""
    from talongym.presets.defaults import get_defaults
    from talongym.training.compute import describe_compute

    typer.echo(json.dumps(describe_compute(get_defaults().get("trainingId")), indent=2))


@app.command()
def evaluate(
    trials: int = 32,
    checkpoint: Path | None = typer.Option(None, "--checkpoint", help="Optional SB3 zip; default is the scripted AUTO"),
) -> None:
    policy = scripted_auto
    if checkpoint is not None:
        from talongym.training.ppo import load_trained_policy

        policy = load_trained_policy(str(checkpoint))
    report = run_trials(trials, policy, record_best=False)
    typer.echo(
        f"mean={report['mean']:.2f} 95% CI [{report['lo']:.2f}, {report['hi']:.2f}] "
        f"p10={report['p10']:.2f} n={report['nTrials']} eligible={report['bestLabelEligible']}"
    )


@app.command()
def replay(seed: int = 0) -> None:
    frames = record_policy_episode(scripted_auto, seed=seed)
    paths.VAR_DIR.mkdir(exist_ok=True)
    path = paths.VAR_DIR / "last_replay.java"
    path.write_text(export_from_replay(frames), encoding="utf-8")
    score = frames[-1]["trueScore"] if frames else 0
    typer.echo(f"frames={len(frames)} trueScore={score} export={path}")


@app.command("defaults")
def defaults_cmd(
    training: str | None = typer.Option(None, "--training", help="Training preset id to become the default bundle"),
    field: str | None = typer.Option(None, "--field", help="Field preset id"),
    robot: str | None = typer.Option(None, "--robot", help="Robot preset id"),
    scoring: str | None = typer.Option(None, "--scoring", help="Scoring preset id"),
) -> None:
    from talongym.presets.defaults import describe_defaults, set_defaults
    from talongym.presets.loader import PresetError

    if any([training, field, robot, scoring]):
        try:
            set_defaults(field_id=field, robot_id=robot, scoring_id=scoring, training_id=training)
        except PresetError as exc:
            raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(describe_defaults(), indent=2))


@app.command("preset")
def preset_lint() -> None:
    from talongym.presets.loader import list_presets, load_preset

    for kind in ("field", "robot", "scoring", "training"):
        for item in list_presets(kind):
            load_preset(kind, item["id"])
            typer.echo(f"ok {kind} {item['id']}")


@app.command()
def calibrate(log: Path, out: Path | None = None) -> None:
    from talongym.calibrate import fit_file
    from talongym.presets.loader import load_preset

    robot = load_preset("robot", "mecanum_meepmeep_defaults")
    overlay = fit_file(log, robot)
    dest = out or (paths.VAR_DIR / "robot_overlay.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(overlay, indent=2), encoding="utf-8")
    typer.echo(f"rmse={overlay.get('calibration', {}).get('rmse')} overlay={dest}")


@app.command("validate-3d")
def validate_3d(steps: int = 40) -> None:
    from talongym.sim.mujoco_backend import MujocoValidationBackend, available, pose_rmse
    from talongym.sim.physics import Body, Planar2DBackend, WorldStep, perimeter_walls

    if not available():
        raise typer.BadParameter("MuJoCo extra missing; pip install -e '.[mujoco]'")
    walls = perimeter_walls(72, 72)
    planar = Planar2DBackend(72, 72)
    mj = MujocoValidationBackend(72, 72)
    bot_a = Body("red_0", 0, -40, 1.57, vx=10, vy=5)
    bot_b = Body("red_0", 0, -40, 1.57, vx=10, vy=5)
    pa, pb = [], []
    for _ in range(steps):
        planar.step_world(WorldStep([bot_a], [], [], walls), 0.02)
        mj.step_world(WorldStep([bot_b], [], [], walls), 0.02)
        pa.append((bot_a.x, bot_a.y))
        pb.append((bot_b.x, bot_b.y))
    typer.echo(f"rmse={pose_rmse(pa, pb):.4f} engine=mujoco")


@app.command("import-field-cad")
def import_field_cad_cmd(
    field: Path | None = typer.Option(None, "--field", help="Field preset JSON"),
    page: str = typer.Option("https://ftc-resources.firstinspires.org/ftc/archive/2027/field", "--page"),
    step: Path | None = typer.Option(None, "--step", help="Official field STEP/STP (copied to var/cad/, not committed)"),
    url: str | None = typer.Option(None, "--url", help="Official field STEP URL (default: FIRST binary endpoint field-cad-step)"),
    tol_linear: float | None = typer.Option(None, "--tol-linear", help="OpenCASCADE linear deflection in STEP source units"),
    skip_pieces: bool = typer.Option(False, "--skip-pieces", help="Do not import POLLEN/NECTAR STEP files"),
    pieces_only: bool = typer.Option(False, "--pieces-only", help="Import scoring-element STEP files only"),
    verify: bool = typer.Option(False, "--verify", help="Check committed CAD manifest and derived files, then exit"),
) -> None:
    from talongym.assets.import_field_cad import CadImportError, import_field_cad, verify_season_cad

    try:
        if verify:
            result = verify_season_cad(field_path=field)
        else:
            result = import_field_cad(
                field_path=field,
                page_url=page,
                step_path=step,
                step_url=url,
                tol_linear=tol_linear,
                include_pieces=not skip_pieces,
                skip_field=pieces_only,
            )
    except CadImportError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, indent=2))


@app.command("import-piece-cad")
def import_piece_cad_cmd(
    piece_type: str = typer.Option(..., "--type", help="pollen, nectar_red, or nectar_blue"),
    step: Path | None = typer.Option(None, "--step", help="Local official piece STEP (copied to var/cad/, not committed)"),
    url: str | None = typer.Option(None, "--url", help="Override AndyMark STEP URL"),
    field: Path | None = typer.Option(None, "--field", help="Field preset JSON (chooses destination season folder)"),
    year_hint: str = typer.Option("2026", "--year-hint"),
) -> None:
    from talongym.assets.import_field_cad import CadImportError, asset_dir_for, load_field_json
    from talongym.assets.import_piece_cad import import_piece_cad

    try:
        dest = asset_dir_for(load_field_json(field), year_hint)
        result = import_piece_cad(piece_type, dest, step_path=step, url=url)
    except (CadImportError, KeyError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps({k: v for k, v in result.items() if k != "manifest"}, indent=2, default=str))


@app.command("import-robot-cad")
def import_robot_cad_cmd(
    robot: str = typer.Option(..., "--robot", help="Robot preset id (assets stored under var/assets/robots/<id>/)"),
    cad: Path = typer.Option(..., "--file", help="GLB, glTF, STL, OBJ, or STEP"),
) -> None:
    from talongym.assets.import_robot_cad import RobotCadError, import_robot_cad

    try:
        result = import_robot_cad(cad, robot)
    except RobotCadError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2))


@app.command()
def distill(steps: int = 256) -> None:
    from talongym.training.distill import distill_from_scripted

    result = distill_from_scripted(n_steps=steps)
    typer.echo(f"onnx={result['path']} n={result['n']}")
