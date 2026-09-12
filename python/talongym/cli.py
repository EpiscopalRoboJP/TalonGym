from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn

from talongym.eval.harness import run_trials
from talongym.export.roadrunner import export_from_replay
from talongym.paths import VAR_DIR
from talongym.training.policies import scripted_auto
from talongym.training.ppo import record_policy_episode, train_ppo

app = typer.Typer(help="TalonGym — FTC Autonomous trainer")


@app.command()
def lab(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve the Lab UI and API."""
    uvicorn.run("talongym.api.app:app", host=host, port=port, reload=False)


@app.command()
def train(steps: int = 8192, n_envs: int = 4, allow_scripted: bool = False, algo: str = "recurrent_ppo") -> None:
    if algo == "rllib_ppo":
        from talongym.training.rllib import train_rllib

        result = train_rllib(total_steps=steps, log=typer.echo)
    else:
        result = train_ppo(total_steps=steps, n_envs=n_envs, log=typer.echo, allow_scripted=allow_scripted)
    typer.echo(f"algo={result.get('algo')} steps={result.get('steps')} ckpt={result.get('checkpoint')}")


@app.command()
def evaluate(trials: int = 32) -> None:
    report = run_trials(trials, scripted_auto, record_best=False)
    typer.echo(
        f"mean={report['mean']:.2f} 95% CI [{report['lo']:.2f}, {report['hi']:.2f}] "
        f"p10={report['p10']:.2f} n={report['nTrials']} eligible={report['bestLabelEligible']}"
    )


@app.command()
def replay(seed: int = 0) -> None:
    frames = record_policy_episode(scripted_auto, seed=seed)
    VAR_DIR.mkdir(exist_ok=True)
    path = VAR_DIR / "last_replay.java"
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
    dest = out or (VAR_DIR / "robot_overlay.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(overlay, indent=2), encoding="utf-8")
    typer.echo(f"rmse={overlay.get('calibration', {}).get('rmse')} overlay={dest}")


@app.command("validate-3d")
def validate_3d(steps: int = 40) -> None:
    from talongym.sim.mujoco_backend import MujocoValidationBackend, available, pose_rmse
    from talongym.sim.physics import Planar2DBackend, WorldStep, perimeter_walls
    from talongym.sim.physics import Body

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


@app.command()
def distill(steps: int = 256) -> None:
    from talongym.training.distill import distill_from_scripted

    result = distill_from_scripted(n_steps=steps)
    typer.echo(f"onnx={result['path']} n={result['n']}")
