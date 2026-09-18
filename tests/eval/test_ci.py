from talongym.eval.harness import bootstrap_ci, policy_spec_for, run_trials
from talongym.training.curriculum import objective_value
from talongym.training.policies import scripted_auto


def test_bootstrap_ci_contains_mean():
    report = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=200)
    assert report["lo"] <= report["mean"] <= report["hi"]
    assert report["p10"] <= report["median"] <= report["p90"]


def test_objective_value_mean_p10_lcb():
    report = {"mean": 10.0, "p10": 2.0, "lo": 4.0}
    assert objective_value(report, "mean_true_score") == 10.0
    assert objective_value(report, "p10_true_score") == 2.0
    assert objective_value(report, "lcb_true_score") == 4.0


def test_run_trials_stores_requested_objective(monkeypatch):
    monkeypatch.setattr(
        "talongym.eval.harness._run_one",
        lambda *a, **k: {
            "score": 5.0,
            "collision": False,
            "collision_time_s": 0.0,
            "first_contact_s": None,
            "entered_restricted": False,
            "frames": [],
        },
    )
    report = run_trials(3, lambda o, i: None, record_best=False, objective="p10_true_score")
    assert report["objective"] == "p10_true_score"
    assert report["objectiveValue"] == report["p10"]
    assert report["evalWorkers"] == 1


def test_unpicklable_policy_does_not_start_a_process_pool(monkeypatch):
    monkeypatch.setenv("TALONGYM_EVAL_WORKERS", "4")

    class BoomPool:
        def __init__(self, *a, **k):
            raise AssertionError("lambdas must stay sequential")

    monkeypatch.setattr("talongym.eval.harness.ProcessPoolExecutor", BoomPool)
    monkeypatch.setattr(
        "talongym.eval.harness._run_one",
        lambda *a, **k: {
            "score": 1.0,
            "collision": False,
            "collision_time_s": 0.0,
            "first_contact_s": None,
            "entered_restricted": False,
            "frames": [],
            "seed": 0,
        },
    )
    report = run_trials(3, lambda o, i: None, record_best=False)
    assert report["nTrials"] == 3
    assert report["evalWorkers"] == 1


def test_scripted_auto_uses_process_pool(monkeypatch):
    monkeypatch.setenv("TALONGYM_EVAL_WORKERS", "2")
    seen: list[int] = []

    class FakePool:
        def __init__(self, *a, **k):
            seen.append(int(k.get("max_workers") or a[0]))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def map(self, fn, jobs):
            return [
                {
                    "score": float(job["seed"]),
                    "collision": False,
                    "collision_time_s": 0.0,
                    "first_contact_s": None,
                    "entered_restricted": False,
                    "frames": [],
                    "seed": job["seed"],
                }
                for job in jobs
            ]

    monkeypatch.setattr("talongym.eval.harness.ProcessPoolExecutor", FakePool)
    report = run_trials(4, scripted_auto, record_best=False, seed0=10)
    assert seen == [2]
    assert report["evalWorkers"] == 2
    assert report["scores"] == [10.0, 11.0, 12.0, 13.0]


def test_policy_spec_for_scripted_and_checkpoint_path():
    assert policy_spec_for(scripted_auto) == {"kind": "scripted"}
    assert policy_spec_for(lambda o, i: None) is None
    adapter = type("A", (), {"path": "/tmp/best.zip"})()
    assert policy_spec_for(adapter) == {"kind": "checkpoint", "path": "/tmp/best.zip"}
