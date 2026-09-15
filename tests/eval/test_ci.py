from talongym.eval.harness import bootstrap_ci, run_trials
from talongym.training.curriculum import objective_value


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
