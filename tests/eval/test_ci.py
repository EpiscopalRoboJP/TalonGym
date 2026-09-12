from talongym.eval.harness import bootstrap_ci


def test_bootstrap_ci_contains_mean():
    report = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=200)
    assert report["lo"] <= report["mean"] <= report["hi"]
    assert report["p10"] <= report["median"] <= report["p90"]
