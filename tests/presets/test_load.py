from talongym.presets.loader import list_presets, load_bundle, load_preset, validate_document


def test_all_shipped_presets_validate():
    for kind in ("field", "robot", "scoring", "training"):
        items = list_presets(kind)
        assert items, kind
        for item in items:
            load_preset(kind, item["id"])


def test_biobuzz_bundle_loads():
    bundle = load_bundle()
    assert bundle.field["season"]["slug"] == "biobuzz"
    assert "mesh_field_collision" in bundle.field["requiredCapabilities"]
    assert bundle.scoring["provenance"]["verifyAgainstManual"] is True
    assert bundle.field["id"] == "biobuzz_2026_field_v1"
    assert bundle.scoring["id"] == "biobuzz_2026_scoring_v1"


def test_physical_actuators_action_tier_validates():
    doc = dict(load_preset("training", "biobuzz_auto_lightweight"))
    doc.pop("_kind", None)
    doc["actionTier"] = "physical_actuators"
    assert validate_document("training", doc) == []


def test_omitted_action_tier_is_valid():
    doc = dict(load_preset("training", "biobuzz_auto_lightweight"))
    doc.pop("_kind", None)
    doc.pop("actionTier", None)
    assert validate_document("training", doc) == []


def test_rp_proxy_probability_is_rejected():
    doc = dict(load_preset("training", "biobuzz_auto_lightweight"))
    doc.pop("_kind", None)
    doc["objective"] = "rp_proxy_probability"
    assert validate_document("training", doc)


def test_cloud_preset_is_recurrent_ppo():
    training = load_preset("training", "biobuzz_auto_cloud")
    assert training["algorithm"]["name"] == "recurrent_ppo"
    assert int(training["algorithm"].get("bcWarmupSteps") or 0) == 0
    assert training["nEnvs"] == 1024
