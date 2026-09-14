from talongym.presets.loader import list_presets, load_bundle, load_preset


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
