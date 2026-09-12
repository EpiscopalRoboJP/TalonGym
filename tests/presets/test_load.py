from talongym.presets.loader import list_presets, load_bundle, load_preset, validate_document


def test_all_shipped_presets_validate():
    for kind in ("field", "robot", "scoring", "training"):
        items = list_presets(kind)
        assert items, kind
        for item in items:
            load_preset(kind, item["id"])


def test_decode_bundle_loads():
    bundle = load_bundle()
    assert bundle.field["season"]["slug"] == "decode"
    assert "planar_drive" in bundle.field["requiredCapabilities"]
    assert bundle.scoring["provenance"]["verifyAgainstManual"] is True


def test_into_the_deep_is_not_decode_shaped():
    field = load_preset("field", "into_the_deep_2024_field")
    types = {el["type"] for el in field["elements"]}
    assert "submersible" in types
    assert "obelisk" not in types
