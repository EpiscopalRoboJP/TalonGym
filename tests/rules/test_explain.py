from talongym.rules.engine import RuleContext, RuleEngine


def test_emit_explain_appends_action_payload():
    engine = RuleEngine(
        {
            "scoreChannels": {"trueScore": "true_score"},
            "accumulators": [{"id": "true_score", "init": 0}],
            "nodes": [
                {
                    "id": "say_hello",
                    "enabledPhases": ["AUTO"],
                    "trigger": {"kind": "alwaysTick"},
                    "actions": [{"kind": "emitExplain", "message": "hello from rules"}],
                    "maxFires": 1,
                }
            ],
        }
    )
    ctx = RuleContext(
        phase="AUTO",
        events=[],
        volume_occupancy={},
        prev_occupancy={},
        accumulators=engine.init_accumulators(),
        match_vars={},
        gate_state={},
        queues={},
        queue_caps={},
        actor_id="red_0",
        actor_volumes=set(),
        fire_counts={},
    )
    explains, delta = engine.evaluate(ctx)
    assert delta == 0
    assert any(e.get("explain") == "hello from rules" for e in explains)
