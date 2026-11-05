from src.agents.financial_mc_selection import PrefixSelector, public_features


def test_selection_depends_only_on_prefix_and_is_reproducible():
    selectors = [PrefixSelector("priority", 42, 2, 1_000_000, 16384) for _ in range(2)]
    records = [s.consider("task/rep0/call3", "task/rep0", [], 3) for s in selectors]
    assert records[0] == records[1]
    assert records[0]["pre_action_selection"]
    assert not records[0]["parent_action_used"]
    assert not records[0]["parent_outcome_used"]


def test_selection_reserves_a_complete_independent_pair_and_one_per_parent():
    selector = PrefixSelector("uniform", 42, 10, 327680, 16384)
    chosen = None
    for index in range(100):
        record = selector.consider(f"task{index}/call0", f"task{index}", [], 0)
        if record["selected"]:
            chosen = record
            break
    assert chosen is not None
    assert chosen["reserved_auxiliary_tokens"] == 2 * 10 * 16384
    assert selector.consider("next/call0", "next", [], 0)["block_reason"] == (
        "PAIR_BUDGET_RESERVATION"
    )
    assert selector.consider(
        chosen["trajectory_key"] + "/call1", chosen["trajectory_key"], [], 1
    )["block_reason"] == "ONE_ANCHOR_PER_PARENT"


def test_priority_observes_past_evidence_not_next_action():
    messages = [
        {"role": "assistant", "content": '{"name":"read_document","arguments":{}}'},
        {"role": "tool", "content": '{"ok":true,"title":"ABC annual report 2010"}'},
    ]
    features = public_features(messages, 3, 10)
    assert features["stage"] == "evidence_to_calculation"
    uniform = PrefixSelector("uniform", 42, 1, 1_000_000, 16384)
    priority = PrefixSelector("priority", 42, 1, 1_000_000, 16384)
    a = uniform.consider("request", "parent", messages, 3)
    b = priority.consider("request", "parent", messages, 3)
    assert a["selection_draw"] == b["selection_draw"]
    assert b["selection_probability"] > a["selection_probability"]
    assert priority.consider("other", "other", [], 0)["nominal_probability"] > 0
