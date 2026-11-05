"""Behavioral tests for the isolated same-prefix gate candidate."""

from dataclasses import replace
from itertools import permutations

import pytest

from src.rewards.same_prefix_gate import (
    COMPLETE_LABEL,
    IncompleteSamePrefixGroup,
    SamePrefixGateSlot,
    compute_same_prefix_gate,
    normalize_economic_order_key,
)


def slots(keys, scores=(5.0, 1.0, -1.0, 3.0)):
    return [
        SamePrefixGateSlot(
            anchor_id="fixed-anchor",
            slot_id=i,
            prompt_sha256="a" * 64,
            financial_state_sha256="b" * 64,
            policy_snapshot_sha256="c" * 64,
            economic_order_key=key,
            score=score,
            label_status=COMPLETE_LABEL,
        )
        for i, (key, score) in enumerate(zip(keys, scores, strict=True))
    ]


SAME_FAMILY = (
    '["buy","A",0.1]',
    '["buy","A",0.2]',
    '["buy","B",0.1]',
    '["buy","B",0.2]',
)
MIXED = ('["buy","A",0.1]', '["buy","A",0.2]', '["sell","B",0]', '["hold"]')


def test_only_order_gate_preserves_same_family_order_differences():
    result = compute_same_prefix_gate(slots(SAME_FAMILY))
    assert result.mean_score == 2.0
    assert result.ungated_plain == (3.0, -1.0, -3.0, 1.0)
    assert result.family_gate is False and result.order_gate is True
    assert result.zero == result.family_gate_plain == (0.0, 0.0, 0.0, 0.0)
    assert result.order_gate_plain == result.ungated_plain
    assert result.scaled_eta025.order_gate_plain == (0.75, -0.25, -0.75, 0.25)
    assert result.scaled_eta025.family_gate_plain == result.zero
    assert result.eta == 0.25


def test_family_and_order_gates_agree_when_original_gate_is_open():
    result = compute_same_prefix_gate(slots(MIXED))
    assert result.family_gate is True and result.order_gate is True
    assert result.family_gate_plain == result.order_gate_plain == (3.0, -1.0, -3.0, 1.0)


def test_duplicate_actions_are_never_pooled():
    result = compute_same_prefix_gate(slots((MIXED[0], MIXED[0], MIXED[2], MIXED[3])))
    # The first two scalar scores remain distinct even though their order keys match.
    assert result.family_gate_plain[:2] == (3.0, -1.0)
    assert result.order_gate_plain == result.ungated_plain


def test_normalized_same_order_closes_both_gates_without_changing_raw_centering():
    keys = (
        '["sell"," msft ",0]',
        '["sell","MSFT",0.0]',
        '["sell","msft",-0.0]',
        '["sell","MSFT",0e0]',
    )
    result = compute_same_prefix_gate(slots(keys))
    assert result.normalized_order_keys == (("sell", "MSFT", 0.0),) * 4
    assert result.family_gate is result.order_gate is False
    assert result.family_gate_plain == result.order_gate_plain == result.zero
    assert result.ungated_plain == (3.0, -1.0, -3.0, 1.0)


def test_hold_and_abstain_are_distinct_families_and_orders():
    result = compute_same_prefix_gate(slots(('["hold"]', '["abstain"]') * 2))
    assert result.family_gate and result.order_gate
    assert result.family_gate_plain == result.order_gate_plain == result.ungated_plain


@pytest.mark.parametrize("keys", (SAME_FAMILY, MIXED, ('["hold"]',) * 4))
def test_equal_scores_never_invent_a_signal(keys):
    result = compute_same_prefix_gate(slots(keys, (7.0,) * 4))
    assert (
        result.ungated_plain == result.family_gate_plain == result.order_gate_plain == result.zero
    )


def test_permutations_follow_input_rows_and_keep_gate_identity():
    original = slots(MIXED)
    expected = compute_same_prefix_gate(original)
    for permutation in permutations(range(4)):
        result = compute_same_prefix_gate([original[i] for i in permutation])
        assert result.ungated_plain == tuple(expected.ungated_plain[i] for i in permutation)
        assert result.order_gate_plain == tuple(expected.order_gate_plain[i] for i in permutation)
        assert result.normalized_order_keys == tuple(
            expected.normalized_order_keys[i] for i in permutation
        )
        assert result.family_gate == expected.family_gate


@pytest.mark.parametrize("shift", (-128.0, -2.5, 0.0, 19.0, 1024.0))
def test_common_hold_baseline_or_additive_shift_does_not_change_any_signal(shift):
    original = slots(SAME_FAMILY)
    before = compute_same_prefix_gate(original)
    after = compute_same_prefix_gate([replace(row, score=row.score + shift) for row in original])
    for name in ("zero", "ungated_plain", "family_gate_plain", "order_gate_plain", "scaled_eta025"):
        assert getattr(after, name) == getattr(before, name)
    assert after.mean_score == before.mean_score + shift


@pytest.mark.parametrize("bad", (None, float("nan"), float("inf"), -float("inf"), True, "1"))
def test_complete_labels_require_finite_numerical_scores(bad):
    rows = slots(MIXED)
    rows[2] = replace(rows[2], score=bad)
    with pytest.raises(ValueError, match="finite number"):
        compute_same_prefix_gate(rows)


@pytest.mark.parametrize(
    "key",
    (
        None,
        "not JSON",
        "{}",
        "[]",
        '["BUY","A",0.1]',
        '["buy"," ",0.1]',
        '["buy","A",NaN]',
        '["buy","A",Infinity]',
        '["buy","A",true]',
        '["buy","A","0.2"]',
        '["buy","A",-0.1]',
        '["sell","A",1.1]',
        '["buy","A"]',
        '["hold","extra"]',
        '["abstain","reason"]',
        '["compute"]',
    ),
)
def test_malformed_or_non_economic_order_keys_are_rejected(key):
    with pytest.raises(ValueError):
        normalize_economic_order_key(key)


@pytest.mark.parametrize(
    "field,new",
    (
        ("anchor_id", "another-anchor"),
        ("prompt_sha256", "d" * 64),
        ("financial_state_sha256", "e" * 64),
        ("policy_snapshot_sha256", "f" * 64),
    ),
)
def test_mixing_prefix_or_policy_is_an_integrity_error(field, new):
    rows = slots(MIXED)
    rows[1] = replace(rows[1], **{field: new})
    with pytest.raises(ValueError, match="must share") as caught:
        compute_same_prefix_gate(rows)
    assert not isinstance(caught.value, IncompleteSamePrefixGroup)


@pytest.mark.parametrize("bad", (True, -1, 4, 1.0))
def test_slot_id_domain_is_strict(bad):
    rows = slots(MIXED)
    rows[0] = replace(rows[0], slot_id=bad)
    with pytest.raises(ValueError, match="slot IDs"):
        compute_same_prefix_gate(rows)


def test_duplicate_slot_is_integrity_error_not_partial_support():
    rows = slots(MIXED)
    rows[3] = replace(rows[3], slot_id=0)
    with pytest.raises(ValueError, match="unique") as caught:
        compute_same_prefix_gate(rows)
    assert not isinstance(caught.value, IncompleteSamePrefixGroup)


@pytest.mark.parametrize("count", (0, 1, 2, 3))
def test_partial_population_never_returns_a_recentered_result(count):
    rows = (slots(MIXED) * 2)[:count]
    with pytest.raises(IncompleteSamePrefixGroup):
        compute_same_prefix_gate(rows)


def test_extra_slot_is_an_integrity_error_not_missing_support():
    rows = [*slots(MIXED), slots(MIXED)[0]]
    with pytest.raises(ValueError, match="extra slots") as caught:
        compute_same_prefix_gate(rows)
    assert not isinstance(caught.value, IncompleteSamePrefixGroup)


def test_one_missing_label_rejects_the_whole_original_group():
    rows = slots(MIXED)
    rows[2] = replace(rows[2], score=None, economic_order_key=None, label_status="NA")
    with pytest.raises(IncompleteSamePrefixGroup, match="partial"):
        compute_same_prefix_gate(rows)


def test_missing_label_does_not_hide_a_wrong_identity():
    rows = slots(MIXED)
    rows[2] = replace(rows[2], score=None, economic_order_key=None, label_status="NA")
    rows[1] = replace(rows[1], prompt_sha256="d" * 64)
    with pytest.raises(ValueError, match="must share") as caught:
        compute_same_prefix_gate(rows)
    assert not isinstance(caught.value, IncompleteSamePrefixGroup)


@pytest.mark.parametrize(
    "field,value", (("anchor_id", " "), ("prompt_sha256", "BAD"), ("label_status", "COMPLETE"))
)
def test_invalid_identity_and_label_metadata_fail_closed(field, value):
    rows = slots(MIXED)
    rows[0] = replace(rows[0], **{field: value})
    with pytest.raises(ValueError):
        compute_same_prefix_gate(rows)


def test_numerical_overflow_does_not_return_infinite_credit():
    with pytest.raises(ValueError, match="finite numerical domain"):
        compute_same_prefix_gate(slots(MIXED, (1e308, 1e308, 1e308, 1e308)))
