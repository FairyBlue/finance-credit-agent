"""Pre-action quota, visible target stage, complete-pair admission and wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.financial_learning_collect import prepare_quota, quota_environment, run
from src.agents.financial_mc_selection import (
    QUOTA_PATH_ENV,
    QUOTA_SHA_ENV,
    PrefixSelector,
    make_parent_quota,
    target_report_features,
)


def public_messages(title="ABC annual report 2010, excerpt", ok=True, calculated=False):
    messages = [
        dict(role="system", content="tools"),
        dict(
            role="user", content=json.dumps(dict(condition=dict(company="ABC", report_year="2010")))
        ),
        dict(
            role="assistant",
            content='<tool_call>{"name":"read_document","arguments":{}}</tool_call>',
        ),
        dict(role="tool", content=json.dumps(dict(ok=ok, title=title))),
    ]
    if calculated:
        messages.extend(
            [
                dict(
                    role="assistant",
                    content='{"name":"calculate","arguments":{"expression":"1/2"}}',
                ),
                dict(role="tool", content='{"ok":false,"error":"calculation failed"}'),
            ]
        )
    return messages


def wrapper_args(tmp_path, **changes):
    path = tmp_path / "batch.jsonl"
    path.write_text(
        "".join(json.dumps(dict(task_id=f"opaque-{i}", split="train")) + "\n" for i in range(8))
    )
    args = argparse.Namespace(
        data=path,
        output=tmp_path / "collect",
        base_model=tmp_path / "base",
        adapter=tmp_path / "adapter",
        seed=43,
        mc_selection_seed=None,
        repetitions=4,
        temperature=1.0,
        max_context_tokens=16384,
        max_response_tokens=512,
        token_budget=3_000_000,
        mc_selection_budget=2_000_000,
        mc_max_anchors=8,
        prepare_only=True,
    )
    for key, value in changes.items():
        setattr(args, key, value)
    return args


def selector_from(args):
    path, digest, manifest = prepare_quota(args)
    selector = PrefixSelector(
        "target_quota",
        43,
        8,
        args.mc_selection_budget,
        16384,
        10,
        quota_manifest=path,
        quota_sha256=digest,
    )
    return selector, manifest, path, digest


def test_frozen_parent_quota_is_reproducible_independent_and_limited(tmp_path):
    args = wrapper_args(tmp_path)
    first = prepare_quota(args)
    second = prepare_quota(args)
    assert first == second
    manifest = first[2]
    assert len(manifest["parent_ids"]) == 32
    assert len(set(manifest["selected_parent_ids"])) == 8
    assert manifest["parent_quota_probability"] == 0.25
    assert manifest["selected_before_any_parent_action"]
    changed_data = make_parent_quota(manifest["task_ids"], 4, 43, 8, args.data, "f" * 64)
    assert changed_data["selected_parent_ids"] == manifest["selected_parent_ids"]
    assert not (args.output / "collection_started.json").exists()


def test_target_prefix_uses_correct_public_report_before_calculation_return(tmp_path):
    selector, manifest, _, _ = selector_from(wrapper_args(tmp_path))
    parent = manifest["selected_parent_ids"][0]
    for call, title in enumerate(("ABCD annual report 2010", "ABC annual report 2009")):
        row = selector.consider(f"{parent}/call{call}", parent, public_messages(title), call)
        assert row["block_reason"] == "TARGET_REPORT_NOT_READ"
    row = selector.consider(f"{parent}/call3", parent, public_messages(), 3)
    assert row["selected"]
    assert row["selection_probability"] == 0.25
    assert row["conditional_stage_selection_probability"] == 1
    assert row["required_pair_reservation"] == 2 * 7 * 16384
    assert row["pre_action_selection"] and not row["parent_outcome_used"]
    assert not row["parent_action_used"] and not row["auxiliary_outcome_used"]
    assert selector.consider(f"{parent}/call4", parent, public_messages(), 4)["block_reason"] == (
        "ONE_ANCHOR_PER_PARENT"
    )


def test_failed_or_only_catalog_read_does_not_select_and_calculate_return_blocks():
    assert target_report_features(public_messages(ok=False), 3, 10)["target_report_reads"] == 0
    messages = public_messages()
    messages[-1]["content"] = json.dumps(
        dict(ok=True, results=[dict(title="ABC annual report 2010")])
    )
    assert target_report_features(messages, 3, 10)["target_report_reads"] == 0
    assert (
        target_report_features(public_messages(calculated=True), 5, 10)["calculation_returns"] == 1
    )


def test_complete_pair_budget_denial_cannot_retry_a_later_cheaper_prefix(tmp_path):
    args = wrapper_args(tmp_path, mc_selection_budget=1)
    selector, manifest, _, _ = selector_from(args)
    parent = manifest["selected_parent_ids"][0]
    denied = selector.consider(f"{parent}/call3", parent, public_messages(), 3)
    assert denied["block_reason"] == "PAIR_BUDGET_RESERVATION"
    assert denied["reserved_auxiliary_tokens"] == 0
    later = selector.consider(f"{parent}/call9", parent, public_messages(), 9)
    assert later["block_reason"] == "FIRST_ELIGIBLE_PREFIX_ALREADY_CONSIDERED"
    other = next(p for p in manifest["parent_ids"] if p not in manifest["selected_parent_ids"])
    assert selector.consider(f"{other}/call3", other, public_messages(), 3)["block_reason"] == (
        "PARENT_NOT_IN_PRE_ROLLOUT_QUOTA"
    )


def test_quota_manifest_or_data_changes_fail_closed(tmp_path):
    args = wrapper_args(tmp_path)
    _, _, path, digest = selector_from(args)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="manifest SHA256"):
        PrefixSelector(
            "target_quota", 43, 8, 2_000_000, 16384, quota_manifest=path, quota_sha256=digest
        )
    path.write_text(path.read_text()[:-1])
    args.data.write_text(args.data.read_text() + "\n")
    with pytest.raises(ValueError, match="data SHA256"):
        PrefixSelector(
            "target_quota", 43, 8, 2_000_000, 16384, quota_manifest=path, quota_sha256=digest
        )


def test_wrapper_prepare_only_does_not_import_or_start_gpu(tmp_path, monkeypatch):
    fake = SimpleNamespace(run=lambda args: pytest.fail("collector must not execute"))
    monkeypatch.setitem(sys.modules, "src.agents.mandate_repair_collect", fake)
    args = wrapper_args(tmp_path)
    result = run(args)
    assert result["status"] == "PREPARED_ONLY"


def test_wrapper_passes_complete_args_and_restores_scoped_config(tmp_path, monkeypatch):
    args = wrapper_args(tmp_path, prepare_only=False)
    prior_path, prior_sha = "previous-path", "previous-sha"
    monkeypatch.setenv(QUOTA_PATH_ENV, prior_path)
    monkeypatch.setenv(QUOTA_SHA_ENV, prior_sha)

    def fake_collect(options):
        assert options.mc_selection == "target_quota"
        assert options.environment == "financial_evidence"
        assert options.decode_mode == "native" and options.temperature == 1
        selector = PrefixSelector("target_quota", 43, 8, 2_000_000, 16384)
        assert len(selector.eligible_parents) == 8
        assert (
            hashlib.sha256(Path(os.environ[QUOTA_PATH_ENV]).read_bytes()).hexdigest()
            == (os.environ[QUOTA_SHA_ENV])
        )
        return dict(status="FAKE_COMPLETE")

    monkeypatch.setitem(
        sys.modules, "src.agents.mandate_repair_collect", SimpleNamespace(run=fake_collect)
    )
    assert run(args)["status"] == "FAKE_COMPLETE"
    assert os.environ[QUOTA_PATH_ENV] == prior_path and os.environ[QUOTA_SHA_ENV] == prior_sha
    with pytest.raises(RuntimeError), quota_environment("temporary", "digest"):
        raise RuntimeError("restore config even if worker fails")
    assert os.environ[QUOTA_PATH_ENV] == prior_path and os.environ[QUOTA_SHA_ENV] == prior_sha
