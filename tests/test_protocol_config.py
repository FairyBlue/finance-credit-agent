"""Frozen human protocol choices must survive the public config loader."""

from pathlib import Path

import exchange_calendars as xcals
import yaml

from src.agents.config import load_experiment_config
from src.rewards import verify_calibration_document

ROOT = Path(__file__).resolve().parents[1]


def _base_config() -> dict:
    return load_experiment_config(ROOT / "configs/base.yaml")


def test_public_event_regimes_cover_every_oos_session_exactly_once() -> None:
    config = _base_config()
    regimes = config["regimes"]

    assert regimes["rule"] == "public_event_anchored_calendar_v1"
    assert regimes["calendar"] == "XNYS"
    assert regimes["timezone"] == "America/New_York"
    assert regimes["boundary_convention"] == "inclusive_complete_sessions"
    assert regimes["return_assignment"] == "ending_session"
    assert regimes["no_oos_outcome_confirmation"] is True
    assert regimes["anchors"] == ["2025-04-02", "2025-04-09"]

    expected_segments = [
        {
            "label": "pre_tariff_announcement",
            "start": "2025-03-03",
            "end": "2025-04-01",
            "sessions": 22,
        },
        {
            "label": "tariff_crash",
            "start": "2025-04-02",
            "end": "2025-04-08",
            "sessions": 5,
            "descriptive_only": True,
            "significance_claims_allowed": False,
        },
        {
            "label": "post_tariff_pause",
            "start": "2025-04-09",
            "end": "2025-06-30",
            "sessions": 56,
        },
    ]
    assert regimes["segments"] == expected_segments

    calendar = xcals.get_calendar(regimes["calendar"])
    covered_sessions: list[str] = []
    for segment in regimes["segments"]:
        sessions = calendar.sessions_in_range(segment["start"], segment["end"])
        rendered = [session.strftime("%Y-%m-%d") for session in sessions]
        assert len(rendered) == segment["sessions"]
        covered_sessions.extend(rendered)

    test_window = config["data"]["windows"]["test"]
    expected_sessions = [
        session.strftime("%Y-%m-%d")
        for session in calendar.sessions_in_range(test_window["start"], test_window["end"])
    ]
    assert covered_sessions == expected_sessions
    assert len(covered_sessions) == len(set(covered_sessions)) == 83


def test_t2_v2_is_preregistered_without_disabling_the_legacy_blocking_gate() -> None:
    config = _base_config()
    evaluation = config["evaluation"]

    legacy = evaluation["shifted_label_probe"]
    assert legacy["implementation_status"] == "legacy_blocking_gate_pending_v2"
    assert legacy["require_before_main_training"] is True
    assert legacy["metric"] == "oos_direction_accuracy"

    probe = evaluation["shifted_label_probe_v2"]
    assert probe["implementation_status"] == "preregistered_pending_implementation"
    assert probe["source_policy"] == {
        "allowed_source": "history",
        "fit_split": "train",
        "decision_split": "valid",
        "stockbench_oos_access": "forbidden",
    }
    assert probe["sample_policy"] == {
        "unit": "stock_session",
        "pooling": "pooled_across_stocks",
        "bootstrap_cluster": "session",
    }
    assert probe["features"] == {
        "frozen": True,
        "availability": "causal_as_of_session_close",
        "names": [
            "stock_return_1d",
            "stock_return_mean_20d",
            "stock_return_volatility_20d",
            "cross_sectional_dispersion_1d",
            "cross_sectional_positive_fraction_1d",
        ],
    }
    assert probe["label"] == {
        "kind": "per_stock_direction",
        "aligned_horizon_sessions": 1,
        "shifted_horizon_sessions": 2,
        "shift_sessions": 1,
    }
    assert probe["metric"] == "balanced_accuracy"
    assert probe["random_baseline"] == 0.5
    assert probe["bootstrap"] == {
        "method": "session_clustered_moving_block",
        "block_size_sessions": 5,
        "samples": 2000,
        "seed": 1729,
        "alpha": 0.05,
    }
    assert probe["detection_rule"] == "one_sided_lower_bound_above_random_baseline"
    assert probe["positive_controls"] == {
        "persistence": "ephemeral_not_persisted",
        "strong": {
            "feature": "exact_future_direction",
            "flip_probability": 0.0,
        },
        "weak": {
            "feature": "future_direction_with_independent_deterministic_flips",
            "flip_probability": 0.4,
            "approximate_balanced_accuracy": 0.6,
            "flip_generator": {
                "kind": "sha256_threshold_by_session_symbol_v1",
                "seed": 1729,
                "input_format": "seed<TAB>YYYY-MM-DD<TAB>SYMBOL<LF>",
            },
        },
    }
    assert probe["pass_criteria"] == {
        "strong_positive_control_detected": True,
        "weak_positive_control_detected": True,
        "real_pipeline_significantly_detected": False,
    }
    assert probe["weak_control_miss_policy"] == "record_probe_power_boundary"
    assert probe["real_pipeline_non_detection_semantics"] == (
        "no_significant_evidence_not_proof_of_absence"
    )
    assert probe["require_before_main_training"] is True


def test_reward_limits_use_the_approved_soft_threshold_semantics() -> None:
    reward = _base_config()["reward"]

    assert reward["threshold_semantics"] == {
        "enforcement": "soft_reward_penalty_not_hard_trade_cap",
        "position_measure": "maximum_single_name_close_weight",
        "turnover_period": "daily_session",
        "turnover_accounting": "double_sided",
        "turnover_limit_1p00_one_way_rotation_approx": 0.5,
    }


def test_base_reward_values_are_bound_to_the_pending_train_only_freeze_table() -> None:
    config = _base_config()
    reward = config["reward"]
    table_path = ROOT / "configs/reward_calibration.pending.yaml"
    table = yaml.safe_load(table_path.read_text(encoding="utf-8"))
    keys = (
        "epsilon",
        "risk_lambda",
        "drawdown_coefficient",
        "turnover_coefficient",
        "turnover_limit",
        "position_limit",
    )

    assert {key: reward[key] for key in keys} == {key: table["reward"][key] for key in keys}
    verify_calibration_document(table)
    assert table["status"] in {"pending_human_signature", "signed"}
    assert table["source"]["data_version"] == config["data"]["sources"]["history"]["version"]
    assert reward["calibration"] == {
        "method": "train_only_deterministic_proxy_v1",
        "table": "configs/reward_calibration.pending.yaml",
        "freeze_table_hash_format": table["hashes"]["freeze_table_hash_format"],
        "freeze_table_sha256": table["hashes"]["freeze_table_sha256"],
        "history_manifest_sha256": table["hashes"]["history_manifest_sha256"],
    }
