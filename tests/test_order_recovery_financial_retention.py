"""CPU/read-only legacy checks plus explicitly synthetic consumer-contract fixtures."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "financial_retention", ROOT / "scripts/diagnostics/order_recovery_financial_retention.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
OLD_PATH = ROOT / module.DEFAULT_OLD


@pytest.fixture(scope="module")
def old():
    if not (OLD_PATH / "raw_audit.json").exists():
        pytest.skip("Historical paired financial integration evidence is unavailable")
    evidence = module.Evidence()
    side = module.load_side(evidence, OLD_PATH, new=False)
    module.check_request_seeds(side)
    evidence.unchanged()
    return side


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def financial_grid(old, changed=None):
    new = copy.deepcopy(old)
    if changed:
        changed(new)
    pairs = module.pair_population(old, new)
    schema = set(old["rows"][0]["terminal_metrics"])
    fill_schema = set(old["rows"][0]["final_ledger"]["fills"][0])
    return [
        {
            "id": a["id"],
            "case_id": a["case_id"],
            "case_kind": a["case_kind"],
            "model_identity": a["model_identity"],
            "sampling": a["sampling"],
            "old": module.finance(a, case, schema, fill_schema),
            "new": module.finance(b, case, schema, fill_schema),
        }
        for a, b, case in pairs
    ]


def synthetic_new_layout(tmp_path, old):
    """Fake audit metadata tests the reader only, never represents new model evidence."""
    directory = tmp_path / "synthetic_consumer_fixture"
    full = directory / "full"
    manifest = copy.deepcopy(old["manifest"])
    manifest["models"] = {"sft": copy.deepcopy(manifest["models"]["sft"])}
    manifest["models"]["sft"]["model_path"] += "-SYNTHETIC-TEST"
    write(full / "manifest.json", manifest)
    result = {
        "profile": module.FULL_PROFILE,
        "status": "OBSERVED",
        "model_key": "sft",
        "full_episode_evaluation": True,
        "start_mode": "fullreset",
        "grid": old["rows"],
    }
    short = {
        "status": "OBSERVED",
        "grid": [{"id": f"synthetic-short-{i}", "status": "ADVANCED"} for i in range(320)],
    }
    for phase, value in (("full", result), ("short", short)):
        worker = directory / phase / "workers/sft"
        write(worker / "state.json", value)
        write(worker / "result.json", value)
        write(worker / "config.json", {"synthetic_consumer_test": True})
        for name in ("requests.jsonl", "raw.jsonl", "submissions.jsonl"):
            (worker / name).touch()
    shutil.copyfile(OLD_PATH / "workers/sft/requests.jsonl", full / "workers/sft/requests.jsonl")
    write(directory / "short/manifest.json", {"synthetic_consumer_test": True})
    parent = {
        "profile": "common_sft_order_recovery_acceptance_v1",
        "mode": "sft",
        "models": manifest["models"],
        "full_cases": manifest["cases"],
        "observation_profile": module.OBSERVATION_PROFILE,
        "action_protocol": module.PROTOCOL,
        "enable_thinking": False,
        "budget": {
            "short_trajectories": 320,
            "full_trajectories": 80,
            "planned_trajectories": 400,
            "full_request_cap": 256,
            "recovery_followup_cap": 4,
            "prompt_cap": 8192,
            "response_cap": 4096,
            "automatic_retry": False,
            "automatic_actions": False,
        },
        "phase_manifest_sha256": {
            p: module.sha(directory / p / "manifest.json") for p in ("short", "full")
        },
    }
    write(directory / "manifest.json", parent)
    for name in ("state.json", "result.json"):
        write(directory / "workers/sft" / name, {"status": "OBSERVED"})
    sources = [
        directory / "manifest.json",
        directory / "short/manifest.json",
        full / "manifest.json",
        *module.worker_paths(directory / "short"),
        *module.worker_paths(full),
        directory / "workers/sft/state.json",
        directory / "workers/sft/result.json",
    ]
    audit = {
        "profile": "common_sft_order_recovery_raw_audit_v1",
        "status": "PASS",
        "mode": "sft",
        "synthetic_consumer_test": True,
        "audit_scope": "complete_population",
        "manifest_sha256": module.sha(directory / "manifest.json"),
        "acceptance_pass": True,
        "short_trajectories": 320,
        "full_trajectories": 80,
        "trajectories": 400,
        "workers": {
            "sft": {
                "short": {"trajectories": 320},
                "full": {
                    "trajectories": 80,
                    "request_count": sum(r["requests"] for r in old["rows"]),
                    "statuses": dict(Counter(r["status"] for r in old["rows"])),
                },
            }
        },
        "source_sha256": {str(p.resolve()): module.sha(p) for p in sources},
    }
    write(directory / "raw_audit.json", audit)
    write(
        directory / "readout.json",
        {
            "status": "OBSERVED",
            "behavioral_gate": "PASS",
            "acceptance": "PASS",
            "manifest_sha256": audit["manifest_sha256"],
            "raw_audit_status": "PASS",
            "raw_audit_sha256": module.sha(directory / "raw_audit.json"),
            "short_grid": short["grid"],
            "full_grid": old["rows"],
        },
    )
    return directory


def test_actual_legacy_evidence(old):
    assert len(old["rows"]) == 80
    normal = module.comparison(financial_grid(old), case_kind="normal")
    assert normal["old"]["complete"] == 20
    assert normal["old"]["with_buy_fill_trajectories"] == 20
    assert normal["old"]["with_sell_fill_trajectories"] == 20
    assert normal["old"]["actual_observed_activity_totals"]["buy_fills"] == 219
    assert normal["old"]["actual_observed_activity_totals"]["sell_fills"] == 160
    assert normal["old"]["financial_means"]["full_window_net_return"] == pytest.approx(
        0.012944987226752325
    )


def test_complete_layout_and_exclusive_output(tmp_path, old):
    new = synthetic_new_layout(tmp_path, old)
    result = module.compare(OLD_PATH, new)
    assert result["paired_trajectories"] == 80
    assert result["groups"]["normal2"]["planned_pairs"] == 20
    assert all(v == 0 for v in result["groups"]["all8"]["mean_delta_new_minus_old"].values())
    destination = new / "financial_retention.json"
    module.write_exclusive(destination, result)
    original = destination.read_bytes()
    with pytest.raises(FileExistsError):
        module.write_exclusive(destination, result)
    assert destination.read_bytes() == original
    # The consumer fails closed if evidence changes after its raw-audit receipt.
    (new / "full/workers/sft/raw.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="Source hash changed"):
        module.compare(OLD_PATH, new)


def test_incomplete_is_retained_without_financial_mean(old):
    def mutate(new):
        row = next(r for r in new["rows"] if r["case_kind"] == "normal")
        row.update(status="INCOMPLETE", failure="RESPONSE_INCOMPLETE")
        del row["terminal_metrics"], row["full_window_net_return"], row["final_state"]
        row["final_ledger"]["fills"] = row["final_ledger"]["fills"][:2]
        row["final_ledger"]["session_index"] = 3

    grid = financial_grid(old, mutate)
    result = module.comparison(grid, case_kind="normal")
    assert len(grid) == 80 and result["planned_pairs"] == 20
    assert result["new"]["complete"] == 19
    assert result["new"]["activity_horizon"] == "mixed_incomplete_horizons"
    assert all(v is None for v in result["new"]["financial_means"].values())
    assert all(v is None for v in result["mean_delta_new_minus_old"].values())
    assert next(r["new"]["failure_reason"] for r in grid if r["new"]["status"] == "INCOMPLETE")


def test_all_hold_finance_remains_visible(old):
    def mutate(new):
        for row in new["rows"]:
            row["final_ledger"].update(fills=[], cash=1_000_000, positions={})
            row["final_state"]["portfolio_equity"] = 1_000_000
            row["full_window_net_return"] = 0
            metrics = row["terminal_metrics"]
            metrics.update(
                raw_net_return=0, commission=0, slippage_cost=0, transaction_cost=0, turnover=0
            )
            metrics["terminal_score"] = -metrics["benchmark_return"] / metrics["score_denominator"]

    result = module.comparison(financial_grid(old, mutate), case_kind="normal")
    assert result["new"]["complete_no_fill_trajectories"] == 20
    assert result["new"]["with_buy_fill_trajectories"] == 0
    assert result["new"]["with_sell_fill_trajectories"] == 0
    assert result["mean_delta_new_minus_old"]["buy_fills"] == -10.95
    assert result["mean_delta_new_minus_old"]["net_return_percentage_points"] < 0


@pytest.mark.parametrize(
    "tamper", ["missing", "duplicate", "environment", "sampling", "initial_state"]
)
def test_pairing_contract_rejects_tampering(old, tamper):
    new = copy.deepcopy(old)
    if tamper == "missing":
        new["rows"].pop()
    elif tamper == "duplicate":
        new["rows"][-1] = new["rows"][0]
    elif tamper == "environment":
        new["manifest"]["cases"][0]["env_kwargs"]["commission_bps"] += 1
    elif tamper == "sampling":
        new["rows"][0]["sampling"] = "sampled"
    else:
        new["rows"][0]["initial_state"]["cash"] += 1
    with pytest.raises(ValueError):
        module.pair_population(old, new)


@pytest.mark.parametrize("tamper", ["cost", "fill_schema", "incomplete_return", "nonterminal"])
def test_financial_contract_rejects_tampering(old, tamper):
    row = copy.deepcopy(old["rows"][0])
    schema = set(row["terminal_metrics"])
    fill_schema = set(row["final_ledger"]["fills"][0])
    if tamper == "cost":
        row["terminal_metrics"]["transaction_cost"] += 1
    elif tamper == "fill_schema":
        row["final_ledger"]["fills"][0]["total_cost"] = 0
    elif tamper == "incomplete_return":
        row["status"] = "INCOMPLETE"
    else:
        row["final_ledger"]["session_index"] -= 1
    with pytest.raises(ValueError):
        module.finance(row, old["manifest"]["cases"][0], schema, fill_schema)
