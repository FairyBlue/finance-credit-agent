import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_sampling_profile_readout.py"
SPEC = importlib.util.spec_from_file_location("sampling_profile_readout", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.fixture
def run(tmp_path):
    (tmp_path / "state.json").write_text('{"status":"ALL_CELLS_OBSERVED"}')
    for model_index, model in enumerate(module.MODELS):
        directory = tmp_path / model
        directory.mkdir()
        rows = []
        for group, episodes in module.GROUPS.items():
            for episode in episodes:
                index = module.EPISODES.index(episode)
                for sibling in range(4):
                    for profile in module.PROFILES:
                        official = profile == "qwen_recommended"
                        net = 0.01 + model_index * 0.005 + index * 0.001 + sibling * 0.002
                        net += (0.004 + index * 0.0001) if official else 0
                        invalid = 1 if official else 2
                        rows.append(
                            {
                                "id": f"full-{episode}-{profile}-sample{sibling}",
                                "context_id": episode,
                                "profile": profile,
                                "sampling": "sample",
                                "horizon": "full",
                                "state_kind": "cash",
                                "context": {"group": group},
                                "identity": {
                                    "step": 1,
                                    "sample_index": index,
                                    "rollout_n": sibling,
                                    "validate": False,
                                },
                                "status": "COMPLETE",
                                "requests": 20 + invalid,
                                "invalid": invalid,
                                "actions": {"buy": 10, "sell": 8, "hold": 2, "invalid": invalid},
                                "model_fills": [
                                    {
                                        "symbol": "GS",
                                        "side": "buy",
                                        "commission": 2,
                                        "slippage_cost": 1,
                                    }
                                ],
                                "takeover_net_return": net,
                                "terminal_metrics": {
                                    "raw_net_return": net,
                                    "terminal_score": net * 10,
                                    "transaction_cost": 100 + index * 10 + sibling - 5 * official,
                                },
                                "final_state": {
                                    "portfolio_equity": 1000 * (1 + net),
                                    "cash": 200,
                                    "positions": {"GS": {"weight": 0.2}},
                                },
                            }
                        )
        (directory / "result.json").write_text(json.dumps({"status": "OBSERVED", "grid": rows}))
    return tmp_path


def change(run, model, callback):
    path = run / model / "result.json"
    value = json.loads(path.read_text())
    callback(value["grid"])
    path.write_text(json.dumps(value))


def test_fixed_means_deltas_and_behavior_counters(run):
    result = module.build(run)
    assert result["status"] == "COMPLETE" and result["final_ready"]
    assert result["complete"] == result["terminal_observed"] == result["planned"] == 384
    assert not result["problems"]
    sft = result["cells"]["sft0"]["groups"]
    assert sft["training_windows"]["current"]["planned"] == 32
    assert sft["evaluation_windows"]["current"]["planned"] == 16
    assert sft["training_windows"]["current"]["net_return_pct"] == pytest.approx(1.65)
    assert sft["evaluation_windows"]["current"]["net_return_pct"] == pytest.approx(2.25)
    assert sft["training_windows"]["current"]["buys_observed"] == 320
    assert sft["training_windows"]["current"]["holds_observed"] == 64
    assert sft["training_windows"]["current"]["fills_observed"] == 32
    assert sft["training_windows"]["current"]["unique_filled_symbols_observed"] == 1
    for point in result["profile_comparisons"]:
        expected = 0.435 if point["group"] == "training_windows" else 0.495
        assert point["mean_net_return_pct_delta"] == pytest.approx(expected)
        assert point["mean_cost_USD_delta"] == pytest.approx(-5)
        assert point["mean_invalid_requests_per_trajectory_delta"] == pytest.approx(-1)
        assert point["mean_invalid_request_rate_pct_delta"] == pytest.approx(100 / 21 - 200 / 22)
    for point in result["rl_vs_sft"]:
        assert point["mean_net_return_pct_delta"] == pytest.approx(
            module.MODELS.index(point["model"]) * 0.5
        )
    assert len(list(csv.DictReader(io.StringIO(module.summary_csv(result))))) == 16


def test_one_missing_sibling_withholds_fixed_window_and_group(run):
    change(run, "sft0", lambda rows: rows.pop(0))
    result = module.build(run)
    assert result["complete"] == 383 and not result["all_cells_observed"]
    assert (
        result["cells"]["sft0"]["groups"]["training_windows"]["current"]["net_return_pct"] is None
    )
    assert (
        result["cells"]["sft0"]["groups"]["evaluation_windows"]["current"]["net_return_pct"]
        is not None
    )
    comparison = result["profile_comparisons"][0]
    assert comparison["planned_window_pairs"] == 8
    assert comparison["usable_net_return_pct_window_pairs"] == 7
    assert comparison["mean_net_return_pct_delta"] is None
    assert comparison["mean_invalid_requests_per_trajectory_delta"] is None
    assert all(
        p["mean_net_return_pct_delta"] is None
        for p in result["rl_vs_sft"]
        if p["group"] == "training_windows" and p["profile"] == "current"
    )
    exported = list(csv.DictReader(io.StringIO(module.summary_csv(result))))
    assert exported[0]["net_return_pct"] == ""


def test_native_cap_is_observed_but_never_a_terminal_return(run):
    def cap(rows):
        row = rows[0]
        row.update(status="INCOMPLETE", failure="REQUEST_CAP", requests=128, invalid=108)
        row["actions"]["invalid"] = 108
        for key in ("takeover_net_return", "terminal_metrics", "final_state"):
            row.pop(key)

    change(run, "sft0", cap)
    (run / "state.json").write_text('{"status":"ALL_CELLS_OBSERVED_WITH_INCOMPLETE"}')
    result = module.build(run)
    assert result["status"] == "INCOMPLETE" and result["final_ready"]
    assert result["all_cells_observed"] and result["terminal_observed"] == 384
    assert result["complete"] == 383 and result["incomplete"] == 1
    group = result["cells"]["sft0"]["groups"]["training_windows"]["current"]
    assert group["net_return_pct"] is None and group["cost_USD"] is None
    assert group["failure_counts"] == {"REQUEST_CAP": 1}
    comparison = result["profile_comparisons"][0]
    assert comparison["mean_net_return_pct_delta"] is None
    assert comparison["mean_invalid_requests_per_trajectory_delta"] == pytest.approx(-4.3125)
    module.write(run, result)
    assert (run / "FINAL_RESULTS.md").exists()


@pytest.mark.parametrize(
    "kind", ["wrong_profile", "wrong_sibling", "duplicate", "nonfinite", "cap_with_fake_outcome"]
)
def test_bad_evidence_never_becomes_final(run, kind):
    def alter(rows):
        row = rows[0]
        if kind == "wrong_profile":
            row["profile"] = "qwen_recommended"
        elif kind == "wrong_sibling":
            row["identity"]["rollout_n"] = 3
        elif kind == "duplicate":
            rows.append(row.copy())
        elif kind == "nonfinite":
            row["terminal_metrics"]["terminal_score"] = float("nan")
        else:
            row.update(status="INCOMPLETE", failure="RESPONSE_INCOMPLETE")

    change(run, "m2_eta025", alter)
    result = module.build(run)
    assert result["status"] == "INCOMPLETE" and not result["final_ready"]
    assert result["problems"]
    if kind != "duplicate":
        assert (
            result["cells"]["m2_eta025"]["groups"]["training_windows"]["current"]["net_return_pct"]
            is None
        )


def test_live_state_and_complete_grid_do_not_finalize_running_controller(run):
    path = run / "sft0/result.json"
    path.rename(run / "sft0/state.json")
    (run / "state.json").write_text('{"status":"RUNNING"}')
    result = module.build(run)
    assert result["terminal_observed"] == 384 and result["all_cells_observed"]
    assert not result["final_ready"] and result["status"] == "INCOMPLETE"
    module.write(run, result)
    assert (run / "LIVE_STATUS.md").exists() and not (run / "FINAL_RESULTS.md").exists()


def test_unstarted_grid_is_listed_but_not_observed(run):
    def reset(rows):
        for row in rows:
            row.update(status="UNSTARTED", requests=0, invalid=0, actions={}, model_fills=[])
            for key in ("takeover_net_return", "terminal_metrics", "final_state"):
                row.pop(key)

    change(run, "sft0", reset)
    result = module.build(run)
    group = result["cells"]["sft0"]["groups"]["training_windows"]["current"]
    assert group["listed"] == 32 and group["observed"] == group["terminal_observed"] == 0
    assert group["net_return_pct"] is None
