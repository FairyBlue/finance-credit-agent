import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_policy_transfer_readout.py"
spec = importlib.util.spec_from_file_location("transfer_readout", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "state.json").write_text(json.dumps({"status": "COMPLETE"}))
    for cell_index, cell in enumerate(module.CELLS):
        directory = tmp_path / cell
        directory.mkdir()
        rows = []
        for group, episodes in module.GROUPS.items():
            for episode_index, episode in enumerate(episodes):
                for mode, lanes in module.MODES.items():
                    for lane_index, lane in enumerate(lanes):
                        net = 0.01 + cell_index * 0.005 + episode_index * 0.001
                        if mode == "sample":
                            net += 0.01 * (lane_index + 1)
                        rows.append(
                            {
                                "id": f"full-{episode}-{lane}",
                                "context_id": episode,
                                "sampling": mode,
                                "context": {"group": group},
                                "status": "COMPLETE",
                                "requests": 20,
                                "invalid": 0,
                                "takeover_net_return": net,
                                "terminal_metrics": {
                                    "raw_net_return": net,
                                    "terminal_score": net * 10,
                                    "transaction_cost": 100,
                                    "benchmark_return": 0.03,
                                },
                                "final_state": {
                                    "portfolio_equity": 1000 * (1 + net),
                                    "cash": 100,
                                    "positions": {},
                                },
                                "model_fills": [],
                                "steps_including_prefix": [],
                            }
                        )
        (directory / "result.json").write_text(json.dumps({"status": "OBSERVED", "grid": rows}))
    return tmp_path


def test_fixed_grid_means_and_paired_differences(run):
    d = module.build(run)
    assert d["status"] == "COMPLETE" and d["complete"] == 240
    groups = d["cells"]["sft0"]["groups"]
    assert groups["training_windows"]["greedy"]["net_return_pct"] == pytest.approx(1.35)
    assert groups["training_windows"]["sample"]["net_return_pct"] == pytest.approx(3.85)
    assert groups["evaluation_windows"]["sample"]["net_return_pct"] == pytest.approx(3.65)
    for paired in d["paired_vs_sft"]:
        expected = module.CELLS.index(paired["cell"]) * 0.5
        assert paired["mean_return_delta_pp"] == pytest.approx(expected)


def test_missing_sibling_does_not_shrink_fixed_denominator(run):
    path = run / "m1/result.json"
    d = json.loads(path.read_text())
    d["grid"] = [r for r in d["grid"] if r["id"] != "full-formal-train-0113-sample0"]
    path.write_text(json.dumps(d))
    result = module.build(run)
    assert result["complete"] == 239 and result["status"] == "INCOMPLETE"
    assert result["cells"]["m1"]["groups"]["evaluation_windows"]["sample"]["net_return_pct"] is None
    paired = [
        r
        for r in result["paired_vs_sft"]
        if r["cell"] == "m1" and r["group"] == "evaluation_windows" and r["sampling"] == "sample"
    ][0]
    assert paired["mean_return_delta_pp"] is None


def test_all_rows_without_completed_controller_are_not_final(run):
    (run / "state.json").write_text('{"status": "RUNNING"}')
    d = module.build(run)
    assert d["complete"] == 240 and d["status"] == "INCOMPLETE"


@pytest.mark.parametrize("change", ["duplicate", "wrong_group", "unfinished", "return_mismatch"])
def test_invalid_evidence_cannot_be_final(run, change):
    path = run / "m2_eta025/result.json"
    d = json.loads(path.read_text())
    if change == "duplicate":
        d["grid"].append(d["grid"][0])
    elif change == "wrong_group":
        d["grid"][0]["context"]["group"] = "evaluation_windows"
    elif change == "unfinished":
        d["grid"][0]["status"] = "INCOMPLETE"
    else:
        d["grid"][0]["terminal_metrics"]["raw_net_return"] += 0.1
    path.write_text(json.dumps(d))
    assert module.build(run)["status"] == "INCOMPLETE"
