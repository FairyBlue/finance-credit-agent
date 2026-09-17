#!/usr/bin/env python3
"""CPU-only reconstruction of observed Train credit signals; no policy counterfactual."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# This analysis constructs small CPU tensors only, never a model or CUDA tensor.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.agents.screening_runtime import unwhitened_step_gae  # noqa: E402
from src.rewards.sibling_credit import SiblingStep, compute_sibling_credit  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return math.fsum(values) / len(values) if values else None


def stats(values):
    average = mean(values)
    return {
        "count": len(values),
        "mean": average,
        "std_population": math.sqrt(mean([(v - average) ** 2 for v in values])),
        "min": min(values),
        "max": max(values),
    }


def sign(value, margin=0.0):
    return "positive" if value > margin else "negative" if value < -margin else "near_zero"


def transitions(rows, before, after, margin=0.0):
    return dict(Counter(f"{sign(r[before], margin)}->{sign(r[after], margin)}" for r in rows))


def observation_payload(interaction):
    messages = json.loads(interaction["observation"]["text"])
    for message in reversed(messages):
        content = message.get("content", "")
        match = re.search(r"<tool_response>.*?(\{.*\}).*?</tool_response>", content, re.S)
        if match:
            return json.loads(match.group(1))
    raise ValueError("no observed tool-response state")


def invalid_details(interaction):
    if not interaction["info"].get("invalid_action", False):
        return {"invalid_category": "legal", "error": ""}
    error = interaction["info"]["error"]
    response = interaction["response"]["text"]
    match = re.search(r"<tool_call>(.*?)</tool_call>", response, re.S)
    call = json.loads(match.group(1)) if match else {}
    args = call.get("arguments", {})
    symbol = args.get("symbol")
    target = args.get("target_weight")
    state = observation_payload(interaction)
    position = state.get("positions", {}).get(symbol, {})
    shares = position.get("shares", 0.0) if isinstance(position, dict) else position
    price = state.get("market", {}).get(symbol, {}).get("close")
    equity = state.get("portfolio_equity")
    weight = shares * price / equity if price is not None and equity else None
    if "buy target_weight" in error:
        category = "buy_target_not_above_current"
        assert weight is not None and target <= weight + 1e-8
    elif "sell target_weight" in error:
        category = "sell_zero_holding" if shares == 0 else "sell_target_not_below_current"
        assert weight is not None and target >= weight - 1e-8
    elif "outside the frozen episode universe" in error:
        category = "symbol_outside_universe"
    else:
        category = "other_invalid"
    return {
        "invalid_category": category,
        "error": error,
        "requested_tool": call.get("name"),
        "requested_symbol": symbol,
        "requested_target_weight": target,
        "observed_current_weight": weight,
        "observed_shares": shares,
    }


def summarize(rows):
    return {
        "steps": len(rows),
        "invalid_categories": dict(Counter(r["invalid_category"] for r in rows)),
        "reward_sign": dict(Counter(sign(r["assigned_reward"]) for r in rows)),
        "actual_A_sign": dict(Counter(sign(r["actual_A"]) for r in rows)),
        "reward_to_suffix_return": transitions(rows, "assigned_reward", "suffix_return"),
        "suffix_return_to_U": transitions(rows, "suffix_return", "reconstructed_U", 1e-5),
        "U_to_combined": transitions(rows, "reconstructed_U", "reconstructed_combined", 1e-5),
        "combined_to_actual_A": transitions(rows, "reconstructed_combined", "actual_A", 1e-5),
        "eta0_to_actual_A": transitions(rows, "same_batch_eta0_A", "actual_A", 1e-5),
        "actual_to_window_center_A": transitions(rows, "actual_A", "window_center_A", 1e-5),
        "actual_to_fixed_penalty_A": transitions(rows, "actual_A", "fixed_penalty_A", 1e-5),
        "fixed_penalty_A_sign": dict(Counter(sign(r["fixed_penalty_A"]) for r in rows)),
        "fixed_penalty_immediate_reward_sign": dict(
            Counter(sign(r["fixed_penalty_reward"]) for r in rows)
        ),
        "mean_abs_A_change_fixed_penalty": mean(
            [abs(r["fixed_penalty_A"] - r["actual_A"]) for r in rows]
        ),
        "positive_A_in_positive_mean_window": sum(
            r["actual_A"] > 0 and r["window_actual_A_mean"] > 0 for r in rows
        ),
        "nonzero_scaled_S": sum(r["scaled_S"] != 0 for r in rows),
        "mean_scaled_S": mean([r["scaled_S"] for r in rows]),
        "mean_abs_scaled_S": mean([abs(r["scaled_S"]) for r in rows]),
        "mean_actual_A_minus_eta0_A": mean([r["actual_A"] - r["same_batch_eta0_A"] for r in rows]),
        "mean_abs_actual_A_minus_eta0_A": mean(
            [abs(r["actual_A"] - r["same_batch_eta0_A"]) for r in rows]
        ),
        "actual_A_times_delta_positive_steps": sum(
            r["actual_A"] * r["delta_logp_mean"] > 0 for r in rows
        ),
    }


def variance_by_window(rows, key):
    """Population ANOVA, weighted by actual step counts, not equal windows."""
    overall = stats([r[key] for r in rows])
    groups = {
        episode: stats([r[key] for r in rows if r["episode_id"] == episode])
        for episode in sorted({r["episode_id"] for r in rows})
    }
    between = math.fsum(
        g["count"] / len(rows) * (g["mean"] - overall["mean"]) ** 2 for g in groups.values()
    )
    within = math.fsum(g["count"] / len(rows) * g["std_population"] ** 2 for g in groups.values())
    total = overall["std_population"] ** 2
    assert math.isclose(between + within, total, rel_tol=1e-10, abs_tol=1e-10)
    return {
        "total_variance": total,
        "between_window_variance": between,
        "within_window_variance": within,
        "between_window_fraction": between / total if total else None,
        "window_values": groups,
    }


def render_report(result):
    m1, m2 = result["arms"]["m1"], result["arms"]["m2_eta025"]
    lines = [
        "# Train 奖励—critic—GAE—M2 信号链路诊断",
        "",
        f"当前进展：{result['at_SGT']}。仅分析已完成 v2 的 128 条 Train 轨迹，未训练模型。",
        "",
        "主要结论：非法请求获得正 advantage 是共同学习信号中的明确弱点，"
        "本批不能归因于 M2 额外 credit。窗口差异确实影响整体归一化，"
        "但离线窗口中心化没有减少非法正 A 的总数。",
        "",
        "| 非法步骤读数 | M1 | M2 η=0.25 |",
        "|---|---:|---:|",
    ]
    for label, field in (
        ("非法步骤", lambda a: a["invalid"]["steps"]),
        ("实际正 A", lambda a: a["invalid"]["actual_A_sign"].get("positive", 0)),
        (
            "同批 η=0 时正 A",
            lambda a: sum(
                v for k, v in a["invalid"]["eta0_to_actual_A"].items() if k.startswith("positive->")
            ),
        ),
        (
            "离线窗口中心化后正 A",
            lambda a: sum(
                v
                for k, v in a["invalid"]["actual_to_window_center_A"].items()
                if k.endswith("->positive")
            ),
        ),
        (
            "离线 c=0.05 拒单罚后正 A",
            lambda a: a["invalid"]["fixed_penalty_A_sign"].get("positive", 0),
        ),
        ("非法步骤非零 M2 S", lambda a: a["invalid"]["nonzero_scaled_S"]),
    ):
        lines.append(f"| {label} | {field(m1)} | {field(m2)} |")
    lines += [
        "",
        "这里三种离线变化都固定已观察的轨迹和旧 critic，不能解释成新策略的收益预测，"
        "也没有重算一次训练后的 critic。",
        "",
        "## 共同信号与 M2 增量",
        "",
        "实际链路为 rₜ=Y/N，Gₜ≈Σₖ≥ₜrₖ，Uₜ=Gₜ−Vₜ，"
        "Aₜ=whiten(Uₜ+ηSₜ)，γ=λ=1。N 包括非法请求；"
        "非法请求不推进 session，但依然获得 terminal Y 的一份分配并进入 GAE/critic/actor loss。",
        "",
        "M1 非法直接 reward 正 68 个，实际正 A 34 个；M2 为 61 和 38 个。"
        "M1 的 G→U 阶段有 38 个正转负、5 个负转正，随后 batch centering 有 "
        "16 个正转负、15 个负转正；M2 相应为 25/5 和 15/12。"
        "这是固定算术链路的符号分解，不是对 critic 的因果归责。",
        "",
        "M2 95 个非法步骤 S 全为 0；同批 η=0 后 95/95 的 A 符号保持一致。"
        "合法推进步骤有 852/1280（66.56%）获得非零 S；相比同批 η=0，"
        "33 个合法步骤改变符号（19 负→正、14 正→负）。scaled std(S)/std(U) "
        "逐轮为约 0.055–0.111，说明 M2 信号存在，但没有解释非法正 A。",
        "",
        "非法类别为：M1 买入目标不高于当前权重 92 次、零持仓卖出 17 次、"
        "域外股票 2 次；M2 对应 77/16/2。这属于接口合法性与状态使用问题，"
        "无需人为规定买卖方向或选股策略。",
        "",
        "## critic 尺度、窗口差异与局部性",
        "",
        f"按各轮实际步骤和 U 方差加权，窗口间成分占 M1 "
        f"{100 * m1['between_window_U_variance_fraction_across_rounds']:.2f}%、M2 "
        f"{100 * m2['between_window_U_variance_fraction_across_rounds']:.2f}%。"
        "逐轮跨度约 0–60%，不能将全部信号都归因于窗口差异。"
        "非法正 A 中有 25/34 和 27/38 位于平均 A 为正的窗口；"
        "但窗口中心化后正 A 总数变为 39 和 38，不是直接的拒单修复。",
        "",
        "| 轮次 | M1 窗口间 U 方差占比 | M2 占比 | M2 scaled std(S)/std(U) |",
        "|---|---:|---:|---:|",
    ]
    for left, right in zip(m1["rounds"], m2["rounds"], strict=True):
        lines.append(
            f"| {left['policy_step']} | "
            f"{100 * left['U_variance_decomposition']['between_window_fraction']:.2f}% | "
            f"{100 * right['U_variance_decomposition']['between_window_fraction']:.2f}% | "
            f"{right['signal_std_ratio_S_over_U']:.3f} |"
        )
    lines += [
        "",
        "第 8 轮 M1 critic pre V 均值约 3.120，return 均值约 −1.132，"
        "pre MSE 约 21.323；M2 pre MSE 约 19.551。"
        "但 γ=λ=1 且之后全批中心化，V 的全批常数偏移会从 A 中消去；"
        "不能把大均值 bias 直接称为 actor 退化原因。"
        "旧 critic 对 return 的 explained variance 末两轮为负，"
        "说明它的逐步区分能力仍弱；同批更新也并非每轮改善 MSE，详见 JSON。",
        "",
        "Uniform 分配还带来结构性局部性限制：Gₜ≈Y(N−t)/N，"
        "同一终末结果会按步骤位置给出不同 return；非法步骤也进入这条后缀和。"
        "轨迹最多 30/29 步，非法步骤最多分走该轨迹 Y 的 33.33%/31.03%。"
        "GSPO 此处按实际响应步骤等权，较长轨迹相对 8 条轨迹等权的权重最高约 "
        "1.35/1.34 倍。这不等于已证明多犯错更有利，因为符号、GAE、whitening "
        "和 optimizer 共同参与。M2 S 使用整条轨迹 Y 的同 session 动作族均值，"
        "也不是该单次动作的局部因果收益。",
        "",
        "## 只推荐一个下一次小试",
        "",
        "优先做 M1/M2 共同、固定 c=0.05 的显式拒单罚：r′ₜ=Y/N−0.05·1_invalid，"
        "保持评价 finance Y/净收益、M2 S 的 finance Y 来源、η=0.25、"
        "学习率和窗口归一化均不变。建议只跑固定 4 轮并沿用未 RL 的 SFT 基线，"
        "关注非法率/错误后恢复及原评价收益；不同时改其他训练条件。"
        "这是训练目标改为 Y−0.05×非法数的新共同设置，应新建合同和运行身份，"
        "本轮未实现或启动。",
        "",
        "固定已观察批次的单一离线情景仅使非法正 A 从 34→30、38→36，"
        "分别 4 和 2 个正转负、没有反向翻转。仍有 61/111 和 57/95 个非法步骤的 "
        "r′为正；平均绝对 A 改变量约 0.0644/0.0501。"
        "这是轻量候选，不保证非法 A 全负，也不保证拒单率或收益改善，不能替代新训练验证。",
        "",
        "## 证据和重跑",
        "",
        "原始逐步 A、reward、请求与终末结果直接来自日志；逐步旧 V 未保存，"
        "U/V 使用 whitening 反推和 bf16 舍入，因此明确标为 reconstructed。"
        f"回算实际 A 最大误差为 {m1['max_reconstruction_A_error']:.3g}/"
        f"{m2['max_reconstruction_A_error']:.3g}，并核对原 GAE、critic V/return 汇总。"
        "这约束了近似重构，不证明每个 bf16 值唯一可恢复；逐步 post V 无法恢复。",
        "",
        "运行：.venv/bin/python scripts/diagnostics/m2_coverage_signal_analysis.py。",
        "",
        "- [逐步信号](step_signals.csv)：2766 行，含 observed A、反推 U/V、S、"
        "错误类别、η=0/窗口中心化/c=0.05 的同批离线信号。",
        "- [轨迹分配](trajectory_allocation.csv)：128 条完整 Train 轨迹。",
        "- [完整机器结果](signal_analysis.json)：16 轮统计、逐窗口 A 均值、"
        "critic pre/post 拟合、输入哈希和重构检查。",
        "- [GAE 与 whitening 源码](../../../src/agents/screening_runtime.py)、"
        "[M2 S 源码](../../../src/rewards/sibling_credit.py)。",
        "",
    ]
    return "\n".join(lines)


def analyze(run_dir, output_dir):
    import numpy as np
    import torch

    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("analysis outputs must be outside the immutable source run")
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {}

    def read_json(path):
        sources[str(path)] = sha(path)
        return json.loads(path.read_text())

    def read_jsonl(folder):
        result = []
        for path in sorted(folder.glob("*.jsonl")):
            raw = path.read_bytes()
            assert raw.endswith(b"\n"), path
            sources[str(path)] = hashlib.sha256(raw).hexdigest()
            result.extend(json.loads(line) for line in raw.splitlines())
        return result

    seal = read_json(run_dir / "source_identity.json")
    for relative in ("src/agents/screening_runtime.py", "src/rewards/sibling_credit.py"):
        assert sha(ROOT / relative) == sha(run_dir / "sources" / relative) == seal[relative]
        sources[str(ROOT / relative)] = sha(ROOT / relative)
    all_rows, all_trajectories, all_rounds = [], [], []
    result = {
        "schema_version": 1,
        "status": "PASS",
        "source_run": str(run_dir),
        "at_SGT": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "arms": {},
        "identification_limits": [
            "Actual A/rewards and terminal outcomes are logged per step; raw critic V is not.",
            "U is approximately inverted from A, whitening mean/std and recomputed S, "
            "with float rounding.",
            "Pre V is inferred from reward suffix return minus reconstructed U, then rounded "
            "to logged bf16 dtype; it is not a saved per-step critic observation.",
            "Round-trip GAE/whitening and logged critic statistics constrain this reconstruction "
            "but do not prove unique recovery of every bf16 value.",
            "No per-step post critic values can be recovered; post-fit comparisons use "
            "saved aggregate statistics only.",
            "Eta=0 keeps observed M2 trajectories/critic fixed: a signal-only offline ablation, "
            "not a policy-return counterfactual.",
            "Sign decomposition describes the fixed arithmetic path; it does not identify "
            "causal effects on policy behavior or returns.",
            "Window centering is a hypothetical offline arithmetic variant, not executed training. "
            "Differences between window means cannot uniquely be called market difficulty.",
            "The single c=0.05 rejection-penalty scenario fixes observed trajectories, "
            "inferred pre V and original M2 S. It does not predict new critic values or returns.",
        ],
        "formulas": {
            "reward": "r_t=Y/N including invalid requests; final step absorbs allocation remainder",
            "gae": "gamma=lambda=1, U_t approximately sum_{k>=t} r_k - V_t",
            "m2": "S_t=family mean terminal Y minus sibling mean in diverse advancing sessions; "
            "invalid S=0",
            "whitening": "A_t=(U_t+eta*S_t-batch_mean)/(sample_std+1e-8)",
            "inverse": "U_hat=A_logged*(logged_std+1e-8)+logged_mean-scaled_S",
            "fixed_penalty_scenario": "r'=r-0.05*invalid; recompute GAE and batch whitening; "
            "keep observed trajectories/pre V/S fixed, original evaluation Y unchanged",
        },
    }
    for cell, arm in (("m1", "M1"), ("m2_eta025", "M2")):
        folder = run_dir / cell / "artifacts"
        actor = sorted(read_jsonl(folder / "actor_update"), key=lambda r: r["policy_step"])
        critic = {r["policy_step"]: r for r in read_jsonl(folder / "critic_fit")}
        diagnostics = {r["policy_step"]: r for r in read_jsonl(folder / "sibling_diagnostics")}
        evidence = [
            r for r in read_jsonl(folder / "screening_evidence") if not r["identity"]["validate"]
        ]
        assert len(actor) == 8 and len(evidence) == 64
        lookup = {
            (r["group_uid"], r["identity"]["step"], r["identity"]["rollout_n"]): r for r in evidence
        }
        assert len(lookup) == 64
        arm_rows, arm_rounds, arm_trajectories = [], [], []
        for record in actor:
            step = record["policy_step"]
            diag, fit = diagnostics[step], critic[step]
            eta = diag["eta"]
            assert eta == (0.25 if arm == "M2" else 0.0)
            assert fit["old_value_dtype"] == "torch.bfloat16"
            rows, sibling_rows = [], []
            by_trajectory = defaultdict(list)
            for original in record["rows"]:
                e = lookup[(original["group_uid"], step, original["rollout_id"])]
                index = original["step_index"]
                trace, interaction, request = (
                    e["steps"][index],
                    e["interactions"][index],
                    e["requests"][index],
                )
                assert original["episode_id"] == e["episode_id"]
                assert original["response_token_ids"] == request["response_token_ids"]
                assert (
                    original["action"] == trace["action"]
                    and original["invalid_action"] == trace["invalid_action"]
                )
                assert original["terminal_Y"] == e["terminal_score"]
                assert math.isclose(
                    original["assigned_step_reward"],
                    float(np.float32(e["credits"][index])),
                    abs_tol=1e-7,
                )
                assert len(set(original["advantage"])) == 1
                sibling_rows.append(
                    SiblingStep(
                        trajectory_uid=original["trajectory_uid"],
                        group_uid=e["group_uid"],
                        episode_id=e["episode_id"],
                        policy_step=step,
                        rollout_id=original["rollout_id"],
                        rollout_n=4,
                        step_index=index,
                        step_count=len(e["steps"]),
                        decision_session_index=trace["decision_session_index"],
                        result_session_index=trace["result_session_index"],
                        action=trace["action"],
                        invalid_action=trace["invalid_action"],
                        done=trace["done"],
                        episode_complete=e["complete"],
                        terminal_score=e["terminal_score"],
                    )
                )
                row = {
                    "arm": arm,
                    "policy_step": step,
                    "episode_id": e["episode_id"],
                    "group_uid": e["group_uid"],
                    "trajectory_uid": original["trajectory_uid"],
                    "rollout_id": original["rollout_id"],
                    "step_index": index,
                    "decision_session_index": trace["decision_session_index"],
                    "result_session_index": trace["result_session_index"],
                    "action": trace["action"],
                    "invalid": trace["invalid_action"],
                    "terminal_Y": e["terminal_score"],
                    "trajectory_step_count": len(e["steps"]),
                    "assigned_reward": original["assigned_step_reward"],
                    "actual_A": original["advantage"][0],
                    "delta_logp_mean": original["delta_logp_mean"],
                    **invalid_details(interaction),
                }
                by_trajectory[original["trajectory_uid"]].append(len(rows))
                rows.append(row)
            assert len(rows) == diag["real_step_count"] == fit["real_step_count"]
            credits = compute_sibling_credit(sibling_rows)
            assert all(credits.diagnostics[k] == diag[k] for k in credits.diagnostics)
            scaled = torch.tensor(credits.pooled, dtype=torch.float32) * eta
            assert all(scaled[i] == 0 for i, r in enumerate(rows) if r["invalid"])
            a = np.array([r["actual_A"] for r in rows], dtype=np.float64)
            combined = (
                a * (diag["whitening_std"] + diag["whitening_epsilon"]) + diag["whitening_mean"]
            )
            u = combined - scaled.numpy().astype(np.float64)
            suffix = np.zeros(len(rows))
            for indices in by_trajectory.values():
                ordered = sorted(indices, key=lambda i: rows[i]["step_index"])
                assert [rows[i]["step_index"] for i in ordered] == list(range(len(ordered)))
                total = 0.0
                for i in reversed(ordered):
                    total += rows[i]["assigned_reward"]
                    suffix[i] = total
                e = lookup[(rows[ordered[0]]["group_uid"], step, rows[ordered[0]]["rollout_id"])]
                assert math.isclose(total, e["terminal_score"], rel_tol=1e-6, abs_tol=1e-6)
                bad = [i for i in ordered if rows[i]["invalid"]]
                arm_trajectories.append(
                    {
                        "arm": arm,
                        "policy_step": step,
                        "episode_id": e["episode_id"],
                        "trajectory_uid": rows[ordered[0]]["trajectory_uid"],
                        "rollout_id": e["identity"]["rollout_n"],
                        "terminal_Y": e["terminal_score"],
                        "steps": len(ordered),
                        "invalid_steps": len(bad),
                        "direct_reward_sum": total,
                        "invalid_direct_reward_sum": math.fsum(
                            rows[i]["assigned_reward"] for i in bad
                        ),
                        "invalid_allocation_fraction": len(bad) / len(ordered),
                        "suffix_targets_sum": math.fsum(suffix[i] for i in ordered),
                        "uniform_suffix_sum_formula": e["terminal_score"] * (len(ordered) + 1) / 2,
                        "gspo_trajectory_step_weight": len(ordered) / len(rows),
                        "gspo_weight_relative_to_equal_8_trajectories": 8
                        * len(ordered)
                        / len(rows),
                    }
                )
            v_estimated = suffix - u
            v_bf16 = torch.tensor(v_estimated, dtype=torch.float64).to(torch.bfloat16)
            tensor_u, tensor_returns = unwhitened_step_gae(
                token_level_rewards=torch.tensor(
                    [r["assigned_reward"] for r in rows], dtype=torch.float32
                )[:, None],
                values=v_bf16[:, None],
                response_mask=torch.ones((len(rows), 1), dtype=torch.long),
                trajectory_uids=np.array([r["trajectory_uid"] for r in rows], dtype=object),
                step_indices=np.array([r["step_index"] for r in rows], dtype=np.int32),
                gamma=1.0,
                lam=1.0,
            )
            replay_combined = tensor_u + scaled
            replay_a = (replay_combined - replay_combined.mean()) / (
                replay_combined.std(correction=1) + 1e-8
            )
            a_error = float(np.abs(replay_a.double().numpy() - a).max())
            u_error = float(np.abs(tensor_u.double().numpy() - u).max())
            recovered_stats = stats(v_bf16.double().tolist())
            critic_stat_error = max(
                abs(recovered_stats[k] - fit["pre"]["values"][k])
                for k in ("mean", "std_population", "min", "max")
            )
            return_stats = stats(tensor_returns.double().tolist())
            return_stat_error = max(
                abs(return_stats[k] - fit["pre"]["returns"][k])
                for k in ("mean", "std_population", "min", "max")
            )
            # Bounds validate a useful reconstruction; they do not assert exact
            # recovery of missing per-row critic observations.
            assert a_error < 2e-4 and u_error < 2e-4 and critic_stat_error < 2e-4
            assert return_stat_error < 2e-4
            penalty_rewards = torch.tensor(
                [r["assigned_reward"] for r in rows], dtype=torch.float32
            ) - 0.05 * torch.tensor([r["invalid"] for r in rows], dtype=torch.float32)
            penalty_u, _ = unwhitened_step_gae(
                token_level_rewards=penalty_rewards[:, None],
                values=v_bf16[:, None],
                response_mask=torch.ones((len(rows), 1), dtype=torch.long),
                trajectory_uids=np.array([r["trajectory_uid"] for r in rows], dtype=object),
                step_indices=np.array([r["step_index"] for r in rows], dtype=np.int32),
                gamma=1.0,
                lam=1.0,
            )
            penalty_combined = penalty_u + scaled
            penalty_a = (penalty_combined - penalty_combined.mean()) / (
                penalty_combined.std(correction=1) + 1e-8
            )
            u0 = (u - u.mean()) / (u.std(ddof=1) + 1e-8)
            for i, row in enumerate(rows):
                row.update(
                    suffix_return=float(suffix[i]),
                    reconstructed_U=float(u[i]),
                    reconstructed_pre_V=float(v_estimated[i]),
                    rounded_bf16_pre_V=float(v_bf16[i].float()),
                    pooled_S=float(credits.pooled[i]),
                    scaled_S=float(scaled[i]),
                    reconstructed_combined=float(combined[i]),
                    same_batch_eta0_A=float(u0[i]),
                    fixed_penalty_reward=float(penalty_rewards[i]),
                    fixed_penalty_U=float(penalty_u[i]),
                    fixed_penalty_A=float(penalty_a[i]),
                    whitening_mean=diag["whitening_mean"],
                    whitening_std=diag["whitening_std"],
                    eta=eta,
                    advancing=credits.advancing[i],
                )
            windows = {}
            for episode in sorted({r["episode_id"] for r in rows}):
                selected = [r for r in rows if r["episode_id"] == episode]
                values = np.array([r["reconstructed_combined"] for r in selected])
                window_mean, window_std = float(values.mean()), float(values.std(ddof=1))
                a_mean = mean([r["actual_A"] for r in selected])
                for row in selected:
                    row.update(
                        window_actual_A_mean=a_mean,
                        window_center_A=(row["reconstructed_combined"] - window_mean)
                        / (diag["whitening_std"] + 1e-8),
                        window_normalized_A=(row["reconstructed_combined"] - window_mean)
                        / (window_std + 1e-8),
                    )
                windows[episode] = {
                    "actual_A_mean": a_mean,
                    "steps": len(selected),
                    "invalid": summarize([r for r in selected if r["invalid"]]),
                    "suffix_return": stats([r["suffix_return"] for r in selected]),
                    "inferred_pre_V": stats([r["reconstructed_pre_V"] for r in selected]),
                    "combined_std": window_std,
                }
            advancing = [r for r in rows if r["advancing"]]
            invalid = [r for r in rows if r["invalid"]]
            round_result = {
                "arm": arm,
                "policy_step": step,
                "eta": eta,
                "all_steps": summarize(rows),
                "invalid": summarize(invalid),
                "advancing": summarize(advancing),
                "reconstruction_checks": {
                    "A_replay_max_abs_error": a_error,
                    "U_replay_max_abs_error": u_error,
                    "critic_pre_summary_max_abs_error": critic_stat_error,
                    "return_target_summary_max_abs_error": return_stat_error,
                    "pooled_S_sum": math.fsum(credits.pooled),
                    "scaled_S_sum": float(scaled.double().sum()),
                    "U_std_difference_from_logged": abs(
                        float(u.std(ddof=1)) - diag["all_real_steps"]["std_U"]
                    ),
                    "scaled_S_std_difference_from_logged": abs(
                        float(scaled.double().std(correction=1)) - diag["all_real_steps"]["std_S"]
                    ),
                },
                "whitening_mean": diag["whitening_mean"],
                "whitening_std": diag["whitening_std"],
                "U_std": float(u.std(ddof=1)),
                "scaled_S_std": float(scaled.double().std(correction=1)),
                "signal_std_ratio_S_over_U": float(scaled.double().std(correction=1))
                / float(u.std(ddof=1)),
                "critic_pre": fit["pre"],
                "critic_post": fit["post"],
                "critic_post_shift": fit["delta_values"],
                "critic_grad_clip_coefficients": fit["estimated_grad_clip_coefficients"],
                "critic_post_delta_exceeds_cliprange_fraction": fit[
                    "post_delta_exceeds_cliprange_fraction"
                ],
                "window_analysis": windows,
                "U_variance_decomposition": variance_by_window(rows, "reconstructed_U"),
                "G_variance_decomposition": variance_by_window(rows, "suffix_return"),
                "V_variance_decomposition": variance_by_window(rows, "reconstructed_pre_V"),
            }
            arm_rows.extend(rows)
            arm_rounds.append(round_result)
        groups = {
            "all_steps": summarize(arm_rows),
            "invalid": summarize([r for r in arm_rows if r["invalid"]]),
            "advancing": summarize([r for r in arm_rows if r["advancing"]]),
        }
        result["arms"][cell] = {
            **groups,
            "rounds": arm_rounds,
            "invalid_category_signals": {
                category: summarize([r for r in arm_rows if r["invalid_category"] == category])
                for category in sorted({r["invalid_category"] for r in arm_rows if r["invalid"]})
            },
            "max_reconstruction_A_error": max(
                r["reconstruction_checks"]["A_replay_max_abs_error"] for r in arm_rounds
            ),
            "max_reconstruction_critic_summary_error": max(
                r["reconstruction_checks"]["critic_pre_summary_max_abs_error"] for r in arm_rounds
            ),
            "critic_pre_mse": [r["critic_pre"]["mse"] for r in arm_rounds],
            "critic_post_mse": [r["critic_post"]["mse"] for r in arm_rounds],
            "between_window_U_variance_fraction_across_rounds": math.fsum(
                r["all_steps"]["steps"] * r["U_variance_decomposition"]["between_window_variance"]
                for r in arm_rounds
            )
            / math.fsum(
                r["all_steps"]["steps"] * r["U_variance_decomposition"]["total_variance"]
                for r in arm_rounds
            ),
            "step_reward_abs_median": float(
                np.median([abs(r["assigned_reward"]) for r in arm_rows])
            ),
            "trajectory_allocation": {
                "count": len(arm_trajectories),
                "min_steps": min(r["steps"] for r in arm_trajectories),
                "max_steps": max(r["steps"] for r in arm_trajectories),
                "max_invalid_allocation_fraction": max(
                    r["invalid_allocation_fraction"] for r in arm_trajectories
                ),
                "max_relative_gspo_trajectory_weight": max(
                    r["gspo_weight_relative_to_equal_8_trajectories"] for r in arm_trajectories
                ),
            },
        }
        all_rows.extend(arm_rows)
        all_trajectories.extend(arm_trajectories)
        all_rounds.extend(arm_rounds)
    result["source_sha256"] = sources
    result["analysis_script_sha256"] = sha(Path(__file__))
    for filename, rows in (
        ("step_signals.csv", all_rows),
        ("trajectory_allocation.csv", all_trajectories),
    ):
        columns = sorted(set().union(*(r.keys() for r in rows)))
        with (output_dir / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "signal_analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    (output_dir / "REPORT.md").write_text(render_report(result))
    manifest = {
        "status": "PASS",
        "at_SGT": result["at_SGT"],
        "analysis_script_sha256": result["analysis_script_sha256"],
        "source_sha256": sources,
        "outputs_sha256": {
            filename: sha(output_dir / filename)
            for filename in (
                "signal_analysis.json",
                "step_signals.csv",
                "trajectory_allocation.csv",
                "REPORT.md",
            )
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/m2-train-coverage-20260916-v2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/m2-train-coverage-diagnosis-20260917/signals",
    )
    args = parser.parse_args()
    result = analyze(args.run_dir.resolve(), args.output_dir.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "arms": {
                    k: {
                        "invalid": v["invalid"],
                        "max_A_replay_error": v["max_reconstruction_A_error"],
                    }
                    for k, v in result["arms"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
