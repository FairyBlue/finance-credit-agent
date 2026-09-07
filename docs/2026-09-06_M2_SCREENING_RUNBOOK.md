# M2 screening 实现与运行手册（2026-09-06）

**2026-09-07：已恢复；R3通过reference和critic更新，R4修正CPU容量检查后继续pilot。主 screening 仍为 0/8。**
执行范围见 [恢复记录](2026-09-07_M2_SCREENING_RESUMPTION.md)；此前暂停时结果见
[实验报告](2026-09-07_M2_SCREENING_EXPERIMENT_REPORT.md)。

本轮依据用户确认的线下讨论与执行授权，见 EVAL_PROTOCOL §20。只有 history/Train 参与运行，
正式 Valid/Final OOS 保持关闭。运行状态以对应目录的 JSON 和实际进程为准，不以本文宣称成功。

当前生成边界采用 [执行修订 R1](2026-09-06_M2_SCREENING_EXECUTION_AMENDMENT_R1.md)：
首轮 2048-token wiring 发生 post-update greedy 截断，因此共同 response cap 改为 4096、
入口 context cap 8192，重做必要验证。4B 工程 pilot 使用相同两个 Train episodes 做两轮更新，
检验两套 Adam 状态均已存在时的内存峰值。原失败保留，尚不能据此宣称 screening 跑通。

工程修复保留 [R2](2026-09-07_M2_SCREENING_EXECUTION_AMENDMENT_R2.md) 的非致命GPU遥测和
`artifacts/resource_phases/`命名阶段日志，并采用 [R3](2026-09-07_M2_SCREENING_EXECUTION_AMENDMENT_R3.md)
的pageable reference/Adam交换布局。R2实测监控超时后继续运行，但reference阶段持续主机内存限流，
已记录原因并主动停止。R3须从新base初始化验证两次更新及保存；该profile尚不支持checkpoint restore。
R3首actor更新前被按soft limit计算的reserve检查拒绝；[R4](2026-09-07_M2_SCREENING_EXECUTION_AMENDMENT_R4.md)
改按hard limit保留相同2 GiB余量，继续监测soft pressure。没有通过完整pilot前不进入主筛选。
最新实际状态见本机 `runs/M2_SCREENING_LATEST.md`，下方示例命令不代表已启动主筛选。

## 输入和实现

- 计划：`docs/2026-09-05_M2_EXPERIMENT_PLAN_V2.md`；实际共享设置：`configs/m2_screening.yaml`。
- 准备：`scripts/prepare_m2_screening.py`，产生忽略目录 `artifacts/m2_screening/` 下的新标定、
  manifest、32/12 主数据及两个 Train episodes 的 pilot 数据。旧奖励 payload 保持不变。
- 环境/flow：screening 显式选择共同 scorer、reward-neutral 观察和严格完成性。A2 训练 terminal
  R_net，其余 uniform Y；所有 arms 的最终评测均返回相同 Y。
- 优化：`src/agents/screening_runtime.py` 仅通过显式环境开关加载，基于完整 logical batch 加入
  sibling residual，critic returns 不变，真实 steps 唯一一次 whitening；不修改 pinned upstream。
- 启动：`python -m src.agents.screening_launcher`；顺序调度：`scripts/run_m2_screening.py`。

CPU 标定首次结果：floor=`0.0007613112285332427`，λ=`1.6912508025279984`；1,408 条代理、
28,160 个账务 steps。该结果来自指定 screen-train，未使用 dev 分数选择参数。新 calibration payload
SHA-256 为 `eaf1e50207ba727536a7fabb21d0e3ee3be12bd7569bd3e21e5d5696ab421a7e`。
代码或 artifact 发生变化时必须重新核对实际 hash，不能仅引用此记录跳过验证。

## 与 SSH 断开隔离

当前工作区已在 H100 主机上。已启用 `loginctl enable-linger tianming`，独立 user service 的生命周期
不依赖 SSH session。终端只用于启动与读取状态；trainer 的 stdin 为 `/dev/null`，stdout/stderr
写入文件，supervisor 由 systemd 持有。未使用易随终端挂断的前台 SSH 训练进程。

下面的命令必须使用已通过测试、已提交的实际 40 位 commit；同一 experiment root 不混合不同版本。

```bash
FCA_M2_COMMIT=$(git rev-parse HEAD)
FCA_M2_RUN=/home/tianming/projects/finance-credit-agent/runs/m2-screening-20260906

systemd-run --user --unit=fca-m2-screening --collect \
  --property=WorkingDirectory=/home/tianming/projects/finance-credit-agent \
  --property=KillMode=control-group \
  /home/tianming/projects/finance-credit-agent/.venv/bin/python \
  scripts/run_m2_screening.py --stage all \
  --experiment-root "$FCA_M2_RUN" --expected-commit "$FCA_M2_COMMIT"
```

服务默认不自动重启失败 cell，避免把策略截断或接线 bug 当成可无限重试的网络故障。网络中断本身
不需要重启服务。机器重启、GPU OOM、文件系统耗尽仍是独立故障，不能承诺被 SSH 隔离机制修复。

```bash
systemctl --user status fca-m2-screening --no-pager
journalctl --user -u fca-m2-screening -n 40 --no-pager
cat runs/m2-screening-20260906/supervisor.json
```

## 结果与故障定位

每个 cell 的目录包含 `resolved_config.json`、`settings.json`、`command.json`、`preflight.json`、
`summary.json`、`trainer.log`、`resources.jsonl` 和 `artifacts/`。后者分别保存请求/轨迹、optimizer
update 和 sibling diagnostics。断线后重新连接，可继续读相同文件。

`PASSED` 必须有正确数量的完整训练/评测轨迹、每 round 的 GAE/residual 诊断及实际 actor/critic
更新，不能只依据 trainer exit 0。主阶段完整后生成 `readout.json` 和 `READOUT.md`；
`PROMISING` 仍只是研发筛选，没有显著性或 Final OOS 含义。

首次资源布局采用 actor Adam / critic 参数留 GPU、critic Adam offload、reference bf16 和较小
vLLM memory utilization。主机仅 64 GiB RAM、没有可用 swap；实际容量由 pilot 验证。调整统一
布局时记录新配置/commit，保留旧失败证据。只有确认 final model 已保留并校验后，才能清理本任务
已结束 cell 的可再生恢复快照；不得清理其他实验或整个 runs 目录。

磁盘实现选择：计划推荐保留两个恢复点，但本机总磁盘约 289 GB、初始可用约 236 GB。
八个最终 actor 加上两个完整 actor/critic/Adam 恢复点会超过可用空间，因此在任何 GPU/dev 运行前
统一设置 `max_checkpoints=1`。每 4 rounds 保存规则不变；完成 cell 的最终 actor 导出为推理使用的
BF16 HF 权重，校验后才删除该 cell 的恢复状态。保存失败可能失去上一恢复点，此时停止并保留
失败证据，不自动重采样。此选择仅影响恢复能力和磁盘峰值，不改变训练或评测预算。
