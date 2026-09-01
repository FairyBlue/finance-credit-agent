# Formal Manifest 审核与生成记录（2026-09-01）

状态：`REVIEWED / PROCESSED_VERIFIED`

## 结论

两位合作者已按最小清单确认 formal episode manifest。正式文件只将已复算 candidate 的状态改为
`reviewed`，其他字段没有变化；随后已生成 Train/Valid/Test processed Parquet，并从冻结原始行情
通过共享 next-open 账务引擎重新得到相同的等权 20 股 benchmark SHA-256。

| Split | Source | Episodes | Sessions/episode | 边界 |
|---|---|---:|---:|---|
| Train | history | 441 | 21 | `2023-01-03`～`2024-10-31` |
| Valid | history | 60 | 21 | `2024-11-01`～`2025-02-28` |
| Test | StockBench | 1 | 83 | `2025-03-03`～`2025-06-30` |

Train/Valid 使用 stride 1 的 21-session rolling windows；Test 是单个不可拆分 Final OOS episode。
21-session horizon 沿用 train-only reward calibration，不引入新的 episode horizon。

## 身份与复算证据

| 项目 | SHA-256 / 值 |
|---|---|
| candidate raw bytes | `4fe7da766d214f6fa14e71e080a537759354caf2de842237d244a332d19dbc98` |
| reviewed manifest raw bytes | `4a8ff4f3ad8c9305c03aab17f44ebda5c4d493c22b6610a67a70ba238a5a54a3` |
| benchmark series | `a86507384e6806db84066a0cd286cc143664ecddb3c2b851f63562ea45b117b8` |
| history raw manifest | `2612e707446b1c8bcd73ac6583b84558d3c4eddbaa9fc20adf7023c3c8a76c90` |
| StockBench raw manifest | `8b48fc7d50f4ecdbc49e920eb83d72df02b9a56d4365bce7f0db6bb4a342bd1f` |
| history / StockBench files | `10,820 / 1,660` |
| independent generation comparison | byte-identical |
| processed Train | 441 rows / `87e9de90fc322f61b6b18e806b50fc59e3a8adce850442b69659f9f2bc21b33b` |
| processed Valid | 60 rows / `5e088b156a04989a20a016d297ba442f6780388f25a1d3cda283c5ddadbacbcf` |
| processed Test | 1 row / `92b6e574f573f3d5e759cce5c14f7f658795cc4f539369e135839410e04f692b` |

正式文件是
[`configs/episode_manifest.reviewed.yaml`](../configs/episode_manifest.reviewed.yaml)。processed Parquet
和 StockBench 原始数据位于 Git-ignored `data/`，不会上传 GitHub。

## 审核与复算证据

两位合作者于 2026-09-01 确认了以下最小字段：

1. Train/Valid/Test source 和日期边界；
2. episode 数 `441/60/1` 与 session 长度 `21/21/83`；
3. 固定 20 股 universe；
4. benchmark series SHA-256；
5. candidate raw-byte SHA-256。

晋级时只把 `status: generated_pending_collaborator_review` 改成 `status: reviewed` 并重命名文件。
把 reviewed 文件的 status 在内存中逆向还原后，raw SHA-256 精确回到 candidate 的
`4fe7da...bc98`，证明其余字节未变。`scripts/prepare_episodes.py` 生成了 `441/60/1` 行；launcher
绑定检查逐行验证 Train/Valid，Test 保持为单个不可拆分 Final OOS row。独立 raw-bar benchmark
重放得到 `a86507...b117b8`，与 manifest 一致。

上述 processed SHA 是本机生成证据；Parquet 的 `env_kwargs` 按设计绑定绝对数据路径，因此换主机
重新生成时 raw bytes 可以不同，`441/60/1` row 数和 reviewed-manifest 逐字段绑定才是可移植门禁。

这一步只验证冻结基准和数据接线，没有运行模型、生成 agent 预测、评估 Final OOS 表现或据此调参。
正式训练仍需干净执行 commit、完整 H100 preflight，以及 A3 主结论所需 M2/M3。
