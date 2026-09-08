# H100 空间清理候选盘点（2026-09-08）

**执行更新：用户已授权清理前三项、保留第四项uv；现已完成。** 删除了pip缓存、扩展安装包
缓存、指定的两个旧VS Code Server、55个conda压缩安装包，以及旧Qwen2.5-3B模型缓存。
可用空间实测由99.11GiB升至113.64GiB，增加约14.53GiB，已达到112GiB磁盘门槛。
uv与其到项目Torch文件的硬链接保持完整；失败checkpoint、Qwen3模型、当前环境均保留。
本次没有启动训练，没有删除Agent-R1源码或任何Python环境。

执行清单和复核收据为本机 `runs/disk-audit-20260908/approved_cleanup_plan.json`、
`cleanup_execution.json`、`cleanup_verification.json`。ZFS在unlink完成后异步回收，最终
释放量以上述复核收据的较后磁盘读数为准。下文保留清理前的评估依据和范围。

**清理前盘点：** 当时可用约99.11GiB，训练准入要求112GiB，还差约12.89GiB。
有机会在保留实验 checkpoint、当前模型和环境的前提下腾出约14.53GiB，但其中5.76GiB
属于旧项目仍有引用的模型缓存，需要确认近期不运行旧示例。仅常规缓存清理预计不足以达标。

计量使用文件实际块占用与 inode/硬链接关系，不把目录显示大小直接当作释放量。以下为
估算；实际删除后必须重新检查 `df`，尤其当前文件系统为ZFS。清理不等于训练自动获准启动。
原始盘点数字保存在本机 `runs/disk-audit-20260908/inventory.json`。

## 优先候选 A：约8.77GiB

| 文件或目录 | 估计可释放 | 判断及影响 |
| --- | ---: | --- |
| `/home/tianming/.cache/pip` | 7.67GiB | 首选。下载/构建缓存，未发现对外硬链接；当前虚拟环境及 micromamba 环境没有指向它的符号链接或 `.pth`/`.egg-link` 引用。清理不卸载现有包，以后重装可能重新下载。 |
| `/home/tianming/.vscode-server/data/CachedExtensionVSIXs` | 0.23GiB | 扩展安装包缓存；保留已安装扩展目录，以后安装时可能重下。 |
| 两个旧 VS Code Server 版本，见下方精确路径 | 0.64GiB | 当前进程未引用这两版。实际清理前再核对进程；旧版客户端重连时可能重新安装。 |
| `/home/tianming/micromamba/pkgs` 中的 `*.conda`、`*.tar.bz2` 安装包 | 0.24GiB | 只清压缩安装包，不删整个 `pkgs`，不删环境。下次安装可能需要下载。 |

两个旧 Server 路径：

- `/home/tianming/.vscode-server/cli/servers/Stable-520fb30b2d3d324b4cb2342f6e88e2cd93751de1`
- `/home/tianming/.vscode-server/cli/servers/Stable-08d4889f9ec4a1685d257b9b95de036c8e1ce1e5`

当前使用的 `Stable-a44adf7f53e00964ab890f9f8758a334f1fc15bc` 保留。与当前 agent host
版本相关的 `110a328...` 也不列入本次候选。A项全部清理后预计约107.88GiB，仍不足112GiB。

## 条件候选 B：旧 Qwen2.5-3B，约5.76GiB

路径：`/home/tianming/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct`。

当前 finance-credit-agent 的 M2/R5 配置使用 Qwen3-1.7B 与 Qwen3-4B，不依赖这个模型。
但另一个项目中确实有引用，不能直接称为无用：

- `/home/tianming/projects/Agent-R1/examples/run_qwen2.5-3b-h100-smoke.sh:16`
- `/home/tianming/projects/Agent-R1/examples/run_qwen2.5-3b.sh:18`

如果近期不用这些旧示例，可以清理该旧模型缓存，代价是之后运行这些示例时要重新下载，
离线模式下则不能直接运行。其文件未发现外部硬链接。

**A+B估计释放14.53GiB，可用空间约113.64GiB，略高于112GiB门槛。**
这只是临时腾挪，余量不宽裕，不能替代管理员增加100GB；实际可释放量和启动前余量须实测。

## 不优先或需进一步确认的候选

| 对象 | 大小 / 可释放估计 | 原因 |
| --- | --- | --- |
| `/home/tianming/.cache/uv` | 文件块占用6.75GiB，但独占inode仅约0.15GiB | 大部分与已安装包共享硬链接。例：缓存和项目 `.venv` 中的 `torch/lib/libtorch_cuda.so` 同为device80/inode479050，link count=2。只删缓存不会释放仍被环境引用的主体，不值得为“6.8G”清理它。 |
| `/home/tianming/micromamba/envs/agent310` | 独占文件约3.69GiB | 不是本项目当前解释器，但尚不能证明其他项目不用；删除前应保存环境清单并检查旧项目依赖。 |
| `/home/tianming/micromamba/envs/agentr1` | 独占文件约6.70GiB | 同上，完整环境不是普通下载缓存，重建成本高；不是首选。 |
| v8 的1.7B诊断 `actor_hf` | 约3.79GiB | 是可由原 checkpoint 重新导出的派生模型，但与历史诊断复核有关；不作为第一批清理项。 |
| v10 的4B诊断 `actor_hf` | 约8.22GiB | 当前 G1/G2/G3 诊断所用导出模型，受来源/hash核验约束；不要无记录移除。 |

`/tmp` 总共约0.10GiB，Ray临时目录约0.03GiB；大量旧失败启动目录只有KB/MB。
删它们收益很小，却容易损失排错记录，不建议为了凑空间清空 `/tmp` 或整批 `runs`。

## 必须保留的主要资产

- 原1.7B失败运行 `runs/m2-screening-20260906-v4` 约36.31GiB。
- 原4B失败 cell `runs/m2-screening-20260907-v9/screening-a2-s42` 约87.42GiB。
  两者状态均为FAILED，包含恢复点与故障研究证据，不是普通缓存。
- Qwen3-1.7B、Qwen3-4B 的 exact base snapshots，后续门禁依赖它们。
- 项目 `.venv` 及 `/home/tianming/micromamba/envs/finance-credit`：`.venv/bin/python`
  实际指向后者的解释器；不要因看似存在两套环境就删除其中一套。
- Train/PIT原始数据、冻结标定、manifest、轨迹、报告和checkpoint完成性/来源记录。

原建议顺序为先处理A项，再按旧Qwen2.5示例是否仍需使用决定B项。用户后续明确表示旧模型
暂时不用，授权A+B；现已完成以上范围，没有卸载包、重建环境或恢复训练。

## Agent-R1 本身的空间

| 对象 | 本次实测块占用 | 是否删除 |
| --- | ---: | --- |
| `/home/tianming/projects/Agent-R1` 源码仓库（含.git、outputs和logs） | 4.15MiB | 保留；源码本身很小 |
| 本项目 `third_party/agent-r1` 依赖源码目录 | 2.11MiB | 保留；本项目依赖它 |
| `/home/tianming/micromamba/envs/agentr1` Python环境 | 6.76GiB | 保留；不是源码仓库，未获授权删除 |

该环境是主要占用；按先前独占inode统计，单独删除它预计可释放约6.70GiB，其余有共享
硬链接，不能把目录总占用全部算成可回收。当前finance-credit-agent的`.venv/bin/python`
指向`envs/finance-credit`，并不指向`envs/agentr1`；这不证明旧Agent-R1项目以后不需要该环境。
