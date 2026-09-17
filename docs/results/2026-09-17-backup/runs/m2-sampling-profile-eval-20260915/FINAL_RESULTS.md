# 当前与Qwen推荐采样参数：同批交叉评估

当前进展：**2026-09-15T21:03:37.194031+08:00 SGT（UTC+8）**。
执行状态：ALL_CELLS_OBSERVED_WITH_INCOMPLETE；已终止观察 384/384，完整 369，未完成 15。

每模型每配置：训练日期8窗×4条、验收日期4窗×4条。先同窗等权，再固定窗口等权；任何缺失或cap不补零、不缩小收益分母。

| 模型 | 日期组 | 配置 | 完整/计划 | 净收益% | Y | 成本USD/窗 | 末期现金% | 非法/请求 | buy/sell/hold | 成交/股票数 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sft0 | training_windows | current | 32/32 | 2.4934 | 3.7714 | 5177.0291 | 68.3254 | 66/706 | 321/247/72 | 563/3 |
| sft0 | training_windows | qwen_recommended | 28/32 | — | — | — | — | 555/1131 | 287/230/59 | 517/3 |
| sft0 | evaluation_windows | current | 16/16 | 1.4239 | -5.3660 | 5578.2354 | 49.7052 | 9/329 | 176/128/16 | 303/4 |
| sft0 | evaluation_windows | qwen_recommended | 16/16 | 1.5502 | -5.0873 | 5947.9607 | 58.5002 | 1/321 | 179/141/0 | 319/3 |
| m1 | training_windows | current | 32/32 | 2.5757 | 3.8959 | 4749.3234 | 79.7250 | 176/816 | 269/233/138 | 502/3 |
| m1 | training_windows | qwen_recommended | 26/32 | — | — | — | — | 894/1446 | 224/199/129 | 423/3 |
| m1 | evaluation_windows | current | 16/16 | 0.2128 | -7.6313 | 5405.2889 | 74.8510 | 6/326 | 156/134/30 | 290/3 |
| m1 | evaluation_windows | qwen_recommended | 16/16 | 0.2674 | -7.6639 | 5808.2175 | 76.2942 | 2/322 | 165/145/10 | 310/3 |
| m2_eta1 | training_windows | current | 32/32 | 3.0721 | 4.4778 | 4354.7551 | 80.3473 | 159/799 | 243/212/185 | 455/3 |
| m2_eta1 | training_windows | qwen_recommended | 27/32 | — | — | — | — | 765/1333 | 223/190/155 | 413/3 |
| m2_eta1 | evaluation_windows | current | 16/16 | 0.1607 | -7.7472 | 4655.4574 | 77.4640 | 19/339 | 133/115/72 | 248/3 |
| m2_eta1 | evaluation_windows | qwen_recommended | 16/16 | 0.4697 | -7.1882 | 5314.9557 | 76.2753 | 3/323 | 152/132/36 | 284/3 |
| m2_eta025 | training_windows | current | 32/32 | 2.8662 | 4.2310 | 4396.0275 | 78.4438 | 134/774 | 249/213/178 | 462/3 |
| m2_eta025 | training_windows | qwen_recommended | 32/32 | 3.3428 | 4.7611 | 4564.5012 | 76.5794 | 176/816 | 257/220/163 | 477/3 |
| m2_eta025 | evaluation_windows | current | 16/16 | 0.6253 | -6.8114 | 5154.5168 | 77.4976 | 11/331 | 146/128/46 | 274/3 |
| m2_eta025 | evaluation_windows | qwen_recommended | 16/16 | 0.3668 | -7.4191 | 5706.5748 | 77.4769 | 3/323 | 161/143/16 | 304/3 |

行为计数包括未完成轨迹；交易次数或股票数增加本身不等于收益改善。

| 推荐配置−current | 日期组 | 净收益Δpp | Y差 | 成本差USD | 非法次数差/轨迹 | 窗口非法率差pp | 收益改善窗 |
|---|---|---:|---:|---:|---:|---:|---:|
| sft0 | training_windows | — | — | — | 15.2812 | 10.6943 | — |
| sft0 | evaluation_windows | 0.1262 | 0.2786 | 369.7254 | -0.5000 | -2.3952 | 3/4 |
| m1 | training_windows | — | — | — | 22.4375 | 18.1880 | — |
| m1 | evaluation_windows | 0.0546 | -0.0326 | 402.9286 | -0.2500 | -1.2123 | 2/4 |
| m2_eta1 | training_windows | — | — | — | 18.9375 | 16.8327 | — |
| m2_eta1 | evaluation_windows | 0.3090 | 0.5591 | 659.4983 | -1.0000 | -4.6588 | 3/4 |
| m2_eta025 | training_windows | 0.4766 | 0.5301 | 168.4738 | 1.3125 | 3.4296 | 7/8 |
| m2_eta025 | evaluation_windows | -0.2585 | -0.6076 | 552.0581 | -0.5000 | -2.3815 | 2/4 |

非法比较需要固定窗口的四条终止观察齐全，可包含cap；收益比较要求四条全部完整。窗口非法率先逐窗计算再等权，与所有请求合并的非法率口径不同。

| RL−同配置SFT | 日期组 | 配置 | 净收益Δpp | Y差 | 收益改善窗 |
|---|---|---|---:|---:|---:|
| m1 | training_windows | current | 0.0823 | 0.1245 | 4/8 |
| m1 | training_windows | qwen_recommended | — | — | — |
| m1 | evaluation_windows | current | -1.2111 | -2.2654 | 0/4 |
| m1 | evaluation_windows | qwen_recommended | -1.2828 | -2.5766 | 1/4 |
| m2_eta1 | training_windows | current | 0.5787 | 0.7064 | 6/8 |
| m2_eta1 | training_windows | qwen_recommended | — | — | — |
| m2_eta1 | evaluation_windows | current | -1.2632 | -2.3812 | 0/4 |
| m2_eta1 | evaluation_windows | qwen_recommended | -1.0805 | -2.1008 | 0/4 |
| m2_eta025 | training_windows | current | 0.3728 | 0.4596 | 6/8 |
| m2_eta025 | training_windows | qwen_recommended | — | — | — |
| m2_eta025 | evaluation_windows | current | -0.7987 | -1.4455 | 0/4 |
| m2_eta025 | evaluation_windows | qwen_recommended | -1.1834 | -2.3317 | 0/4 |

已记录的失败类型：
- sft0 / training_windows / qwen_recommended: {"REQUEST_CAP": 4}
- m1 / training_windows / qwen_recommended: {"REQUEST_CAP": 6}
- m2_eta1 / training_windows / qwen_recommended: {"REQUEST_CAP": 5}

INCOMPLETE表示轨迹或证据尚未全部完整，与服务执行状态分开。所有窗口仍属于Train，单seed、滚动窗口重叠；本批未训练新模型，不混入旧worker收益。

逐窗点、固定分母、来源SHA256见readout.json；平面表见summary.csv。
