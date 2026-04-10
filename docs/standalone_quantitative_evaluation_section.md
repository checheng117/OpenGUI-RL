# 独立量化评测部分：Cross-Website GUI Grounding with Verifiable Reward Optimization

## 量化评测设置

为系统性评估 `screenshot + instruction -> target element / bounding box / click point / action type` 的多模态 GUI grounding 能力，本项目将量化测试拆分为三个泛化层级：`test_task`、`test_website` 与 `test_domain`。其中，`test_task` 衡量同网站内任务迁移，`test_website` 衡量跨站点迁移，`test_domain` 衡量跨领域迁移，用来逐级刻画模型在真实开放网页环境中的泛化退化曲线。

训练主源采用 Multimodal-Mind2Web，并结合 ScreenSpot-v2、VisualWebBench 与 ScreenSpot-Pro 进行补充验证。实验主线围绕两类输入设置展开：`pure vision` 与 `screenshot + OCR / DOM candidates`。工程上采用“候选生成 + reward-based rerank”作为主方案，使评测既能反映 first-choice 基线质量，也能反映 candidate pool 中仍可恢复的 headroom。

本节中的数字不是手工摘录表格，而是于 `2026-04-10` 通过运行 `python scripts/run_quantitative_metrics_suite.py` 从仓库现有真实评测 artifacts 重新计算得到，汇总结果保存在 `outputs/quantitative_metrics_suite/summary.json`。

## 指标定义与测试口径

量化测试覆盖四个 headline 指标和一组 reward 诊断指标。除非特别说明，本文中的“增益”均表示绝对百分点提升（absolute percentage points）。

- `Element accuracy`：预测目标元素与标注元素完全一致的比例，对应 repo 中的 `element_accuracy`。
- `Click-point accuracy`：预测点击点落在目标元素标注框内的比例，对应 repo 中的 `point_accuracy`。
- `IoU@0.5`：预测框与标注框的 IoU 不低于 `0.5` 的比例，对应 repo 中的 `iou_at_threshold(..., threshold=0.5)`。
- `Action-type accuracy`：预测动作类型与标注动作类型一致的比例，对应 repo 中的 `action_type_accuracy`。
- `Invalid format rate`：预测输出存在框坐标非法、点击点越界、或框与点击点同时缺失的比例；该指标用于 verifiable reward 的稳定性约束，对应 repo 中的 `invalid_format_rate` 与 reward 层的 `invalid_format_penalty`。

对于 Stage B，本节优先报告当前 artifacts 真正支持的 `oracle best-of-k headroom`，也就是 candidate pool 内理论最优选择相对 first-choice 的上界增益。这样可以避免把实际 reranker 在不同阶段、不同候选池上的不稳定收益误写成一个单一固定数字。对于效率评测，则单独记录 first-choice 推理时延、structured support path 时延、verifier 额外耗时以及端到端总时延。

## 主结果：跨任务、跨网站与跨域泛化

表 1 给出了当前最强 Mind2Web hybrid Stage A 在三种泛化设置下的真实结果。整体上，`test_task` 仍是最容易的划分；`test_website` 出现最明显退化；`test_domain` 虽然样本更少，但仍保持接近 `89%` 的 element / point / IoU@0.5 水平。三种划分上的 `Invalid format rate` 均为 `0%`，说明输出格式稳定性不是当前瓶颈。

| 指标 | `test_task` | `test_website` | `test_domain` |
| --- | ---: | ---: | ---: |
| `Element accuracy` | `95.00%` | `80.00%` | `89.47%` |
| `Click-point accuracy` | `95.00%` | `85.00%` | `89.47%` |
| `IoU@0.5` | `95.00%` | `85.00%` | `89.47%` |
| `Action-type accuracy` | `90.00%` | `95.00%` | `94.74%` |
| `Invalid format rate` | `0.00%` | `0.00%` | `0.00%` |

这些结果表明，当前 hybrid candidate-aware 表征已经把 Mind2Web 上的主要误差压缩到少量跨网站困难样本，而不是大面积格式失败或动作类型崩塌。因此，本项目的关键收益主要来自表示层与候选锚定，而不是单纯增加输出约束。

## Rerank、输入形态与平台差异分析

为进一步分析模型性能来源，表 2 汇总了 candidate-pool headroom、目标类型差异、输入模态差异以及平台迁移影响。真实 artifacts 显示，历史 Stage B 候选池在旧版弱 Stage A 上仍存在明显 headroom：`k=4` 与 `k=8` 的 oracle best-of-k point gain 都约为 `+10.17 pts`，而 `k=8` 在 reward gain 上略高（`0.0431` vs `0.0376`）。但在最终 hybrid rebuild 上，`k=4` 的 oracle point gain 已经缩小到 `+5.08 pts`，这与项目最终结论一致：强监督会显著压缩 rerank 可恢复空间。

同时，在 ScreenSpot-v2 最终 dual-path verifier 上，文本元素比 icon 元素高出 `+19.70 pts` point accuracy；而在 Mind2Web 官方子集均值上，`screenshot + OCR / DOM` 相比纯视觉带来 `Element +88.16 pts`、`Click-point +88.07 pts`、`IoU@0.5 +89.82 pts`、`Action-type +13.68 pts` 的提升，说明结构候选不是“微调增益”，而是决定系统是否具备有效 grounding 能力的主因。跨平台方面，以 ScreenSpot-v2 的 web 子集为参照，desktop 低 `1.58 pts`，mobile 反而高 `4.81 pts`，说明当前 held-out 数据上 mobile 并非更难，平台差异更多来自数据分布而不是统一难度排序。

| 分析项 | 结果 |
| --- | --- |
| `Oracle best-of-k point gain` | `+10.17 pts` (`historical k=4`), `+10.17 pts` (`historical k=8 expanded`), `+5.08 pts` (`final hybrid rebuild k=4`) |
| `Oracle best-of-k reward gain` | `0.0376` (`historical k=4`), `0.0431` (`historical k=8 expanded`), `0.0983` (`final hybrid rebuild k=4`) |
| `text element` 相对 `icon` | `+19.70 pts` |
| `screenshot + OCR / DOM` 相对 `pure vision` | `Element +88.16 pts`; `Click-point +88.07 pts`; `IoU@0.5 +89.82 pts`; `Action-type +13.68 pts` |
| `desktop / mobile` 相对 `web` | `-1.58 pts / +4.81 pts` |

从这个角度看，reward-based reranking 不是替代基础 grounding 能力，而是作为 second-stage selection 机制去吸收候选池中剩余的 recoverable headroom；而 OCR / DOM cues 则直接决定了 first-choice 是否足够强。

## 推理效率与部署可行性

除准确率外，部署可行性同样重要。表 3 给出了当前可从真实 artifacts 直接恢复的效率指标。对于最终 held-out ScreenSpot-v2 dual-path 流程，point-native first-choice 路径耗时约 `2.1185 s / image`，structured support path 耗时约 `1.3599 s / image`，verifier 选择本身只增加 `0.000167 s / image`，端到端总时延约 `3.4786 s / image`。在 Mind2Web Stage B 上，最终 rebuild 的候选池大小为 `4`，历史 expanded pool 为 `8`。

| 指标 | 数值 |
| --- | ---: |
| Mind2Web Stage B 候选数 | `4`（final rebuild）, `8`（historical expanded） |
| ScreenSpot-v2 point-native first-choice | `2.1185 s / image` |
| ScreenSpot-v2 structured support path | `1.3599 s / image` |
| ScreenSpot-v2 verifier 额外耗时 | `0.000167 s / image` |
| ScreenSpot-v2 dual-path 全流程时延 | `3.4786 s / image` |

## 小结

这一组真实复算后的量化测试说明，本项目的主要贡献不在于完整执行长链路网页 agent，而在于把最关键、最可验证的一步 GUI grounding 做强，并用 verifiable reward 将“候选是否值得保留或重排”转化为可度量、可比较、可优化的问题。当前 strongest Mind2Web hybrid Stage A 已经在官方子集上达到 `95.00 / 80.00 / 89.47` 的 element accuracy，而 Stage B 的最终结论也被实测支持：历史弱基线下存在约 `+10 pts` 的 oracle headroom，但在最终 hybrid rebuild 上只剩 `+5.08 pts`。在 held-out GUI benchmark 上，ScreenSpot-v2 dual-path verifier 达到 `77.91%` point accuracy、`17.22%` IoU@0.5，VisualWebBench point-native 仍保持 `87.21%` official choice accuracy。整体上，这些结果共同支撑了“强表示优先、reward 作为条件性增益”的最终项目结论。
