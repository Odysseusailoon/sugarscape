基于对 Epstein & Axtell 经典研究的搜索和对我们当前数据的初步审查，我制定了以下深度分析计划，旨在挖掘超越基础统计量的“有意义的 Finding”。

### 核心目标
不仅仅是画出“不平等曲线”，而是要回答 **“为什么不平等？是谁活下来了？财富和属性有什么关系？”** 这些更深层次的问题。

### 1. 新增分析脚本 `scripts/analyze_sugarscape_deep.py`
我们将编写一个专门的分析脚本，处理之前生成的 `detailed_data.json`（快照数据）和 `metrics.csv`，重点分析以下四个维度的相关性：

#### **A. 属性-生存优势分析 (Selection Effects)**
*   **假设**：低代谢、高视野的 Agent 更容易活下来。
*   **分析方法**：
    *   对比 **Initial Population** (t=0) 和 **Final Population** (t=500) 的属性分布。
    *   计算 **Metabolism** (1-4) 和 **Vision** (1-6) 的平均值变化。
    *   **可视化**：绘制“幸存者偏差图”——不同代谢率 Agent 的占比随时间变化的堆叠面积图。

#### **B. 属性-财富相关性 (Wealth Determinants)**
*   **假设**：在这两座糖山的世界里，位置（出生点）可能比能力（视野）更重要（“出身论” vs “能力论”）。
*   **分析方法**：
    *   对最终时刻的 Agent 进行回归分析或分组统计。
    *   **Scatter Plot**: X轴=Vision, Y轴=Wealth (看相关性)。
    *   **Box Plot**: 不同 Metabolism 等级的财富中位数对比。

#### **C. 阶层固化与流动性 (Mobility & Stratification)**
*   **假设**：富人倾向于占据糖山顶峰并待着不动，穷人则被迫在贫瘠区游荡。
*   **分析方法**：
    *   计算 **Wealth** 与 **Displacement** (移动距离) 的相关性。
    *   **可视化**：将 Agent 分为“富裕组 (Top 20%)”和“贫穷组 (Bottom 50%)”，对比他们的平均移动距离和探索率。

#### **D. 承载力与周期性 (Carrying Capacity)**
*   **假设**：如果人口不强制恒定，环境承载力会限制人口上限。虽然我们目前是恒定人口，但可以观察“饥饿死亡率”的波动。
*   **分析方法**：
    *   分析死亡原因（如果是代码支持的话，或者推断）：多少人是饿死的（Wealth <= 0），多少是老死的。
    *   **Finding**: 如果大多数人是老死的，说明资源充足；如果是饿死的，说明竞争残酷。

### 2. 预期产出 (Deliverables)
脚本运行后将生成一份 Markdown 格式的 **《Sugarscape 深度洞察报告》 (`results/.../insight_report.md`)** 和一组新的深度分析图表：
1.  `selection_effect_metabolism.png`: 代谢率分布随时间的演变。
2.  `wealth_vs_attributes.png`: 财富与视野/代谢的相关性热力图。
3.  `mobility_by_class.png`: 富人 vs 穷人的移动性对比。

### 3. 执行步骤
1.  编写 `scripts/analyze_sugarscape_deep.py`。
2.  运行脚本分析已有的实验数据（如 `baseline_no_llm`）。
3.  解读生成的结果，为您总结出类似于“在糖域中，低消耗比高视野更重要”这样的高价值结论。