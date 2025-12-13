为了消除随机性带来的误差，并得出具有统计显著性的结论，我将执行大规模的蒙特卡洛实验。

### 实验方案
我们将运行 **50 组** 严格控制变量的对比实验。每组实验包含两个平行世界：
1.  **SugarOnly**: 仅有糖资源。
2.  **SugarSpice**: 糖 + 香料资源。

### 关键控制点
*   **种子同步 (Seed Synchronization)**: 每一组 (Pair $i$) 的 SugarOnly 和 SugarSpice 将共享同一个随机种子 `seed_i`。这确保了两个世界在初始时刻拥有完全相同的地形（糖分布）、完全相同的 Agent 列表（位置、代谢率、视野、Persona）。唯一的区别是 SugarSpice 世界额外生成了香料层和 Agent 的香料属性。
*   **规模**: 总计运行 50 * 2 = 100 次仿真。

### 执行步骤
1.  **编写脚本 (`scripts/run_batch_comparison.py`)**:
    *   循环 50 次。
    *   每次生成一个随机种子。
    *   分别运行 SugarOnly 和 SugarSpice。
    *   收集所有幸存 Agent 的数据（Persona, Age, Utility）。
2.  **数据聚合与分析**:
    *   将 50 次实验的数据合并。
    *   计算每个 Persona 在两种环境下的平均寿命和平均财富，并计算 **置信区间 (Confidence Interval)**。
    *   进行 T 检验，判断差异是否显著。
3.  **产出**:
    *   `batch_report.md`: 包含统计检验结果的最终报告。
    *   `batch_survival.png` & `batch_wealth.png`: 带误差条的对比柱状图。

这个方案将彻底验证之前的“冒险者致富”和“游牧者生存”结论是否在统计上站得住脚。