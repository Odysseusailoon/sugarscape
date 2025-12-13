为了全面挖掘 Persona 实验数据的潜力，我将设计一套**全维度的深度分析方案**。不再局限于简单的“寿命”和“财富”均值，而是深入到“性格 x 属性 x 环境”的交互效应中，生成一系列高信息密度的图表。

### 1. 分析维度矩阵 (Dimensions of Analysis)
我们将对 50 组实验的聚合数据 (`all_data`) 进行多维交叉分析：

*   **X 轴 (自变量)**:
    *   **Persona**: A (保守), B (规划), C (游牧), D (冒险)
    *   **Environment**: SugarOnly vs SugarSpice
    *   **Attributes**: Vision (1-6), Metabolism (1-4)

*   **Y 轴 (因变量/指标)**:
    *   **Survival**: Age (寿命)
    *   **Economic**: Utility (效用), Wealth (绝对财富)
    *   **Efficiency**: Utility per Age (单位时间积累效率)
    *   **Equality**: Gini Coefficient (组内不平等)

### 2. 拟生成的图表清单 (Visualization Plan)

#### **第一组：性格与生存策略 (Persona & Survival)**
1.  **生存曲线对比 (Survival Curves)**:
    *   画出 A/B/C/D 四条线的存活率随 Tick 变化的曲线。
    *   *目的*: 看谁死得快（前期死亡率），谁能活到最后（长尾生存率）。
2.  **代谢率淘汰图 (Metabolism Selection by Persona)**:
    *   分面图 (Facet Grid): 每个 Persona 一张图。
    *   展示初始代谢率 vs 幸存者代谢率。
    *   *假设*: 也许 D (冒险者) 能容忍更高的代谢率？或者 A (保守者) 只有低代谢才能活？

#### **第二组：财富与能力 (Wealth & Ability)**
3.  **视野-财富热力图 (Vision-Wealth Heatmap)**:
    *   X轴=Vision, Y轴=Persona, 颜色=Mean Utility。
    *   *目的*: 回答“高视野对哪种性格加成最大？”（也许规划者 B 最吃视野红利）。
4.  **贫富分化箱线图 (Inequality Boxplot)**:
    *   展示每个 Persona 内部的财富分布离散度。
    *   *目的*: D 组内部是否贫富差距最大（赢家通吃）？A 组是否最均富？

#### **第三组：福利经济学 (Welfare Analysis)**
5.  **福利-寿命散点图 (Welfare-Age Scatter)**:
    *   每个点代表一个 Agent，颜色代表 Persona。
    *   *目的*: 寻找“幸福且长寿”的帕累托前沿 (Pareto Frontier)。
6.  **资源利用效率 (Efficiency Barplot)**:
    *   Y轴 = Total Utility / Age。
    *   *目的*: 谁是最高效的资源转化机器？（也许 D 活得短，但活着的每一天都很爽？）

### 3. 执行计划
1.  **数据加载**: 复用之前的 50 组实验数据（如果不复用，需重新运行，建议复用以节省时间）。
2.  **编写分析脚本 `scripts/analyze_comprehensive.py`**:
    *   使用 `seaborn` 和 `matplotlib` 实现上述 6 类图表。
    *   计算并输出详细的统计表格。
3.  **生成全景报告**: 将所有图表整合到一个 Markdown 报告中，配合深度解读。

这个方案将把您的研究从“谁赢了”提升到“为什么赢”以及“赢得怎么样的”机制层面。