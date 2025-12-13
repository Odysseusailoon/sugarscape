为了实现更合理、结构化的实验数据存储，并扩展宏观指标的量化与可视化，我重新设计了如下方案。

### 1. 目录结构设计 (优化版)

我们将采用分层结构，以 `results/sugarscape/` 为根目录，区分模型类型和具体实验：

```
results/sugarscape/
├── baseline_no_llm/                  # 实验类别（无 LLM 基线）
│   ├── experiment_20231027_103001/   # 单次实验目录（时间戳+ID）
│   │   ├── config.json               # 完整配置参数
│   │   ├── metrics.csv               # 核心指标的时间序列 (CSV 方便读取)
│   │   ├── detailed_data.json        # 详细数据 (Agent 状态快照等)
│   │   ├── plots/                    # 生成的图表
│   │   │   ├── wealth_distribution.png
│   │   │   ├── lorenz_curve.png
│   │   │   ├── inequality_series.png
│   │   │   ├── spatial_clustering.png # 新增：空间聚集
│   │   │   ├── survival_curve.png     # 新增：生存曲线
│   │   │   └── mobility_stats.png     # 新增：迁移统计
│   │   └── summary.txt               # 简要文本报告
│   └── ...
├── gpt4_agents/                      # 实验类别（LLM 模型）
│   ├── ...
└── ...
```

### 2. 核心指标量化扩展

除了基础的 Gini 系数，我们将新增以下量化指标的计算逻辑（在 `simulation.py` 或 `analysis.py` 中实现）：

1.  **空间聚集 (Spatial Clustering)**:
    *   **Moran's I (莫兰指数)**: 衡量财富或人口的空间自相关性（富人是否扎堆）。
    *   **局部密度**: 计算 Agent 周围 R 范围内邻居数量的均值。

2.  **生存曲线 (Survival Analysis)**:
    *   **人口数量**: 随时间变化（已有）。
    *   **平均寿命**: 当前存活 Agent 的平均年龄。
    *   **死亡率**: 每 50 步的死亡人数统计。

3.  **迁移/移动性 (Mobility)**:
    *   **平均移动距离**: Agent 每一跳的平均距离。
    *   **探索率**: Agent 历史访问过的不同格子数量（需要 Agent 记录路径历史）。
    *   **扩散半径**: 当前位置相对于出生点的距离。

### 3. 代码实施计划

1.  **创建 `redblackbench/sugarscape/experiment.py`**:
    *   实现 `ExperimentLogger` 类，负责目录管理和数据持久化。
    *   实现 `MetricsCalculator` 类，封装 Moran's I 等复杂指标的计算。

2.  **增强 `SugarAgent`**:
    *   增加 `metrics` 属性，记录 `trajectory` (路径历史) 用于计算迁移指标。

3.  **修改 `SugarSimulation`**:
    *   在 `step()` 中集成新的指标采集逻辑。
    *   支持每 N 步自动调用 Logger 保存数据。

4.  **更新 `scripts/visualize_sugarscape.py`**:
    *   适配新的目录结构。
    *   **新增绘图函数**：
        *   `plot_spatial_clustering()`: 绘制 Moran's I 随时间变化。
        *   `plot_survival()`: 绘制人口与平均寿命双轴图。
        *   `plot_mobility()`: 绘制移动距离分布或扩散半径。

### 4. 预期产出

运行脚本后，将生成一个包含完整数据和 6 张关键图表的标准化实验报告文件夹，不仅能看到“贫富分化”，还能看到“阶层固化”（空间隔离）和“生存压力”（迁移模式）。