为了引入更加复杂和异质化的 Agent 行为模式，我计划实施“Persona 系统”。这将允许我们在实验中按比例分配四种不同类型的 Agent（保守、远见、游牧、冒险），并观察它们在同一环境下的竞争结果。

### 1. 配置扩展 (`config.py`)
更新 `SugarscapeConfig`：
*   **Persona 开关**: `enable_personas` (默认 False)。
*   **Persona 比例**: `persona_distribution` (字典，定义 A/B/C/D 的占比)。
*   **Persona 参数**: 定义全局可调的超参数（S*, $\gamma$, $\lambda$, $\kappa$, $\beta$ 等）。

### 2. 核心逻辑重构 (`agent.py`)
我们需要重写 `SugarAgent` 的决策逻辑，特别是移动/择地逻辑 (`_move_and_harvest`)。
*   **引入 `persona` 属性**: 在初始化时分配 A/B/C/D 标签。
*   **通用评分函数框架**: 创建一个 `calculate_score(candidate_cell, env)` 方法，根据 Agent 的 Persona 类型调用不同的评分公式。
*   **辅助计算**:
    *   `local_density`: 计算目标格子周围（如 Von Neumann 邻域）的 Agent 数量。
    *   `local_regen_rate`: 获取目标格子的资源再生速率（需要环境提供接口或 Agent 记忆）。
    *   `novelty`: 简单实现用 `dist`，或者如果有记忆模块，用“未访问过”作为奖励。

### 3. 实现四种 Persona 逻辑
在 `agent.py` 中实现具体的评分公式：

*   **Type A (保守屯粮者)**:
    *   **生存优先**: 若 `wealth < S*`，极大化 `sugar - 0.8*dist - kappa*density`，且严格避免长距离移动（风险厌恶）。
    *   **富足时**: 正常评分，但探索奖励极低 ($\lambda * dist$)。

*   **Type B (远见规划者)**:
    *   **重视未来**: 评分中 `beta * local_regen_rate` 权重高 (0.75)。
    *   **稳健探索**: `0.15 * lambda * dist`。
    *   **生存恢复**: 低于 S* 时切换回短视模式（只看眼前糖分）。

*   **Type C (探索游牧者)**:
    *   **重视新奇**: `0.55 * lambda * novelty` 权重极高。
    *   **轻视距离**: `dist` 惩罚较低 (-0.25)。
    *   **底线思维**: 只要不死 (`wealth > 0`) 就敢浪。

*   **Type D (冒险投机者)**:
    *   **极大化当前收益**: `1.15 * sugar_now`。
    *   **轻视风险**: 对拥挤 `density` 和距离 `dist` 不太敏感。
    *   **硬约束**: 仅在必死时才回避（生死看淡，不服就干）。

### 4. 环境与仿真更新
*   **环境**: 确保 `SugarEnvironment` 提供查询 `local_density` 和 `regen_rate` 的接口。
*   **仿真**: 在初始化人口时，根据配置的比例 (`[0.36, 0.29, 0.21, 0.14]`) 分配 Persona。

### 5. 实验脚本 (`scripts/run_persona_experiment.py`)
*   创建一个对比实验，开启 `enable_personas`。
*   **分析重点**:
    *   **分类型生存率**: 哪种性格活得最久？（是保守的 A 还是远见的 B？）
    *   **分类型财富**: 谁最富有？（冒险的 D 是否有暴富的机会？）
    *   **空间分布**: 游牧者 C 是否分布更分散？

这个计划将把模拟从“简单的贪婪算法”提升到“异质策略博弈”，让我们能观察到性格与环境的匹配效应。