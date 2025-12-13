为了实现支持单资源（Sugar）和双资源（Sugar + Spice）场景的 Persona 系统，我更新了实施计划。核心在于**评分函数（Scoring Function）的泛化**，使其能根据当前环境配置动态计算“价值”。

### 1. 环境与配置 (`config.py` & `environment.py`)
*   **配置**: 已添加 `enable_personas` 及 A/B/C/D 比例。
*   **环境增强**:
    *   `get_local_density(pos, radius=1)`: 计算目标格周围的拥挤度（0.0-1.0）。
    *   `get_site_quality(pos)`: 返回该格子的长期价值。
        *   单资源: `sugar_capacity`
        *   双资源: `sugar_capacity + spice_capacity` (或加权)

### 2. Agent 逻辑重构 (`agent.py`)
我们需要将硬编码的“找最多糖”逻辑替换为基于 Persona 的通用评分系统。

#### **泛化价值计算 (`_calculate_utility`)**
*   **Single Resource**: $U = wealth + sugar\_at\_pos$
*   **Dual Resource**: $U = CobbDouglas(wealth + sugar\_at\_pos, spice + spice\_at\_pos)$

#### **泛化生存阈值 (`_is_survival_mode`)**
*   **Single Resource**: `wealth < S*`
*   **Dual Resource**: `wealth < S*_s` OR `spice < S*_p`

#### **Persona 评分公式 (适配双模式)**
公式中的 `sugar_now` 将被替换为 `utility_gain` (即移动带来的效用增量或绝对效用)。

*   **Type A (保守者)**:
    *   `score = utility_gain - 0.8*dist - κ*density`
    *   *双资源特性*: 极度厌恶任一资源短缺，优先去能补充短板的地方。
*   **Type B (规划者)**:
    *   `score = 0.75*utility_gain + 0.75*β*site_quality ...`
    *   *双资源特性*: 会去那些虽然当前没多少糖/香料，但**容量高**（再生潜力大）的地方蹲守。
*   **Type C (游牧者)**:
    *   `score = ... + 0.55*λ*novelty`
    *   *双资源特性*: 喜欢探索未知的地图区域，可能会偶然发现新的资源富集区。
*   **Type D (冒险者)**:
    *   `score = 1.15*utility_gain ...`
    *   *双资源特性*: 只看当前哪里效用最高（通常是资源最丰富的地方），不顾拥挤和距离，敢于冲进红海竞争。

### 3. 实验脚本 (`scripts/run_persona_experiment.py`)
我们将运行两组实验来验证 Persona 的效果：
1.  **Sugar Only**: 观察 A/B/C/D 在经典环境下的生存率。
2.  **Sugar + Spice**: 观察在双资源压力下，哪种性格更具优势（例如规划者 B 可能更能平衡两种资源需求）。

这个设计确保了代码的一致性，无论是否开启 Spice，Persona 的性格特征（保守/激进/远见）都能通过数学公式体现出来。