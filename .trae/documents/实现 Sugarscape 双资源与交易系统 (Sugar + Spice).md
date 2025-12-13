为了实现 Sugarscape 的双资源（Sugar + Spice）模型，并支持**允许交易 vs 禁止交易**的对比实验，我更新了实施计划。核心在于将“交易”作为一个独立的可配置模块，并在兼容旧版单资源模型的基础上，增加双资源和市场机制。

### 1. 配置扩展 (`config.py`)
在 `SugarscapeConfig` 中增加双资源和交易开关：
*   **资源开关**: `enable_spice` (默认 False)。开启后环境生成 Spice，Agent 拥有 Spice 属性。
*   **交易开关**: `enable_trade` (默认 False)。开启后 Agent 在每步移动后会尝试与邻居交易。
*   **参数**:
    *   `max_spice_capacity`, `spice_growback_rate`。
    *   `initial_spice_range`, `metabolism_spice_range`。

### 2. 环境升级 (`environment.py`)
*   **双层资源网格**: 新增 `spice_capacity` 和 `spice_amount`。
*   **互补地形生成**: 实现经典的 Epstein & Axtell 地形：
    *   Sugar 峰值在 (15, 15) 和 (35, 35)。
    *   Spice 峰值在 (15, 35) 和 (35, 15)（即与 Sugar 峰错开，制造“资源错配”）。
*   **兼容性**: 只有当 `enable_spice=True` 时才初始化 Spice 层。

### 3. Agent 升级 (`agent.py`)
*   **属性**: 增加 `spice` (库存) 和 `m_p` (代谢率)。
*   **福利函数 (Welfare)**:
    *   若 `enable_spice=True`: 使用 Cobb-Douglas $W = w_s^{m_s} w_p^{m_p}$。
    *   若 `enable_spice=False`: 退化为 $W = w_s$ (仅看糖)。
*   **移动逻辑**:
    *   **目标函数**: 寻找能最大化 $W$ 的格子（即去缺的那种资源多的地方）。
*   **MRS 计算**: 实现 $MRS = \frac{m_s w_p}{m_p w_s}$ 供交易使用。

### 4. 交易模块 (`trade.py`) - *核心新增*
实现标准的双边讨价还价算法：
*   **输入**: 两个 Agent。
*   **流程**:
    1.  计算双方 MRS。
    2.  确定买卖方向（MRS 高者买糖）。
    3.  计算几何平均价格 $p = \sqrt{MRS_a \cdot MRS_b}$。
    4.  如果交易能提高双方福利 $W$，则交换 1 单位（或 $\epsilon$ 量）。
    5.  重复直到 MRS 收敛或无法获利。

### 5. 仿真与对比脚本
*   **修改 `SugarSimulation`**:
    *   在 `step()` 中加入 `Trade Phase`: 如果 `config.enable_trade` 为真，打乱顺序让 Agent 与邻居尝试交易。
    *   **统计扩展**: 记录“平均交易价格”、“总交易量”、“福利 (Welfare) Gini 系数”（比单纯的财富 Gini 更能反映双资源下的不平等）。

*   **创建 `scripts/compare_trade_effect.py`**:
    *   **实验 A (No Trade)**: `enable_spice=True`, `enable_trade=False`。Agent 只能靠移动自给自足，容易因一种资源耗尽而死。
    *   **实验 B (With Trade)**: `enable_spice=True`, `enable_trade=True`。Agent 可以通过交换互通有无。
    *   **对比分析**:
        *   **生存率**: 交易是否降低了死亡率？
        *   **福利分布**: 交易是否让社会整体福利更高？是否减少了极度贫困？
        *   **价格曲线**: 绘制市场价格随时间的演变。

### 6. 执行顺序
1.  更新 Config 和 Environment (支持 Spice)。
2.  更新 Agent (支持 Cobb-Douglas 和 MRS)。
3.  实现 Trade 逻辑。
4.  编写并运行对比实验脚本。