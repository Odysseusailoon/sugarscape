from redblackbench.sugarscape.config import SugarscapeConfig
from redblackbench.sugarscape.agent import SugarAgent
from redblackbench.sugarscape.environment import SugarEnvironment
from redblackbench.sugarscape.simulation import SugarSimulation
from redblackbench.sugarscape.welfare import WelfareCalculator

# Import plotting module only if matplotlib is available
try:
    from redblackbench.sugarscape.welfare_plots import WelfarePlotter
    __all__ = [
        "SugarscapeConfig",
        "SugarAgent",
        "SugarEnvironment",
        "SugarSimulation",
        "WelfareCalculator",
        "WelfarePlotter"
    ]
except ImportError:
    WelfarePlotter = None
    __all__ = [
        "SugarscapeConfig",
        "SugarAgent",
        "SugarEnvironment",
        "SugarSimulation",
        "WelfareCalculator"
    ]
