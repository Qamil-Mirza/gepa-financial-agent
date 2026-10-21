"""PitStop: a supervisor that detects drift in multi-agent research workflows."""
__version__ = "0.1.0"

from .sdk.context import agent, workflow  # noqa: E402
from .sdk.errors import BudgetExceeded, Halted  # noqa: E402
from .sdk.prompt import prompt  # noqa: E402
from .sdk.runtime import init  # noqa: E402
from .sdk.tool import tool  # noqa: E402
from .sdk.wrap import wrap  # noqa: E402

__all__ = ["init", "wrap", "tool", "prompt", "workflow", "agent", "Halted", "BudgetExceeded", "__version__"]
