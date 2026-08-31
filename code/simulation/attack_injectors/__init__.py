"""Attack injectors. Each plugin mutates telemetry and/or commands."""
from .base import AttackInjector
from .fdi import FDIInjector
from .replay import ReplayInjector
from .command_spoof import CommandSpoofInjector
from .dos import DoSInjector
from .benign_echo import BenignEchoInjector
from .benign_command import BenignCommandInjector

REGISTRY = {
    "fdi": FDIInjector,
    "replay": ReplayInjector,
    "command_spoof": CommandSpoofInjector,
    "dos": DoSInjector,
    # benign protocol quirks; active() is always False so they never
    # contribute to ground-truth attack windows.
    "benign_echo": BenignEchoInjector,
    "benign_command": BenignCommandInjector,
}

__all__ = ["AttackInjector", "REGISTRY"]
