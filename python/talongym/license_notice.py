"""GPL startup notice for TalonGym entry points."""

from __future__ import annotations

import sys

from talongym import __version__
from talongym.credits import CREDIT_LINE

NOTICE = (
    f"TalonGym {__version__} — Copyright (C) 2026 TalonGym contributors.\n"
    f"{CREDIT_LINE}\n"
    "Licensed under GNU GPL v3 or later; see LICENSE in the repository.\n"
    "This program comes with ABSOLUTELY NO WARRANTY."
)

_emitted = False


def emit_license_notice(stream=None) -> None:
    """Print the GPL notice once per process at program startup."""
    global _emitted
    if _emitted:
        return
    _emitted = True
    out = stream or sys.stderr
    print(NOTICE, file=out, flush=True)
