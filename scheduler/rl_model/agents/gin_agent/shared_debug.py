from __future__ import annotations

from typing import Optional
import threading
import numpy as np

# Simple thread-local storage for last action probabilities (flat vector)
_tl = threading.local()


def set_last_action_probs(probs: np.ndarray) -> None:
    """Store the last action probability vector (flat) for debugging.
    Expected shape: (num_tasks * num_vms,)
    """
    _tl.last_action_probs = np.asarray(probs, dtype=float)


def get_last_action_probs() -> Optional[np.ndarray]:
    return getattr(_tl, "last_action_probs", None)
