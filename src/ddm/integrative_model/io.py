from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_training_history(history: Any, path: str | Path) -> str:
    """
    Save BayesFlow (or Keras-like) history object as pickle.
    Expects `history.history` to be a dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = getattr(history, "history", history)
    with path.open("wb") as f:
        pickle.dump(payload, f)

    return str(path)


def load_training_history(path: str | Path) -> dict:
    """Load training history dict from a pickle file."""
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)
