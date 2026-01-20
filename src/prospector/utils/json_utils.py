from __future__ import annotations

from typing import Any
import json

import numpy as np

try:
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jnp = None  # type: ignore


def to_jsonable(x: Any) -> Any:
    """
    Recursively convert common scientific Python objects into JSON-serializable
    Python primitives (dict/list/float/int/bool/None/str).

    Handles:
      - numpy arrays + scalars
      - jax arrays
      - nested dict/list/tuple
    """
    # None / primitives
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    # NumPy scalar -> Python scalar
    if isinstance(x, np.generic):
        return x.item()

    # NumPy array -> list
    if isinstance(x, np.ndarray):
        return x.tolist()

    # JAX array -> list (via np.asarray)
    if jnp is not None:
        # JAX DeviceArray is not a public class; safest is duck-typing
        if hasattr(x, "shape") and hasattr(x, "dtype") and "jax" in type(x).__module__:
            return np.asarray(x).tolist()

    # dict
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}

    # list/tuple
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]

    # Fallback: string representation (keeps json.dump from exploding)
    return str(x)


def dump_json(path, obj: Any, *, indent: int = 2) -> None:
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=indent)
