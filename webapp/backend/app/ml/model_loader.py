"""Loader for the trained species distribution model bundle.

The bundle was produced by
``new_species_model/build_species_distribution_model.py`` on the ``peter_code``
branch.  It is a ``joblib`` pickle that contains:

* ``feature_columns`` - list of 13 feature names expected by every model
* ``species_columns`` - 369 snake_case species names
* ``models``          - dict mapping species name -> sklearn Pipeline or
                        ``ConstantProbabilityModel`` instance
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from ..config import MODEL_BUNDLE_PATH

log = logging.getLogger(__name__)


@dataclass
class ConstantProbabilityModel:
    """Shim matching the class pickled in the training script.

    The original class was defined at module scope in
    ``build_species_distribution_model.py``; we need an importable copy here so
    ``joblib.load`` can resolve the reference during unpickling.
    """

    probability: float

    def predict_proba(self, x) -> np.ndarray:  # pragma: no cover - trivial
        probs = np.full(len(x), self.probability, dtype=float)
        return np.column_stack([1.0 - probs, probs])


# Register the shim under every module path sklearn's unpickler might look up.
# The bundle was pickled from ``__main__`` on the training machine, so we have
# to inject the class there too.
sys.modules.setdefault("build_species_distribution_model", sys.modules[__name__])
import __main__ as _main_module  # noqa: E402
if not hasattr(_main_module, "ConstantProbabilityModel"):
    _main_module.ConstantProbabilityModel = ConstantProbabilityModel


class SpeciesDistributionModel:
    """High-level wrapper around the joblib bundle."""

    def __init__(self, bundle: dict) -> None:
        self.feature_columns: List[str] = list(bundle["feature_columns"])
        self.species_columns: List[str] = list(bundle["species_columns"])
        self.models: Dict[str, object] = dict(bundle["models"])

    @classmethod
    def load(cls, path=MODEL_BUNDLE_PATH) -> "SpeciesDistributionModel":
        log.info("Loading species distribution model from %s", path)
        bundle = joblib.load(path)
        return cls(bundle)

    # ---- prediction ------------------------------------------------------
    def predict_species(self, species: str, features: pd.DataFrame) -> np.ndarray:
        model = self.models[species]
        return model.predict_proba(features[self.feature_columns])[:, 1]

    def predict_all(self, features: pd.DataFrame) -> pd.DataFrame:
        """Run every species classifier on the given feature frame.

        Returns a DataFrame with one column per species and the same index as
        ``features``.
        """
        x = features[self.feature_columns]
        out: Dict[str, np.ndarray] = {}
        for species, model in self.models.items():
            out[species] = model.predict_proba(x)[:, 1]
        return pd.DataFrame(out, index=features.index)
