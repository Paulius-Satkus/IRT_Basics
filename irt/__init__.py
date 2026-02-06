"""
IRT - Item Response Theory Library

A Python library for fitting Item Response Theory models using
Marginal Maximum Likelihood (MML-EM) and Joint Maximum Likelihood (JMLE)
estimation methods.

Supported Models
----------------
Binary: Rasch (1PL), 2PL, 3PL
Polytomous: PCM, RSM, GRM, GPCM, NRM

Supported Estimators
--------------------
- MML-EM: Marginal Maximum Likelihood via Expectation-Maximization
- JMLE: Joint Maximum Likelihood Estimation (Rasch only)

Scoring Methods
---------------
- EAP: Expected A Posteriori (posterior mean)
- MAP: Maximum A Posteriori (posterior mode)
- MLE: Maximum Likelihood Estimate

Quick Start
-----------
>>> import numpy as np
>>> from irt import fit
>>>
>>> # Create response data (N persons x J items)
>>> X = np.random.binomial(1, 0.6, size=(100, 20)).astype(float)
>>>
>>> # Fit a Rasch model
>>> result = fit(X, model="rasch", estimator="mml_em")
>>>
>>> # Get item parameters
>>> print(result.params["b"])  # Item difficulties
>>>
>>> # Score persons
>>> scores = result.score(method="eap")
>>> print(scores.theta)  # Ability estimates
>>> print(scores.se)     # Standard errors
"""

from .api import fit
from .types import FitResult, ScoreResult

# Expose plotting and diagnostics modules for direct import
from . import plotting
from . import diagnostics

__version__ = "0.1.0"
__all__ = [
    "fit",
    "FitResult",
    "ScoreResult",
    "plotting",
    "diagnostics",
    "__version__",
]
