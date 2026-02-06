# IRT - Item Response Theory Library

A production-quality Python library for fitting Item Response Theory (IRT) models using Marginal Maximum Likelihood (MML-EM) and Joint Maximum Likelihood (JMLE) estimation methods.

## Features

- **Binary models**: Rasch (1PL), Two-Parameter Logistic (2PL), Three-Parameter Logistic (3PL)
- **Polytomous models**: PCM, RSM, GRM, GPCM, NRM (MML-EM only)
- **Estimators**: MML-EM and JMLE (Rasch only)
- **Scoring**: EAP, MAP, and MLE ability estimation
- **Diagnostics**: Infit/outfit statistics, point-biserial correlations
- **Robust**: Handles missing data, extreme scores, and convergence issues
- **Minimal dependencies**: Only NumPy and SciPy required

## Installation

```bash
pip install numpy scipy

# Optional: for DataFrame support
pip install pandas
```

## Quick Start

```python
import numpy as np
from irt import fit

# Create response data (N persons x J items)
np.random.seed(42)
X = np.random.binomial(1, 0.6, size=(100, 20)).astype(float)

# Fit a Rasch model using MML-EM
result = fit(X, model="rasch", estimator="mml_em")

print(f"Converged: {result.converged}")
print(f"Log-likelihood: {result.loglik:.2f}")
print(f"Item difficulties: {result.params['b'][:5]}")

# Score persons using EAP
scores = result.score(method="eap")
print(f"Ability estimates: {scores.theta[:5]}")
print(f"Standard errors: {scores.se[:5]}")
```

## Model Parameterization

The library uses the standard IRT parameterization:

```
P(X=1|θ) = 1 / (1 + exp(-a(θ - b)))
```

Where:
- `θ` (theta): Person ability parameter
- `a`: Item discrimination parameter (fixed at 1.0 for Rasch)
- `b`: Item difficulty parameter

For 3PL, a guessing parameter `c` is included:

```
P(X=1|θ) = c + (1 - c) / (1 + exp(-a(θ - b)))
```

## API Reference

### `fit(X, model="rasch", estimator="mml_em", ...)`

Fit an IRT model to response data.

**Parameters:**
- `X`: Response matrix (N × J). Binary: {0, 1, NaN}. Polytomous: {0, 1, …, m-1, NaN}
- `model`: Binary: `"rasch"`, `"2pl"`, `"3pl"`. Polytomous: `"pcm"`, `"rsm"`, `"grm"`, `"gpcm"`, `"nrm"`
- `estimator`: `"mml_em"` or `"jmle"` (JMLE only supports Rasch)
- `technical`: Dict of technical parameters (see below)
- `start`: Starting values `{"a": array, "b": array}`
- `fixed`: Fixed parameter masks `{"a": bool_array, "b": bool_array}`
- `constraints`: Constraints `{"center_b": True/False}`

**Returns:** `FitResult` object

### Technical Parameters

#### MML-EM (defaults)
```python
{
    "quadpts": 61,           # Number of quadrature points
    "theta_lo": -4.0,        # Lower bound of theta grid
    "theta_hi": 4.0,         # Upper bound of theta grid
    "prior": ("normal", 0.0, 1.0),  # Prior distribution
    "max_iter": 200,         # Maximum EM iterations
    "tol": 1e-4,            # Convergence tolerance
    "a_bounds": (0.25, 4.0), # Discrimination bounds (2PL)
    "b_bounds": (-6.0, 6.0), # Difficulty bounds
    "mstep_max_iter": 25,    # M-step iterations
    "c_bounds": (1e-6, 0.35), # Guessing bounds (3PL)
}
```

#### JMLE (defaults)
```python
{
    "max_iter": 50,          # Maximum iterations
    "tol": 1e-4,            # Convergence tolerance
    "nudge": 0.3,           # Nudge for extreme scores
    "theta_bounds": (-6.0, 6.0),
    "b_bounds": (-6.0, 6.0),
}
```

### `FitResult` Object

**Attributes:**
- `model`: Model type (`"rasch"`, `"2pl"`, `"3pl"`, or polytomous)
- `estimator`: Estimation method
- `params`: Dict with `"a"` and `"b"` arrays
- `converged`: Boolean convergence indicator
- `n_iter`: Number of iterations
- `loglik`: Final log-likelihood (MML-EM only)
- `history`: Convergence history
- `item_names`, `person_names`: Labels

**Methods:**
- `score(X=None, method="eap")`: Score persons
- `item_report()`: Item parameter summary (requires pandas)
- `person_report()`: Person summary with abilities (requires pandas)

### `ScoreResult` Object

**Attributes:**
- `theta`: Ability estimates array
- `se`: Standard error array (or None)
- `method`: Scoring method used
- `person_names`: Person labels

**Methods:**
- `to_dataframe()`: Convert to pandas DataFrame

## Scoring Methods

| Method | Description | Extreme Scores | SE Available |
|--------|-------------|----------------|--------------|
| `"eap"` | Expected A Posteriori (posterior mean) | Always finite | Yes |
| `"map"` | Maximum A Posteriori (posterior mode) | Always finite | Yes |
| `"mle"` | Maximum Likelihood Estimate | Infinite (clamped to bounds) | Yes |

## Examples

### Fit 2PL Model

```python
result = fit(X, model="2pl", estimator="mml_em")
print(f"Discriminations: {result.params['a']}")
print(f"Difficulties: {result.params['b']}")
```

### Fit 3PL Model

```python
result = fit(
    X,
    model="3pl",
    estimator="mml_em",
    technical={"c_bounds": (0.05, 0.30)},
)
print(f"Guessing: {result.params['c']}")
```

By default, 3PL uses a weak gamma prior on `c` unless you supply a custom prior.

### Fit Polytomous Model (PCM, GPCM, GRM, etc.)

```python
# Likert-scale data: 5 categories (0–4) per item
X = np.random.randint(0, 5, size=(200, 10)).astype(float)
X[np.random.random(X.shape) < 0.05] = np.nan  # Add missing

result = fit(X, model="pcm")   # Partial Credit Model
# Or: model="gpcm", "grm", "rsm", "nrm"
print(result.item_report())
result.plot_icc(items=[0])     # Category Characteristic Curves
```

See `notebooks/07_polytomous_models.ipynb` for a full demo.

### Custom Technical Parameters

```python
result = fit(
    X,
    model="rasch",
    technical={
        "quadpts": 81,
        "max_iter": 500,
        "tol": 1e-6,
    }
)
```

### JMLE Estimation

```python
result = fit(X, model="rasch", estimator="jmle")
scores = result.score(method="map")  # EAP not available for JMLE
```

### Handle Missing Data

```python
# Missing data represented as NaN
X = np.array([
    [1, 0, np.nan, 1],
    [0, 1, 1, np.nan],
    [1, 1, 0, 0],
])
result = fit(X)  # Works with missing data
```

### Score New Data

```python
# Fit model on training data
result = fit(X_train)

# Score new persons
scores = result.score(X=X_new, method="eap")
```

### Model Diagnostics

```python
from irt.diagnostics import infit_outfit, point_biserial_items

# Get ability estimates
scores = result.score(method="eap")

# Compute fit statistics
item_fit, person_fit = infit_outfit(
    result.X,
    result.mask_obs,
    scores.theta,
    result.params["a"],
    result.params["b"],
)

# Flag misfitting items (infit > 1.5)
print(item_fit[item_fit["infit_ms"] > 1.5])

# Point-biserial correlations
rpb = point_biserial_items(result.X, result.mask_obs, scores.theta)
```

### Model Fit Summary

```python
fit_stats = result.model_fit(method="eap")
print(fit_stats.get("rmsea"), fit_stats.get("cfi"), fit_stats.get("tli"))
print(fit_stats.get("srmsr"), fit_stats.get("q3_mean"), fit_stats.get("q3_max_abs"))

summary = result.summary()
coef_irt = result.coef(irt=True)
coef_slope = result.coef(irt=False)
```

## Identification

### MML-EM
The ability scale is identified through the prior distribution on θ. The default N(0, 1) prior means:
- Population mean ability is 0
- Population SD of ability is 1

### JMLE
The ability scale is identified by centering item difficulties:
- mean(b) = 0

## Algorithm Details

### MML-EM Algorithm

1. **E-step**: Compute posterior P(θ|X) using quadrature
2. **M-step**: Update item parameters to maximize expected complete-data log-likelihood
3. Iterate until convergence (log-likelihood change < tol)

### JMLE Algorithm

1. Update person abilities θ given item parameters
2. Update item difficulties b given abilities
3. Center difficulties for identification
4. Iterate until convergence

## Development

**Full test suite** (runs all tests and generates `QA_REPORT.md`):

```bash
make test
# or
python scripts/run_tests_with_qa_report.py
```

**R validation**: Binary models are validated against R `ltm` and `mirt`; polytomous (PCM, GPCM, GRM, RSM) against R `mirt`. Reference files are pre-committed; tests run without R. To regenerate polytomous refs: `make refs`. See `CONTRIBUTING.md`.

## References

- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46(4), 443-459.
- Wright, B. D., & Stone, M. H. (1979). *Best Test Design*. MESA Press.
- Baker, F. B., & Kim, S. H. (2004). *Item Response Theory: Parameter Estimation Techniques*. CRC Press.

## License

No license specified. All rights reserved by default.
