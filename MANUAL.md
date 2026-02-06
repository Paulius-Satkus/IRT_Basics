# IRT Python Library - User Manual

A lightweight Python library for Item Response Theory (IRT) analysis, implementing binary models (Rasch, 2PL, 3PL) and polytomous models (PCM, RSM, GRM, GPCM, NRM) with multiple estimation methods.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [API Reference](#api-reference)
5. [Examples](#examples)
6. [Technical Details](#technical-details)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Requirements

- Python 3.9+
- NumPy
- SciPy
- pandas (optional, for DataFrame support)

### Install from source

```bash
cd "Item Response Theory Python"
pip install -e .
```

Or install dependencies directly:

```bash
pip install numpy scipy pandas
```

---

## Quick Start

### Basic Usage

```python
import numpy as np
from irt import fit

# Create response data (N persons × J items)
# Values: 0 = incorrect, 1 = correct, NaN = missing
np.random.seed(42)
X = np.random.binomial(1, 0.6, size=(500, 20)).astype(float)

# Fit a Rasch model
result = fit(X, model="rasch")

print(result)
# FitResult(model='rasch', estimator='mml_em', n_items=20, n_persons=500, 
#           n_iter=15, converged, loglik=-5832.41)

# View item parameters
print("Item difficulties:", result.params["b"][:5])

# Get person ability estimates
scores = result.score(method="eap")
print("Ability estimates:", scores.theta[:5])
print("Standard errors:", scores.se[:5])
```

### Using pandas DataFrames

```python
import pandas as pd
from irt import fit

# Load your data
df = pd.read_csv("responses.csv", index_col=0)

# Fit model - column names become item names, index becomes person names
result = fit(df, model="2pl")

# Generate reports
print(result.item_report())
print(result.person_report())
```

---

## Core Concepts

### IRT Models

This library implements binary and polytomous unidimensional IRT models.

#### Rasch Model (1PL)

The simplest IRT model where all items have equal discrimination (a = 1):

$$P(X = 1 | \theta) = \frac{1}{1 + e^{-(\theta - b)}}$$

- **θ (theta)**: Person ability parameter
- **b**: Item difficulty parameter

Use when you want a simpler model or when items are expected to have similar discriminating power.

#### Two-Parameter Logistic Model (2PL)

Extends the Rasch model by estimating item discrimination:

$$P(X = 1 | \theta) = \frac{1}{1 + e^{-a(\theta - b)}}$$

- **θ (theta)**: Person ability parameter  
- **a**: Item discrimination parameter (a > 0)
- **b**: Item difficulty parameter

Use when items may differ in how well they distinguish between ability levels.

#### Three-Parameter Logistic Model (3PL)

Adds a guessing parameter to account for non-zero lower asymptotes:

$$P(X = 1 | \theta) = c + (1 - c)\frac{1}{1 + e^{-a(\theta - b)}}$$

- **θ (theta)**: Person ability parameter  
- **a**: Item discrimination parameter (a > 0)
- **b**: Item difficulty parameter
- **c**: Guessing parameter (0 ≤ c < 1)

Use when items are multiple-choice and guessing is expected.

#### Polytomous Models

For Likert-scale, partial credit, or multi-category items, the library supports:

| Model | Description | Use when |
|-------|-------------|----------|
| **PCM** | Partial Credit Model (Masters 1982) | Ordered categories, Rasch-like |
| **RSM** | Rating Scale Model (Andrich 1978) | Same rating scale across items |
| **GRM** | Graded Response Model (Samejima 1969) | Cumulative boundaries, discrimination varies |
| **GPCM** | Generalized Partial Credit Model (Muraki 1992) | PCM with item discrimination |
| **NRM** | Nominal Response Model (Bock 1972) | Unordered categories |

Polytomous models use integer responses 0, 1, …, m−1 per item. Fit with `fit(X, model="pcm")` etc. See `notebooks/07_polytomous_models.ipynb` for examples.

### Estimation Methods

#### MML-EM (Marginal Maximum Likelihood via EM)

- **Best for**: Most situations, especially with missing data
- **How it works**: Treats person abilities as random effects from a prior distribution (typically N(0,1)), integrates them out
- **Identification**: Via the prior distribution
- **Supports**: Rasch, 2PL, 3PL (binary) and PCM, RSM, GRM, GPCM, NRM (polytomous)

#### JMLE (Joint Maximum Likelihood Estimation)

- **Best for**: Rasch models when you want person parameters estimated directly
- **How it works**: Estimates item and person parameters simultaneously
- **Identification**: By centering item difficulties (mean = 0)
- **Supports**: Rasch model only

### Scoring Methods

After fitting a model, estimate person abilities using:

| Method | Description | Best for |
|--------|-------------|----------|
| **EAP** | Expected A Posteriori (posterior mean) | General use, most stable |
| **MAP** | Maximum A Posteriori (posterior mode) | When you want the most likely ability |
| **MLE** | Maximum Likelihood | When you don't want prior influence |

---

## API Reference

### `fit()` - Main Fitting Function

```python
from irt import fit

result = fit(
    X,                          # Response data (N × J matrix)
    model="rasch",              # "rasch", "2pl", or "3pl"
    estimator="mml_em",         # "mml_em" or "jmle"
    technical=None,             # Override default settings
    start=None,                 # Starting values
    fixed=None,                 # Fixed parameters
    priors=None,                # Item parameter priors
    constraints=None            # Constraints
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | array-like or DataFrame | Response matrix (N × J). Binary: 0, 1, NaN. Polytomous: 0..m−1, NaN |
| `model` | str | Binary: `"rasch"`, `"2pl"`, `"3pl"`. Polytomous: `"pcm"`, `"rsm"`, `"grm"`, `"gpcm"`, `"nrm"` |
| `estimator` | str | `"mml_em"` or `"jmle"` (Rasch only) |
| `technical` | dict | Technical parameters (see below) |
| `start` | dict | Starting values `{"a": array, "b": array, "c": array}` |
| `fixed` | dict | Boolean arrays for fixed params `{"a": bool_array, "b": bool_array, "c": bool_array}` |
| `priors` | dict or DataFrame | Item parameter priors (per-item or grouped) |
| `constraints` | dict | Constraints like `{"center_b": True}` |

#### Technical Parameters for MML-EM

```python
technical = {
    "quadpts": 61,              # Number of quadrature points
    "theta_lo": -4.0,           # Lower bound of theta grid
    "theta_hi": 4.0,            # Upper bound of theta grid
    "prior": ("normal", 0, 1),  # Prior: (type, mean, sd)
    "max_iter": 200,            # Maximum EM iterations
    "tol": 1e-4,                # Convergence tolerance
    "a_bounds": (0.25, 4.0),    # Discrimination bounds
    "b_bounds": (-6.0, 6.0),    # Difficulty bounds
    "mstep_max_iter": 25,       # M-step iterations
    "c_bounds": (1e-6, 0.35)    # Guessing bounds (3PL)
}
```

#### Technical Parameters for JMLE

```python
technical = {
    "max_iter": 50,             # Maximum iterations
    "tol": 1e-4,                # Convergence tolerance
    "jmle_method": "newton",    # Update method
    "nudge": 0.3,               # Nudge for extreme scores
    "theta_bounds": (-6.0, 6.0),# Theta bounds
    "b_bounds": (-6.0, 6.0)     # Difficulty bounds
}
```

#### Item Parameter Priors

You can provide priors per item and per parameter (`a`, `b`, `c`). Supported
distributions: **normal**, **lognormal**, **gamma**, **beta**. For 3PL, a
weak gamma prior is applied to `c` by default unless you supply a custom prior.

**Dict-style (grouped + per-item):**

```python
priors = {
    "a": [
        {"dist": "lognormal", "mean": 0.0, "sd": 0.5, "items": ["item_0", "item_1"]},
        {"dist": "gamma", "shape": 2.0, "scale": 0.4, "items": ["item_2"]},
    ],
    "b": {
        "item_3": {"dist": "normal", "mean": 0.0, "sd": 1.0},
        "item_4": {"dist": "normal", "mean": 0.5, "sd": 0.8},
    },
    "c": {
        "item_0": {"dist": "beta", "alpha": 2.0, "beta": 8.0},
    },
}
result = fit(X, model="3pl", priors=priors)
```

**DataFrame-style:**

```python
import pandas as pd

df_priors = pd.DataFrame(
    [
        {"item": "item_0", "param": "a", "dist": "lognormal", "mean": 0.0, "sd": 0.5},
        {"item": "item_1", "param": "a", "dist": "gamma", "shape": 2.0, "scale": 0.4},
        {"item": "item_3", "param": "b", "dist": "normal", "mean": 0.0, "sd": 1.0},
        {"item": "item_0", "param": "c", "dist": "beta", "alpha": 2.0, "beta": 8.0},
    ]
)
result = fit(X, model="3pl", priors=df_priors)
```

Notes:
- Distributions with restricted support (lognormal/gamma/beta) must be compatible
  with parameter bounds in `technical` (e.g., `b_bounds` must be positive).

#### Returns: `FitResult`

### `FitResult` - Model Results

```python
result.model           # str: "rasch", "2pl", "3pl", or polytomous ("pcm", etc.)
result.estimator       # str: "mml_em" or "jmle"
result.params          # dict: {"a": array, "b": array, "c": array}
result.converged       # bool: Did estimation converge?
result.n_iter          # int: Number of iterations
result.loglik          # float: Log-likelihood (MML-EM only)
result.history         # dict: Convergence history
result.item_names      # list: Item identifiers
result.person_names    # list: Person identifiers
```

#### Methods

```python
# Score persons
scores = result.score(X=None, method="eap")

# Generate reports (requires pandas)
item_df = result.item_report()      # Item parameters and stats
person_df = result.person_report()  # Person scores and stats
```

### Model Fit Summary and Residual Diagnostics

Use `model_fit()` to compute fit indices and residual summaries. This provides
M2/RMSEA along with additional indices (CFI/TLI/SRMSR) and Q3 residual
correlation summaries for local dependence.

```python
fit_stats = result.model_fit(method="eap")
print(fit_stats["m2"], fit_stats.get("rmsea"))
print(fit_stats.get("cfi"), fit_stats.get("tli"), fit_stats.get("srmsr"))
print(fit_stats.get("q3_mean"), fit_stats.get("q3_max_abs"))
```

For concise reporting, `summary()` wraps `model_fit()` and adds model metadata,
and `coef()` returns item parameters in common formats:

```python
summary = result.summary()
coef_irt = result.coef(irt=True)
coef_slope = result.coef(irt=False)
```

### `ScoreResult` - Ability Estimates

```python
scores.theta           # np.ndarray: Ability estimates
scores.se              # np.ndarray: Standard errors (if available)
scores.method          # str: Scoring method used
scores.person_names    # list: Person identifiers

# Convert to DataFrame
df = scores.to_dataframe()
```

---

## Examples

### Example 1: Basic Rasch Analysis

```python
import numpy as np
from irt import fit

# Simulate data: 500 persons, 25 items
np.random.seed(123)
N, J = 500, 25

# True parameters
theta_true = np.random.standard_normal(N)
b_true = np.linspace(-2, 2, J)

# Generate responses
from scipy.special import expit
p = expit(theta_true[:, None] - b_true)
X = (np.random.rand(N, J) < p).astype(float)

# Fit Rasch model
result = fit(X, model="rasch", estimator="mml_em")

print(f"Converged: {result.converged}")
print(f"Iterations: {result.n_iter}")
print(f"Log-likelihood: {result.loglik:.2f}")

# Compare estimated vs true difficulties
import matplotlib.pyplot as plt
plt.scatter(b_true, result.params["b"])
plt.xlabel("True difficulty")
plt.ylabel("Estimated difficulty")
plt.plot([-3, 3], [-3, 3], 'r--')
plt.title("Parameter Recovery")
plt.show()
```

### Example 2: 2PL Model with Reports

```python
import pandas as pd
from irt import fit

# Load data
df = pd.DataFrame({
    "item1": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    "item2": [1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    "item3": [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
    "item4": [1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
    "item5": [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
}, index=[f"person_{i}" for i in range(10)])

# Fit 2PL model
result = fit(df, model="2pl")

# Item report
print("\n=== Item Parameters ===")
print(result.item_report())

# Person report with ability estimates
print("\n=== Person Report ===")
print(result.person_report())
```

### Example 3: Handling Missing Data

```python
import numpy as np
from irt import fit

# Create data with missing responses (NaN)
X = np.array([
    [1, 0, 1, np.nan, 1],
    [0, 1, np.nan, 1, 0],
    [1, 1, 1, 0, np.nan],
    [np.nan, 0, 0, 1, 1],
    [1, np.nan, 1, 1, 0],
])

# MML-EM handles missing data automatically
result = fit(X, model="rasch")

# Score all persons (missing responses handled appropriately)
scores = result.score(method="eap")
print("Ability estimates:", scores.theta)
```

### Example 4: Custom Technical Settings

```python
from irt import fit
import numpy as np

X = np.random.binomial(1, 0.5, size=(1000, 30)).astype(float)

# Use more quadrature points for higher accuracy
result = fit(
    X,
    model="2pl",
    technical={
        "quadpts": 121,           # More points (default: 61)
        "max_iter": 500,          # More iterations allowed
        "tol": 1e-6,              # Stricter convergence
        "a_bounds": (0.5, 3.0),   # Narrower discrimination bounds
    }
)

print(f"Converged in {result.n_iter} iterations")
```

### Example 5: Scoring New Data

```python
from irt import fit
import numpy as np

# Training data
X_train = np.random.binomial(1, 0.6, size=(500, 20)).astype(float)

# Fit model on training data
result = fit(X_train, model="rasch")

# New response data (same items, different persons)
X_new = np.random.binomial(1, 0.6, size=(50, 20)).astype(float)

# Score new persons using fitted model
new_scores = result.score(X=X_new, method="eap")
print("New person abilities:", new_scores.theta[:10])
```

### Example 6: Comparing Scoring Methods

```python
from irt import fit
import numpy as np
import matplotlib.pyplot as plt

X = np.random.binomial(1, 0.5, size=(200, 15)).astype(float)
result = fit(X, model="rasch")

# Get scores using different methods
eap = result.score(method="eap")
map_scores = result.score(method="map")

# Compare
plt.figure(figsize=(8, 6))
plt.scatter(eap.theta, map_scores.theta, alpha=0.5)
plt.xlabel("EAP Estimates")
plt.ylabel("MAP Estimates")
plt.plot([-3, 3], [-3, 3], 'r--', label="y=x")
plt.legend()
plt.title("EAP vs MAP Ability Estimates")
plt.show()

# Correlation
corr = np.corrcoef(eap.theta, map_scores.theta)[0, 1]
print(f"Correlation: {corr:.4f}")
```

### Example 7: JMLE for Rasch Model

```python
from irt import fit
import numpy as np

X = np.random.binomial(1, 0.6, size=(300, 15)).astype(float)

# Use JMLE estimator (Rasch only)
result = fit(X, model="rasch", estimator="jmle")

print(f"Converged: {result.converged}")
print(f"Item difficulties: {result.params['b']}")
print(f"Mean difficulty: {result.params['b'].mean():.4f}")  # Should be ~0 (centered)
```

### Example 8: Polytomous Models (PCM, GPCM, GRM)

```python
from irt import fit
import numpy as np

# Likert-scale data: 200 persons, 10 items, 5 categories (0-4)
np.random.seed(42)
X = np.random.randint(0, 5, size=(200, 10)).astype(float)
X[np.random.random(X.shape) < 0.05] = np.nan  # Add missing

# Fit Partial Credit Model
result = fit(X, model="pcm")
print(f"Converged: {result.converged}")
print(result.item_report())

# Category Characteristic Curves
fig, ax = result.plot_icc(items=[0])
ax.set_title("Item 0: Category Characteristic Curves")

# Other polytomous models: "gpcm", "grm", "rsm", "nrm"
```

See `notebooks/07_polytomous_models.ipynb` for a full polytomous demo.

---

## Technical Details

### Model Parameterization

This library uses the **traditional IRT parameterization**:

$$P(X = 1 | \theta, a, b) = \frac{1}{1 + e^{-a(\theta - b)}}$$

**Note**: Some software (e.g., R's mirt package) uses slope-intercept form internally:
- mirt: P = 1 / (1 + exp(-(a*θ + d))) where d = -a*b
- To convert: b = -d/a

### Identification Constraints

**MML-EM**: 
- Person abilities assumed to follow N(0, 1) prior
- No constraint on item parameters needed

**JMLE**:
- Item difficulties centered: mean(b) = 0
- Necessary to avoid indeterminacy

### Numerical Stability

The implementation includes several safeguards:
- Log-sum-exp trick for posterior calculations
- Probability clipping to avoid log(0)
- Bounded optimization for item parameters
- Damped Newton-Raphson updates

### Default Settings Comparison

| Setting | MML-EM | JMLE |
|---------|--------|------|
| Max iterations | 200 | 50 |
| Tolerance | 1e-4 | 1e-4 |
| Quadrature points | 61 | N/A |
| Theta range | [-4, 4] | [-6, 6] |
| b bounds | [-6, 6] | [-6, 6] |
| a bounds | [0.25, 4] | N/A |

---

## Troubleshooting

### Common Issues

#### "No persons remaining after filtering"

```
ValueError: No persons remaining after filtering. All persons had fewer than 2 observed responses.
```

**Cause**: Too many missing values per person.  
**Solution**: Check your data for rows with mostly NaN values.

#### Model doesn't converge

**Symptoms**: `result.converged = False`, high `n_iter`  
**Solutions**:
1. Increase `max_iter` in technical settings
2. Use more lenient `tol`
3. Check for items with extreme difficulty (all 0s or all 1s)
4. Try different starting values

#### Extreme parameter estimates

**Symptoms**: Very large |b| or |a| values  
**Causes**: 
- Items with very low/high proportion correct
- Small sample size

**Solutions**:
1. Remove problematic items
2. Tighten parameter bounds in technical settings
3. Increase sample size

#### JMLE with 2PL

```
ValueError: JMLE estimator only supports Rasch model.
```

**Solution**: Use `estimator="mml_em"` for 2PL models.

#### Polytomous model with JMLE

```
ValueError: Polytomous model 'pcm' only supports estimator='mml_em'.
```

**Solution**: Polytomous models (PCM, RSM, GRM, GPCM, NRM) require MML-EM. Omit `estimator` or use `estimator="mml_em"`.

### Validation Against R mirt

To verify your results, compare with R's mirt package:

```r
# R code
library(mirt)
mod <- mirt(data, 1, itemtype='Rasch')
coef(mod, simplify=TRUE, IRTpars=TRUE)
fscores(mod, method='EAP')
```

Expected agreement:
- Item parameters: within ±0.05 logits
- EAP scores: correlation > 0.99

---

## Version History

- **0.1.0**: Initial release with Rasch and 2PL MML-EM, JMLE for Rasch
- **0.2.0** (planned): Polytomous models (PCM, RSM, GRM, GPCM, NRM) with MML-EM

---

## License

MIT License

---

## References

- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46(4), 443-459.
- Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems*. Erlbaum.
- Rasch, G. (1960). *Probabilistic Models for Some Intelligence and Attainment Tests*. Danish Institute for Educational Research.
- Masters, G. N. (1982). A Rasch model for partial credit scoring. *Psychometrika*, 47(2), 149-174.
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph Supplement*, 17, 1-100.
