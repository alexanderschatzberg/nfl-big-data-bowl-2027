"""Does Combine tracking predict NFL outcomes beyond what teams already have?

All scores are out-of-fold. R^2 is measured against the training-fold mean, so
a model with no signal scores zero or below rather than benefiting from the
in-sample mean.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .hypotheses import Design

ALPHAS = np.logspace(-1, 4, 20)
MAX_MISSING = 0.5
N_SPLITS = 5
SPARSE_K = 3


def _pipe():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         StandardScaler(), RidgeCV(alphas=ALPHAS))


class Dense(BaseEstimator, RegressorMixin):
    """Ridge on every column."""

    def fit(self, X, y, sample_weight=None):
        self.model_ = _pipe().fit(X, y, ridgecv__sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class SparseIncrement(BaseEstimator, RegressorMixin):
    """Baseline ridge, then the k tracking traits that best explain what it missed.

    Trait selection happens inside `fit`, so under cross-validation it never
    sees the held-out players. With ~70 players this has far more power than a
    ridge over every trait.
    """

    def __init__(self, n_base: int, k: int = SPARSE_K):
        self.n_base = n_base
        self.k = k

    def fit(self, X, y, sample_weight=None):
        base, track = X[:, : self.n_base], X[:, self.n_base :]
        self.base_ = _pipe().fit(base, y, ridgecv__sample_weight=sample_weight)
        self.track_ = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
            SelectKBest(f_regression, k=min(self.k, track.shape[1])), RidgeCV(alphas=ALPHAS),
        ).fit(track, y - self.base_.predict(base), ridgecv__sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.base_.predict(X[:, : self.n_base]) + self.track_.predict(X[:, self.n_base :])

    def selected(self, names: list[str]) -> list[str]:
        return [n for n, keep in zip(names, self.track_[-2].get_support()) if keep]


def oof_predict(X, y, w, seed: int, model=Dense) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions and the matching training-mean predictions."""
    pred, null = np.empty(len(y)), np.empty(len(y))
    for tr, te in KFold(N_SPLITS, shuffle=True, random_state=seed).split(y):
        null[te] = np.average(y[tr], weights=w[tr])
        pred[te] = model().fit(X[tr], y[tr], sample_weight=w[tr]).predict(X[te])
    return pred, null


def cv_r2(X, y, w, repeats: int, seed: int = 0, model=Dense) -> float:
    scores = []
    for r in range(repeats):
        pred, null = oof_predict(X, y, w, seed + r, model)
        scores.append(1 - np.sum(w * (y - pred) ** 2) / np.sum(w * (y - null) ** 2))
    return float(np.mean(scores))


def cv_spearman(X, y, w, repeats: int, seed: int = 0, model=Dense) -> float:
    return float(np.mean([stats.spearmanr(oof_predict(X, y, w, seed + r, model)[0], y)[0] for r in range(repeats)]))


def prepare(design: Design, outcome: str) -> dict:
    df = design.df[design.df[outcome].notna()].reset_index(drop=True)
    tracking = [c for c in design.tracking if df[c].isna().mean() <= MAX_MISSING]
    return {
        "df": df,
        "y": df[outcome].to_numpy(float),
        "w": df[f"{outcome}__rel"].clip(lower=0.05).to_numpy(float),
        "blocks": {
            "stopwatch": df[design.stopwatch].to_numpy(float),
            "stopwatch+draft": df[design.stopwatch + design.draft].to_numpy(float),
            "tracking": df[tracking].to_numpy(float),
        },
        "tracking_cols": tracking,
    }


def incremental_r2(base: np.ndarray, track: np.ndarray, y, w, repeats: int, seed: int = 0, k: int = SPARSE_K) -> float:
    """Out-of-fold R^2 gained by adding the best k tracking traits to the baseline."""
    sparse = lambda: SparseIncrement(base.shape[1], k)  # noqa: E731
    return cv_r2(np.hstack([base, track]), y, w, repeats, seed, sparse) - cv_r2(base, y, w, repeats, seed)


def permuted_increments(base, track, y, w, seeds: list[int], repeats: int) -> list[float]:
    """Null distribution: shuffle which player owns which tracking profile."""
    out = []
    for s in seeds:
        perm = np.random.default_rng(s).permutation(len(y))
        out.append(incremental_r2(base, track[perm], y, w, repeats))
    return out


def _residualize(v: np.ndarray, Z: np.ndarray) -> np.ndarray:
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    return v - Z1 @ np.linalg.lstsq(Z1, v, rcond=None)[0]


def feature_table(p: dict, base_name: str = "stopwatch+draft") -> pd.DataFrame:
    """Partial Spearman of each tracking feature with the outcome, net of the baseline."""
    Z = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(p["blocks"][base_name])
    Z = Z[:, Z.std(axis=0) > 0]
    rows = []
    for j, col in enumerate(p["tracking_cols"]):
        x = p["blocks"]["tracking"][:, j]
        ok = ~np.isnan(x)
        if ok.sum() < 20:
            continue
        rho, pval = stats.spearmanr(_residualize(x[ok], Z[ok]), _residualize(p["y"][ok], Z[ok]))
        rows.append({"feature": col, "n": int(ok.sum()), "partial_rho": rho, "p": pval})
    t = pd.DataFrame(rows)
    if len(t):
        t["q"] = stats.false_discovery_control(t["p"], method="bh")
        t = t.sort_values("p")
    return t


def leave_class_out_r2(p: dict, cols: np.ndarray, model=Dense) -> float:
    """Train on two draft classes, predict the third: the real scouting use case."""
    y, w, years = p["y"], p["w"], p["df"]["draft_year"].to_numpy()
    sse = sst = 0.0
    for yr in np.unique(years):
        tr, te = years != yr, years == yr
        if te.sum() < 5 or tr.sum() < 20:
            continue
        pred = model().fit(cols[tr], y[tr], sample_weight=w[tr]).predict(cols[te])
        sse += np.sum(w[te] * (y[te] - pred) ** 2)
        sst += np.sum(w[te] * (y[te] - np.average(y[tr], weights=w[tr])) ** 2)
    return float(1 - sse / sst) if sst > 0 else float("nan")


def fit(design: Design, outcome: str, repeats: int = 20) -> dict:
    p = prepare(design, outcome)
    y, w, b = p["y"], p["w"], p["blocks"]
    base = b["stopwatch+draft"]
    full = np.hstack([base, b["tracking"]])
    sparse = lambda: SparseIncrement(base.shape[1])  # noqa: E731
    res = {
        "outcome": outcome,
        "n_players": len(y),
        "mean_reliability": float(p["df"][f"{outcome}__rel"].mean()),
        "median_plays": float(p["df"][f"{outcome}__n"].median()),
        "n_tracking_features": len(p["tracking_cols"]),
        "r2": {k: cv_r2(v, y, w, repeats) for k, v in b.items()}
              | {"dense": cv_r2(full, y, w, repeats), "sparse": cv_r2(full, y, w, repeats, model=sparse)},
        "spearman": {"stopwatch+draft": cv_spearman(base, y, w, repeats),
                     "sparse": cv_spearman(full, y, w, repeats, model=sparse)},
        "leave_class_out_r2": {"stopwatch+draft": leave_class_out_r2(p, base),
                               "sparse": leave_class_out_r2(p, full, sparse)},
        "selected": sparse().fit(full, y, sample_weight=w).selected(p["tracking_cols"]),
        "features": feature_table(p).to_dict("records"),
    }
    res["delta_r2"] = {m: res["r2"][m] - res["r2"]["stopwatch+draft"] for m in ("dense", "sparse")}
    return res


def add_permutation(res: dict, observed: float, null: list[float]) -> dict:
    null = np.asarray(null)
    res["permutation"] = {
        "n": len(null),
        "observed_delta_r2": observed,
        "null_mean": float(null.mean()) if len(null) else None,
        "null_p95": float(np.percentile(null, 95)) if len(null) else None,
        "p_value": float((1 + np.sum(null >= observed)) / (1 + len(null))) if len(null) else None,
    }
    return res
