"""
=============================================================
  Duolingo "Learning Age" Pipeline  (v3)
=============================================================
Architecture: Option 2 + Option 3 combined
-------------------------------------------

STEP A — Per-user memory model (real ground truth)
  For each user we fit:
      log(p_recall) = b0 + b1*log(delta+1) + b2*log(history_seen+1)
  b1 = forgetting rate  (the core cognitive signal; more negative = forgets faster)
  b2 = learning speed
  b0 = baseline memory level

STEP B — Predictive regression model for b1  [OPTION 2]
  b1 has real ground truth (fitted from actual recall data).
  We train a GradientBoosting regression model to predict b1 from
  behavioural features (vocab breadth, session frequency, accuracy, etc.).
  This gives us genuine, honest metrics: R², MAE, feature importances.
  It also answers: "which behaviours drive forgetting rate?"

STEP C — Learning Age index  [OPTION 3]
  The learning age is presented as a transparent scoring rubric, not a
  learned model. It is a weighted composite of meaningful components,
  mapped to a [20, 80] year scale anchored to cognitive aging research:
    age = 20 + 60 * (1 - score)^2
  20 = peak ease of learning (best score)
  80 = lowest ease of learning (worst score)
  The age is an INDEX (like a credit score), not a prediction.
  We are explicit about this: the weights are chosen by design, not learned.

STEP D — Metrics
  A) b1 regression: R², MAE  (real ground truth)
  B) GBM model predicting b1: 5-fold CV R², MAE, feature importances
  C) Age index properties: distribution, bootstrap CI width, correlation
     of age with b1 (sanity check: more negative b1 should mean older age)

Usage
-----
  Edit DATA_PATH below, then:
      python learning_age_pipeline.py
  Optional CLI overrides:
      python learning_age_pipeline.py --sample 500000 --output results/
      python learning_age_pipeline.py --user_id u:abc123
=============================================================
"""


# ──────────────────────────────────────────────────────────
#  SET YOUR DATA PATH HERE
# ──────────────────────────────────────────────────────────
DATA_PATH   = r"C:\Users\thoma\OneDrive\Documenten\KUL\Master AI\Datathon\Spaced Repetition Data\Spaced Repetition Data"
OUTPUT_DIR  = "learning_age_output"
SAMPLE_SIZE = None   # set None to use full dataset
# ──────────────────────────────────────────────────────────

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RANDOM_STATE        = 42
MIN_EVENTS_PER_USER = 2000

# ── Age index anchors ──────────────────────────────────────
# Anchored to the right side of the cognitive aging bell curve.
# score=1.0 (best learner) -> age 20  (peak ease of learning)
# score=0.0 (worst learner) -> age 80  (lowest ease of learning)
AGE_PEAK     = 18    # peak of the critical period curve (best learner -> age 18)
AGE_MAX      = 80    # asymptote (weakest learner -> age 80)
DECAY_ALPHA  = 0.5   # power term: controls steepness of initial drop from peak
DECAY_BETA   = 3.0   # exponential term: creates fast drop near score=1, long tail near 0
# Together: age = 18 + 62 * (1-score)^0.5 * exp(-3*score)
# This replicates the RIGHT SIDE of the critical period bell curve:
#   score=1.0 -> age 18  (peak, steepest learning ability)
#   score=0.9 -> age ~19 (steep drop)
#   score=0.5 -> age ~28 (rapid decline)
#   score=0.1 -> age ~62 (long gradual tail)
#   score=0.0 -> age 80  (asymptote)

# ── Fixed reference ranges for beta scaling ───────────────
# Grounded in HLR literature and the memory decay model.
# These are FIXED (not data-driven) so the age index is stable
# across different sample sizes and datasets.
#
#   b1 (forgetting rate)  : theoretically negative; [-2.0, 0.0]
#     b1 near  0.0 = almost no forgetting over time
#     b1 near -2.0 = very fast decay
#
#   b2 (learning speed)   : theoretically positive; [0.0, 2.0]
#     b2 near 0.0 = repetitions barely help
#     b2 near 2.0 = very fast learning from repetition
#
#   b0 (baseline memory)  : log-scale intercept; [-3.0, 1.0]
#     b0 near -3.0 = very poor baseline recall
#     b0 near  1.0 = strong baseline recall
BETA1_RANGE = (-2.0,  0.0)   # (worst, best) for retention
BETA2_RANGE = ( 0.0,  2.0)   # (worst, best) for learning speed
BETA0_RANGE = (-3.0,  1.0)   # (worst, best) for baseline memory

# ── Composite score weights (must sum to 1.0) ──────────────
# These are design choices, not learned — documented transparently.
# All three beta components are included since they each carry
# distinct cognitive information about the user's memory profile.
# Behavioural features (vocab, accuracy, effort, frequency) are
# kept as supporting signals.
SCORE_WEIGHTS = {
    "retention":    0.25,   # b1 scaled: less forgetting = higher score
    "learning_spd": 0.20,   # b2 scaled: faster learning = higher score
    "baseline":     0.15,   # b0 scaled: better baseline = higher score
    "vocab":        0.15,   # log vocab breadth
    "accuracy":     0.10,   # overall recall accuracy
    "effort":       0.10,   # log total events
    "frequency":    0.05,   # inverse avg gap
}

# ── Age bracket definitions ────────────────────────────────
AGE_BRACKETS = [
    (18, 25, "Peak Learner (18-25)"),
    (26, 35, "Early Decline (26-35)"),
    (36, 50, "Mid Decline (36-50)"),
    (51, 65, "Late Decline (51-65)"),
    (66, 80, "Advanced Decline (66-80)"),
]

# ── Features used to predict b1 ───────────────────────────
# Behavioural features ONLY — no betas as predictors, to keep
# the GBM model honest (we predict b1, not from itself).
PREDICTOR_COLS = [
    "overall_accuracy",
    "log_vocab_breadth",
    "log_avg_history_seen",
    "avg_delta_days",
    "std_delta_days",
    "session_acc_rate",
    "log_total_events",
]


# ═══════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_data(path: str, sample: int | None = None) -> pd.DataFrame:
    print(f"[1/5] Loading data from {path} ...")
    df = pd.read_csv(path, sep=",", low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {"p_recall", "timestamp", "delta", "user_id",
                "history_seen", "history_correct",
                "session_seen", "session_correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE)

    df["p_recall"]        = df["p_recall"].clip(1e-6, 1 - 1e-6)
    df["delta"]           = df["delta"].clip(0)
    df["history_seen"]    = df["history_seen"].clip(0)
    df["history_correct"] = df["history_correct"].clip(0)

    print(f"    Loaded {len(df):,} rows, {df['user_id'].nunique():,} unique users.")
    return df


# ═══════════════════════════════════════════════════════════
# 2.  PER-USER β ESTIMATION  (Step A — real ground truth)
# ═══════════════════════════════════════════════════════════

def estimate_user_betas(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Fit per-user Ridge regression:
        log(p_recall) = b0 + b1*log(delta+1) + b2*log(history_seen+1)

    b1 is the forgetting rate — the primary target for the predictive model.
    More negative b1 = faster forgetting = cognitively "older" learner.

    Returns betas DataFrame and quality metrics dict.
    """
    print("[2/5] Fitting per-user memory models (beta regression) ...")

    df = df.copy()
    df["log_p"]  = np.log(df["p_recall"])
    df["log_d"]  = np.log(df["delta"] + 1)
    df["log_hs"] = np.log(df["history_seen"] + 1)

    records  = []
    r2_list  = []
    mae_list = []

    for uid, g in df.groupby("user_id"):
        if len(g) < MIN_EVENTS_PER_USER:
            continue
        X = g[["log_d", "log_hs"]].values
        y = g["log_p"].values
        reg = Ridge(alpha=0.01, fit_intercept=True)
        try:
            reg.fit(X, y)
            y_pred_log = reg.predict(X)
            p_pred = np.exp(y_pred_log).clip(1e-6, 1)
            p_true = g["p_recall"].values

            r2  = r2_score(y, y_pred_log)
            mae = mean_absolute_error(p_true, p_pred)
            r2_list.append(r2)
            mae_list.append(mae)

            records.append({
                "user_id":  uid,
                "beta0":    reg.intercept_,
                "beta1":    reg.coef_[0],   # forgetting rate — primary target
                "beta2":    reg.coef_[1],   # learning speed
                "n_events": len(g),
                "beta_r2":  r2,
                "beta_mae": mae,
            })
        except Exception:
            continue

    betas = pd.DataFrame(records).set_index("user_id")

    beta_metrics = {
        "n_users_fitted":     len(betas),
        "median_r2":          float(np.median(r2_list)),
        "mean_r2":            float(np.mean(r2_list)),
        "pct_positive_r2":    float(np.mean(np.array(r2_list) > 0) * 100),
        "median_mae_precall": float(np.median(mae_list)),
        "mean_mae_precall":   float(np.mean(mae_list)),
    }

    print(f"    Users fitted       : {beta_metrics['n_users_fitted']:,}")
    print(f"    Median R² (log p)  : {beta_metrics['median_r2']:.4f}")
    print(f"    % users R² > 0     : {beta_metrics['pct_positive_r2']:.1f}%")
    print(f"    Median MAE (p)     : {beta_metrics['median_mae_precall']:.4f}")
    print(f"    b1 range           : {betas['beta1'].min():.3f} to {betas['beta1'].max():.3f}")
    return betas, beta_metrics


# ═══════════════════════════════════════════════════════════
# 3.  BEHAVIOURAL FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════

def build_user_features(df: pd.DataFrame, betas: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] Building behavioural feature matrix ...")

    agg_kwargs = dict(
        overall_accuracy = ("p_recall",      "mean"),
        vocab_breadth    = ("lexeme_id",      "nunique"),
        total_events     = ("p_recall",       "count"),
        avg_history_seen = ("history_seen",   "mean"),
        avg_delta_days   = ("delta",           lambda x: x.mean() / 86400),
        std_delta_days   = ("delta",           lambda x: x.std()  / 86400),
        session_accuracy = ("session_correct", "sum"),
        session_total    = ("session_seen",    "sum"),
    )
    if "learning_language" in df.columns:
        agg_kwargs["n_languages"] = ("learning_language", "nunique")

    feats = df.groupby("user_id").agg(**agg_kwargs)
    feats["session_acc_rate"] = feats["session_accuracy"] / feats["session_total"].clip(1)
    feats.drop(columns=["session_accuracy", "session_total"], inplace=True)

    for col in ["vocab_breadth", "total_events", "avg_history_seen"]:
        feats[f"log_{col}"] = np.log1p(feats[col])

    combined = betas.join(feats, how="inner")
    combined.fillna(combined.median(), inplace=True)
    print(f"    Feature matrix: {combined.shape[0]:,} users x {combined.shape[1]} features.")
    return combined


# ═══════════════════════════════════════════════════════════
# 4.  PREDICTIVE MODEL: predict b1 from behaviour  (Step B)
# ═══════════════════════════════════════════════════════════

def fit_forgetting_model(user_feats: pd.DataFrame) -> tuple[GradientBoostingRegressor, StandardScaler, dict]:
    """
    Train a GradientBoosting regressor to predict b1 (forgetting rate)
    from behavioural features only (no betas used as predictors).

    This is the core honest model:
      - target  = b1  (fitted from real p_recall data — real ground truth)
      - features = behavioural aggregates (vocab, accuracy, gap, etc.)
      - metrics  = 5-fold CV R², MAE, feature importances

    Answers: which user behaviours predict how fast someone forgets?
    """
    print("[4/5] Fitting forgetting rate prediction model (GBM -> b1) ...")

    X = user_feats[PREDICTOR_COLS].values
    y = user_feats["beta1"].values   # real ground truth target

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    gbm = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_r2  = cross_val_score(gbm, X_sc, y, cv=kf, scoring="r2")
    cv_mae = cross_val_score(gbm, X_sc, y, cv=kf, scoring="neg_mean_absolute_error")

    # Fit on full data for feature importances + predictions
    gbm.fit(X_sc, y)
    y_pred      = gbm.predict(X_sc)
    train_r2    = r2_score(y, y_pred)
    train_mae   = mean_absolute_error(y, y_pred)

    imp_df = pd.DataFrame({
        "feature":    PREDICTOR_COLS,
        "importance": gbm.feature_importances_,
    }).sort_values("importance", ascending=False)

    model_metrics = {
        "cv_r2_mean":   float(cv_r2.mean()),
        "cv_r2_std":    float(cv_r2.std()),
        "cv_mae_mean":  float((-cv_mae).mean()),
        "cv_mae_std":   float((-cv_mae).std()),
        "train_r2":     float(train_r2),
        "train_mae":    float(train_mae),
        "importances":  imp_df,
        "y_pred":       y_pred,
    }

    print(f"    5-fold CV R²  : {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
    print(f"    5-fold CV MAE : {(-cv_mae).mean():.4f} +/- {(-cv_mae).std():.4f}")
    print(f"    Train R²      : {train_r2:.4f}  (in-sample, for reference)")
    print()
    print("    Feature importances (predicting forgetting rate b1):")
    for _, row in imp_df.iterrows():
        bar = "█" * int(row["importance"] * 40)
        print(f"      {row['feature']:<25} {row['importance']:.4f}  {bar}")

    return gbm, scaler, model_metrics


# ═══════════════════════════════════════════════════════════
# 5.  LEARNING AGE INDEX  (Step C — transparent scoring rubric)
# ═══════════════════════════════════════════════════════════

def score_to_age(score: np.ndarray) -> np.ndarray:
    """
    Map composite score [0,1] -> learning age [AGE_PEAK, AGE_MAX].

    Formula: age = AGE_PEAK + (AGE_MAX - AGE_PEAK) * (1-score)^ALPHA * exp(-BETA*score)

    This replicates the RIGHT SIDE of the critical period bell curve:
      - The (1-score)^ALPHA term: power decay — age approaches AGE_PEAK as score->1
      - The exp(-BETA*score) term: exponential suppression near score=1, ensuring
        the steepest drop happens just below the peak, with a long gradual tail
        toward score=0 (matching the slow right tail of the bell curve)

    Key values:
      score=1.0 -> age 18  (peak ease of learning)
      score=0.9 -> age ~19 (steep initial drop)
      score=0.5 -> age ~28 (rapid middle decline)
      score=0.1 -> age ~62 (slow tail)
      score=0.0 -> age 80  (asymptote — lowest ease of learning)
    """
    span = AGE_MAX - AGE_PEAK
    return AGE_PEAK + span * (1.0 - score) ** DECAY_ALPHA * np.exp(-DECAY_BETA * score)


def compute_learning_age(user_feats: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Learning Age Index for each user.

    This is a SCORING RUBRIC, not a learned model. The weights are
    chosen by design based on cognitive aging literature and documented
    transparently. The age is presented as an index analogous to a
    credit score — constructed from meaningful components, not predicted.

    All three beta values are included as they each carry distinct
    cognitive information:
      b1 (forgetting rate)  — how fast memory decays over time
      b2 (learning speed)   — how much each repetition helps
      b0 (baseline memory)  — starting recall strength

    Beta components use FIXED reference ranges from the HLR literature
    so the age index is stable across different sample sizes and datasets.
    Behavioural features (vocab, accuracy, effort, frequency) use
    data-driven min-max scaling as they have no fixed theoretical bounds.
    """
    print("[5/5] Computing Learning Age index + sanity checks ...")

    uf = user_feats.copy()

    def minmax_data(s):
        """Data-driven min-max for behavioural features (no fixed anchors)."""
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9)

    def fixed_scale(s, worst, best):
        """Scale using fixed reference range from cognitive literature.
        Values outside the range are clipped to [0, 1].
        worst -> 0.0 (weakest learner signal)
        best  -> 1.0 (strongest learner signal)
        """
        return ((s - worst) / (best - worst)).clip(0, 1)

    components = pd.DataFrame(index=uf.index)

    # ── Beta components: fixed scaling (stable across datasets) ──
    # b1: more negative = faster forgetting = lower score
    #   BETA1_RANGE worst=-2.0 (very fast forgetting) -> 0
    #   BETA1_RANGE best=0.0   (no forgetting)        -> 1
    components["retention"]    = fixed_scale(uf["beta1"],
                                             worst=BETA1_RANGE[0],
                                             best=BETA1_RANGE[1])

    # b2: higher = faster learning from repetition = higher score
    #   BETA2_RANGE worst=0.0 (repetitions don't help) -> 0
    #   BETA2_RANGE best=2.0  (very fast learning)     -> 1
    components["learning_spd"] = fixed_scale(uf["beta2"],
                                             worst=BETA2_RANGE[0],
                                             best=BETA2_RANGE[1])

    # b0: higher = better baseline recall = higher score
    #   BETA0_RANGE worst=-3.0 (very poor baseline) -> 0
    #   BETA0_RANGE best=1.0   (strong baseline)    -> 1
    components["baseline"]     = fixed_scale(uf["beta0"],
                                             worst=BETA0_RANGE[0],
                                             best=BETA0_RANGE[1])

    # ── Behavioural components: data-driven scaling ───────────────
    components["vocab"]        = minmax_data( uf["log_vocab_breadth"])
    components["accuracy"]     = minmax_data( uf["overall_accuracy"])
    components["effort"]       = minmax_data( uf["log_total_events"])
    components["frequency"]    = minmax_data(-uf["avg_delta_days"])

    w = np.array([SCORE_WEIGHTS[k] for k in components.columns])
    raw_score = components.values @ w
    point_age = score_to_age(raw_score)

    # Bootstrap CI (200 resamples with +-5% noise)
    rng = np.random.default_rng(RANDOM_STATE)
    bootstrap_ages = np.zeros((len(uf), 200))
    for i in range(200):
        noise = rng.normal(0, 0.05, size=components.shape)
        comp_noisy = (components.values + noise).clip(0, 1)
        bootstrap_ages[:, i] = score_to_age(comp_noisy @ w)

    age_lo = np.percentile(bootstrap_ages,  5, axis=1)
    age_hi = np.percentile(bootstrap_ages, 95, axis=1)

    uf["composite_score"]  = np.round(raw_score, 4)
    uf["learning_age"]     = np.round(point_age).astype(int)
    uf["learning_age_lo"]  = np.round(age_lo).astype(int)
    uf["learning_age_hi"]  = np.round(age_hi).astype(int)
    uf["age_ci_width"]     = uf["learning_age_hi"] - uf["learning_age_lo"]

    def assign_bracket(age):
        for lo, hi, label in AGE_BRACKETS:
            if lo <= age <= hi:
                return label
        return AGE_BRACKETS[-1][2]

    uf["age_bracket"] = uf["learning_age"].apply(assign_bracket)

    # ── Sanity check: Spearman correlation between b1 and learning age ──
    # If the index is sensible, more negative b1 should correlate with
    # older learning age. This is our key validation check.
    spearman_b1, p_b1 = stats.spearmanr(uf["beta1"], uf["learning_age"])
    spearman_b2, p_b2 = stats.spearmanr(uf["beta2"], uf["learning_age"])
    spearman_b0, p_b0 = stats.spearmanr(uf["beta0"], uf["learning_age"])

    print(f"    Age range     : {uf['learning_age'].min()} - {uf['learning_age'].max()} yrs")
    print(f"    Mean / Median : {uf['learning_age'].mean():.0f} / {uf['learning_age'].median():.0f} yrs")
    print(f"    Mean CI width : +/-{(uf['age_ci_width']/2).mean():.0f} years (90%)")
    print()
    print("    Sanity checks — Spearman correlations with learning_age:")
    print(f"      b1 (forgetting) : r={spearman_b1:+.4f}  p={p_b1:.2e}  "
          f"{'OK (positive = more forgetting -> older)' if spearman_b1 > 0.2 else 'WEAK'}")
    print(f"      b2 (learning)   : r={spearman_b2:+.4f}  p={p_b2:.2e}  "
          f"{'OK (negative = less learning speed -> older)' if spearman_b2 < -0.2 else 'WEAK'}")
    print(f"      b0 (baseline)   : r={spearman_b0:+.4f}  p={p_b0:.2e}  "
          f"{'OK (negative = worse baseline -> older)' if spearman_b0 < -0.2 else 'WEAK'}")
    print()
    print("    Age bracket distribution:")
    for bracket, cnt in uf["age_bracket"].value_counts().items():
        print(f"      {bracket}: {cnt:,}")

    return uf, {
        "spearman_b1": spearman_b1, "p_b1": p_b1,
        "spearman_b2": spearman_b2, "p_b2": p_b2,
        "spearman_b0": spearman_b0, "p_b0": p_b0,
    }


# ═══════════════════════════════════════════════════════════
# METRICS REPORT
# ═══════════════════════════════════════════════════════════

def compute_metrics_report(
    user_feats: pd.DataFrame,
    beta_metrics: dict,
    model_metrics: dict,
    age_metrics: dict,
    output_dir: Path,
):
    print("\n[Metrics] Writing full metrics report ...")
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    def log(s=""):
        print("   ", s)
        lines.append(s)

    # ── A: Memory model (b1 regression) ──────────────────────────
    log("=" * 62)
    log("A) PER-USER MEMORY MODEL QUALITY  (real ground truth)")
    log("=" * 62)
    log("  Model: log(p_recall) = b0 + b1*log(delta+1) + b2*log(history_seen+1)")
    log("  Target: p_recall  [actual recall probability — real ground truth]")
    log()
    log(f"  Users fitted               : {beta_metrics['n_users_fitted']:,}")
    log(f"  Median R2  (log p_recall)  : {beta_metrics['median_r2']:.4f}")
    log(f"  Mean   R2  (log p_recall)  : {beta_metrics['mean_r2']:.4f}")
    log(f"  % users with R2 > 0        : {beta_metrics['pct_positive_r2']:.1f}%")
    log(f"  Median MAE on p_recall     : {beta_metrics['median_mae_precall']:.4f}")
    log(f"  Mean   MAE on p_recall     : {beta_metrics['mean_mae_precall']:.4f}")
    log()
    log("  Interpretation:")
    log("  R2 measures how well delta and history_seen explain each user's")
    log("  recall probability. Median R2 = average model fit per user.")
    log("  MAE is in probability space [0,1]: lower = better recall prediction.")
    log()

    # ── B: Forgetting rate prediction ────────────────────────────
    log("=" * 62)
    log("B) FORGETTING RATE PREDICTION MODEL  (real ground truth)")
    log("=" * 62)
    log("  Model: GradientBoosting -> b1 (forgetting rate)")
    log("  Target: b1  [fitted from p_recall — real ground truth]")
    log("  Features: behavioural aggregates only (no betas as predictors)")
    log()
    log(f"  5-fold CV R2              : {model_metrics['cv_r2_mean']:.4f} +/- {model_metrics['cv_r2_std']:.4f}")
    log(f"  5-fold CV MAE             : {model_metrics['cv_mae_mean']:.4f} +/- {model_metrics['cv_mae_std']:.4f}")
    log(f"  Train R2 (in-sample)      : {model_metrics['train_r2']:.4f}")
    log(f"  Train MAE (in-sample)     : {model_metrics['train_mae']:.4f}")
    log()
    log("  Feature importances (GBM predicting forgetting rate b1):")
    for _, row in model_metrics["importances"].iterrows():
        bar = "X" * int(row["importance"] * 40)
        log(f"    {row['feature']:<25} {row['importance']:.4f}  {bar}")
    log()
    log("  Interpretation:")
    log("  CV R2 shows how well behavioural features generalize to predict")
    log("  forgetting rate. Feature importances reveal which habits matter most.")
    log()

    # ── C: Age index validation ───────────────────────────────────
    log("=" * 62)
    log("C) LEARNING AGE INDEX VALIDATION")
    log("=" * 62)
    log("  The age is a SCORING INDEX (design choice, not learned model).")
    log("  Weights are documented transparently. Validated via sanity checks.")
    log()
    log(f"  Age range               : {user_feats['learning_age'].min()} - {user_feats['learning_age'].max()} years")
    log(f"  Mean +/- std            : {user_feats['learning_age'].mean():.0f} +/- {user_feats['learning_age'].std():.0f} years")
    log(f"  IQR (25-75%)            : {user_feats['learning_age'].quantile(0.25):.0f} - {user_feats['learning_age'].quantile(0.75):.0f} years")
    log(f"  Mean 90% CI width       : +/-{(user_feats['age_ci_width']/2).mean():.0f} years")
    log()
    log("  Component weights (by design):")
    for k, v in SCORE_WEIGHTS.items():
        log(f"    {k:<20} {v:.2f}")
    log()
    log("  Beta reference ranges (fixed, from HLR literature):")
    log(f"    b1: {BETA1_RANGE[0]} (worst) to {BETA1_RANGE[1]} (best)")
    log(f"    b2: {BETA2_RANGE[0]} (worst) to {BETA2_RANGE[1]} (best)")
    log(f"    b0: {BETA0_RANGE[0]} (worst) to {BETA0_RANGE[1]} (best)")
    log()
    log("  Sanity checks — Spearman correlations with learning_age:")
    log(f"    b1 (forgetting) : r={age_metrics['spearman_b1']:+.4f}  p={age_metrics['p_b1']:.2e}"
        f"  (expected: positive — more forgetting -> older)")
    log(f"    b2 (learning)   : r={age_metrics['spearman_b2']:+.4f}  p={age_metrics['p_b2']:.2e}"
        f"  (expected: negative — less speed -> older)")
    log(f"    b0 (baseline)   : r={age_metrics['spearman_b0']:+.4f}  p={age_metrics['p_b0']:.2e}"
        f"  (expected: negative — worse baseline -> older)")
    log()

    # Pearson correlation matrix: age vs all features
    corr_cols = ["beta0", "beta1", "beta2", "overall_accuracy",
                 "log_vocab_breadth", "log_total_events", "avg_delta_days", "learning_age"]
    corr_matrix = user_feats[corr_cols].corr(method="spearman")
    log("  Spearman correlation matrix (key features vs learning_age):")
    age_corrs = corr_matrix["learning_age"].drop("learning_age").sort_values()
    for feat, corr in age_corrs.items():
        bar = "+" * int(abs(corr) * 20) if corr > 0 else "-" * int(abs(corr) * 20)
        log(f"    {feat:<25} {corr:>+.4f}  {bar}")
    log()
    log("=" * 62)

    report_path = output_dir / "metrics_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"    Report saved to {report_path}")


# ═══════════════════════════════════════════════════════════
# VISUALISATIONS
# ═══════════════════════════════════════════════════════════

def plot_results(user_feats: pd.DataFrame, model_metrics: dict, output_dir: Path):
    print("\n[Plots] Generating visualisations ...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. b1 distribution ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(user_feats["beta1"], bins=60, color="#4C9BE8", edgecolor="white")
    ax.axvline(user_feats["beta1"].median(), color="navy", linewidth=1.5,
               linestyle="--", label=f"Median={user_feats['beta1'].median():.3f}")
    ax.set_xlabel("b1 — Forgetting Rate (more negative = forgets faster)")
    ax.set_ylabel("Number of Users")
    ax.set_title("Distribution of Forgetting Rate (b1) across Users")
    ax.legend(); plt.tight_layout()
    fig.savefig(output_dir / "b1_distribution.png", dpi=150); plt.close(fig)

    # ── 2. Predicted vs actual b1 ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    y_true = user_feats["beta1"].values
    y_pred = model_metrics["y_pred"]
    ax.scatter(y_true, y_pred, s=8, alpha=0.3, color="#4C9BE8")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual b1"); ax.set_ylabel("Predicted b1")
    ax.set_title(f"GBM: Predicted vs Actual Forgetting Rate\n"
                 f"CV R2={model_metrics['cv_r2_mean']:.3f}  "
                 f"CV MAE={model_metrics['cv_mae_mean']:.4f}")
    ax.legend(); plt.tight_layout()
    fig.savefig(output_dir / "b1_pred_vs_actual.png", dpi=150); plt.close(fig)

    # ── 3. Feature importances ────────────────────────────────────
    imp_df = model_metrics["importances"]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2E86AB" if i == 0 else "#4C9BE8" for i in range(len(imp_df))]
    ax.barh(imp_df["feature"], imp_df["importance"], color=colors)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Which behaviours predict forgetting rate (b1)?\nGBM Feature Importances")
    ax.invert_yaxis(); plt.tight_layout()
    fig.savefig(output_dir / "feature_importances.png", dpi=150); plt.close(fig)

    # ── 4. b1 per-user R2 distribution ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(user_feats["beta_r2"].dropna(), bins=60,
            color="#4C9BE8", edgecolor="white")
    ax.axvline(0, color="red", linewidth=1.5, linestyle="--", label="R2=0")
    ax.axvline(user_feats["beta_r2"].median(), color="navy", linewidth=1.5,
               label=f"Median={user_feats['beta_r2'].median():.3f}")
    ax.set_xlabel("Per-user R2 (memory model fit)")
    ax.set_ylabel("Number of Users")
    ax.set_title("Distribution of per-user Memory Model R2")
    ax.legend(); plt.tight_layout()
    fig.savefig(output_dir / "memory_model_r2.png", dpi=150); plt.close(fig)

    # ── 5. Learning age distribution ─────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(user_feats["learning_age"], bins=50,
            color="#4C9BE8", edgecolor="white")
    ymax = ax.get_ylim()[1]
    for lo, hi, lbl in AGE_BRACKETS:
        ax.axvline(lo, color="gray", linestyle="--", alpha=0.4)
        ax.text((lo + hi) / 2, ymax * 0.92, lbl.split("(")[0].strip(),
                fontsize=7, ha="center", color="dimgray", rotation=15)
    ax.set_xlabel("Learning Age (years)")
    ax.set_ylabel("Number of Users")
    ax.set_title("Distribution of Learning Age Index")
    plt.tight_layout()
    fig.savefig(output_dir / "age_distribution.png", dpi=150); plt.close(fig)

    # ── 6. b1 vs learning age scatter (sanity check) ─────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(user_feats["beta1"], user_feats["learning_age"],
               s=8, alpha=0.3, color="#4C9BE8")
    # add regression line
    m, b = np.polyfit(user_feats["beta1"], user_feats["learning_age"], 1)
    x_line = np.linspace(user_feats["beta1"].min(), user_feats["beta1"].max(), 100)
    ax.plot(x_line, m * x_line + b, "r-", linewidth=1.5, label="Trend")
    ax.set_xlabel("b1 — Forgetting Rate")
    ax.set_ylabel("Learning Age (years)")
    ax.set_title("Sanity Check: Forgetting Rate vs Learning Age\n"
                 "(more negative b1 should -> older learning age)")
    ax.legend(); plt.tight_layout()
    fig.savefig(output_dir / "b1_vs_age_sanity.png", dpi=150); plt.close(fig)

    # ── 7. Score-to-age mapping curve: side-by-side with critical period ──
    from scipy.stats import lognorm as _lognorm

    scores = np.linspace(0, 1, 500)
    ages   = score_to_age(scores)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Learning Age: Mirroring the Critical Period Curve",
                 fontsize=14, fontweight="bold")

    # Left panel: full critical period bell curve for reference
    ax = axes[0]
    ages_full = np.linspace(0, 90, 500)
    ease = _lognorm.pdf(ages_full, s=0.6, scale=np.exp(np.log(10)))
    ease = ease / ease.max()
    ax.plot(ages_full, ease, color="darkred", linewidth=2.5)
    ax.axvline(AGE_PEAK, color="steelblue", linestyle="--", linewidth=1.5,
               label=f"Peak = age {AGE_PEAK}")
    ax.fill_betweenx([0, 1.05], AGE_PEAK, 90, alpha=0.10, color="steelblue",
                     label="Right side (our range)")
    ax.set_xlabel("Age (Years)", fontsize=12, color="darkred", fontweight="bold")
    ax.set_ylabel("Ease of Learning", fontsize=12, color="darkred", fontweight="bold")
    ax.set_title("Reference: Critical Period Bell Curve\n"
                 "(we model the shaded right side)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 90); ax.set_ylim(0, 1.15)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Right panel: our score -> age mapping
    ax = axes[1]
    ax.plot(scores, ages, color="#2E86AB", linewidth=3,
            label=f"age = {AGE_PEAK} + {AGE_MAX-AGE_PEAK}"
                  r" * (1-s)^0.5 * exp(-3s)")
    key_scores = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    for ks in key_scores:
        ka = score_to_age(np.array([ks]))[0]
        ax.scatter([ks], [ka], s=60, color="#2E86AB", zorder=5)
        ax.annotate(f"age {ka:.0f}", (ks, ka),
                    textcoords="offset points", xytext=(6, -4),
                    fontsize=8, color="#1a5c7a")
    ax.set_xlabel(
        "Composite Score  (b1=forgetting, b2=learning speed, b0=baseline,\n"
        "vocab breadth, accuracy, effort, frequency)\n"
        "Score = 1 (best learner)  →  Score = 0 (weakest learner)",
        fontsize=10)
    ax.set_ylabel("Learning Age (years)", fontsize=11)
    ax.set_title("Our Score → Age Mapping\n"
                 "(mirrors right side of critical period curve)", fontsize=11)
    ax.set_xlim(1.02, -0.02)   # inverted: best on left, worst on right
    ax.set_ylim(15, 85)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Add bracket shading
    bracket_colors = ["#d4efff", "#aed9f7", "#7ec8f0", "#4baee0", "#1a7db5"]
    for (lo, hi, lbl), col in zip(AGE_BRACKETS, bracket_colors):
        ax.axhspan(lo, hi, alpha=0.15, color=col)
        ax.text(0.01, (lo + hi) / 2, lbl.split("(")[0].strip(),
                fontsize=7, va="center", color="steelblue", alpha=0.8)

    plt.tight_layout()
    fig.savefig(output_dir / "score_to_age_curve.png", dpi=150,
                bbox_inches="tight"); plt.close(fig)

    # ── 8. PCA coloured by learning age ──────────────────────────
    all_cols = PREDICTOR_COLS + ["beta1", "beta2"]
    feat_matrix = user_feats[all_cols].fillna(user_feats[all_cols].median())
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(feat_matrix)
    pca    = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_sc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("User Feature Space (PCA)", fontsize=14, fontweight="bold")

    sc1 = ax1.scatter(coords[:, 0], coords[:, 1],
                      c=user_feats["beta1"].values,
                      s=6, alpha=0.5, cmap="RdYlGn_r")
    plt.colorbar(sc1, ax=ax1, label="b1 (Forgetting Rate)")
    ax1.set_title("Coloured by Forgetting Rate (b1)")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    sc2 = ax2.scatter(coords[:, 0], coords[:, 1],
                      c=user_feats["learning_age"].values,
                      s=6, alpha=0.5, cmap="RdYlGn_r")
    plt.colorbar(sc2, ax=ax2, label="Learning Age (years)")
    ax2.set_title("Coloured by Learning Age Index")
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    plt.tight_layout()
    fig.savefig(output_dir / "pca_overview.png", dpi=150); plt.close(fig)

    # ── 9. Spearman correlation heatmap ──────────────────────────
    corr_cols = ["beta0", "beta1", "beta2", "overall_accuracy", "log_vocab_breadth",
                 "log_total_events", "avg_delta_days", "session_acc_rate",
                 "learning_age", "composite_score"]
    corr_matrix = user_feats[corr_cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, square=True)
    ax.set_title("Spearman Correlation Matrix\n(features, b1, learning age)")
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=150); plt.close(fig)

    print(f"    Plots saved to {output_dir}/")


# ═══════════════════════════════════════════════════════════
# SINGLE-USER PROFILE CARD
# ═══════════════════════════════════════════════════════════

def print_profile(user_id: str, user_feats: pd.DataFrame):
    if user_id not in user_feats.index:
        print(f"User '{user_id}' not found.")
        return
    r = user_feats.loc[user_id]
    filled = int((r["learning_age"] - AGE_PEAK) / (AGE_MAX - AGE_PEAK) * 30)
    bar    = "X" * filled + "." * (30 - filled)
    print(f"""
+======================================================+
|           DUOLINGO  LEARNING AGE  CARD               |
+======================================================+
|  User ID       : {str(user_id):<35}|
|  Learning Age  : {r['learning_age']:>3} years                        |
|  90% CI range  : {r['learning_age_lo']:>3} - {r['learning_age_hi']:>3} years                  |
|  Age Bracket   : {r['age_bracket']:<35}|
+------------------------------------------------------+
|  Age 20 [{bar}] 80 |
+------------------------------------------------------+
|  b0 (baseline)       : {r['beta0']:>+8.4f}                      |
|  b1 (forgetting)     : {r['beta1']:>+8.4f}  [primary signal]    |
|  b2 (learning speed) : {r['beta2']:>+8.4f}                      |
|  Composite score     : {r['composite_score']:>8.4f}                      |
+------------------------------------------------------+
|  Recall accuracy  : {r['overall_accuracy']:>6.1%}                         |
|  Vocabulary seen  : {int(r['vocab_breadth']):>6,}  words                 |
|  Total events     : {int(r['total_events']):>6,}                         |
|  Avg gap          : {r['avg_delta_days']:>6.1f}  days                   |
+======================================================+
""")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Duolingo Learning Age Pipeline")
    parser.add_argument("--data",    type=str, default=None,
                        help="Override DATA_PATH")
    parser.add_argument("--sample",  type=int, default=None,
                        help="Subsample N rows")
    parser.add_argument("--output",  type=str, default=None,
                        help="Override OUTPUT_DIR")
    parser.add_argument("--user_id", type=str, default=None,
                        help="Print profile card for a specific user_id")
    args = parser.parse_args()

    data_path = args.data   or DATA_PATH
    out_dir   = Path(args.output or OUTPUT_DIR)
    sample    = args.sample or SAMPLE_SIZE

    # ── Pipeline ──────────────────────────────────────────────────
    df                        = load_data(data_path, sample=sample)
    betas, beta_metrics       = estimate_user_betas(df)
    user_feats                = build_user_features(df, betas)
    gbm, scaler, model_metrics= fit_forgetting_model(user_feats)
    user_feats, age_metrics   = compute_learning_age(user_feats)

    compute_metrics_report(user_feats, beta_metrics, model_metrics,
                           age_metrics, out_dir)
    plot_results(user_feats, model_metrics, out_dir)

    # Save CSV
    out_csv = out_dir / "user_learning_ages.csv"
    user_feats.to_csv(out_csv)
    print(f"\nResults saved to {out_csv}")

    # Profile card
    uid = args.user_id or user_feats.index[0]
    print_profile(uid, user_feats)

    print("\nPipeline complete!")
    return user_feats, model_metrics


if __name__ == "__main__":
    main()


