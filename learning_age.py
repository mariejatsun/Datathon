# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 20:19:06 2026

@author: janas
"""

"""
Duolingo Learning Age — Minimal Pipeline
Outputs:
  1) user_learning_ages.csv          (volledige user_feats incl. betas + age)
  2) user_learning_age_for_app.csv   (mini: id + age + CI + bracket + score)
"""

# ──────────────────────────────────────────
#  SET YOUR DATA PATH HERE (FILE, not folder)
# ──────────────────────────────────────────
DATA_PATH   = r"C:\Users\thoma\OneDrive\Documenten\KUL\Master AI\Datathon\Spaced Repetition Data\Spaced Repetition Data"  # or .csv
SAMPLE_SIZE = None   # None = full dataset
OUTPUT_DIR  = "."    # "." = current folder (waar je script runt)
# ──────────────────────────────────────────

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from scipy.stats import beta

warnings.filterwarnings("ignore")

RANDOM_STATE        = 42
MIN_EVENTS_PER_USER = 1500

AGE_PEAK     = 18
AGE_MAX      = 80
DECAY_ALPHA  = 0.5
DECAY_BETA   = 3.0


SCORE_WEIGHTS = {
    "retention":    0.25,
    "learning_spd": 0.20,
    "baseline":     0.15,
    "vocab":        0.15,
    "accuracy":     0.10,
    "effort":       0.10,
    "frequency":    0.05,
}

AGE_BRACKETS = [
    (18, 25, "Peak Learner (18-25)"),
    (26, 35, "Early Decline (26-35)"),
    (36, 50, "Mid Decline (36-50)"),
    (51, 65, "Late Decline (51-65)"),
    (66, 80, "Advanced Decline (66-80)"),
]

PREDICTOR_COLS = [
    "overall_accuracy",
    "log_vocab_breadth",
    "log_avg_history_seen",
    "avg_delta_days",
    "std_delta_days",
    "session_acc_rate",
    "log_total_events",
]

def load_data(path: str, sample: int | None = None) -> pd.DataFrame:
    print(f"[1/3] Loading data: {path}")
    df = pd.read_csv(path, sep=",", low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {"p_recall", "timestamp", "delta", "user_id",
                "history_seen", "history_correct",
                "session_seen", "session_correct", "lexeme_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE)

    df["p_recall"]        = df["p_recall"].clip(1e-6, 1 - 1e-6)
    df["delta"]           = df["delta"].clip(0)
    df["history_seen"]    = df["history_seen"].clip(0)
    df["history_correct"] = df["history_correct"].clip(0)

    print(f"    Rows: {len(df):,} | Users: {df['user_id'].nunique():,}")
    return df

def estimate_user_betas(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/3] Fitting per-user betas ...")

    df = df.copy()
    df["log_p"]  = np.log(df["p_recall"])
    df["log_d"]  = np.log(df["delta"] + 1)
    df["log_hs"] = np.log(df["history_seen"] + 1)

    records = []
    for uid, g in df.groupby("user_id"):
        if len(g) < MIN_EVENTS_PER_USER:
            continue
        X = g[["log_d", "log_hs"]].values
        y = g["log_p"].values
        reg = Ridge(alpha=0.01, fit_intercept=True)
        try:
            reg.fit(X, y)
            records.append({
                "user_id": uid,
                "beta0": reg.intercept_,
                "beta1": reg.coef_[0],
                "beta2": reg.coef_[1],
                "n_events": len(g),
            })
        except Exception:
            continue

    betas = pd.DataFrame(records).set_index("user_id")
    print(f"    Users fitted: {len(betas):,}")
    return betas

def build_user_features(df: pd.DataFrame, betas: pd.DataFrame) -> pd.DataFrame:
    feats = df.groupby("user_id").agg(
        overall_accuracy=("p_recall", "mean"),
        vocab_breadth=("lexeme_id", "nunique"),
        total_events=("p_recall", "count"),
        avg_history_seen=("history_seen", "mean"),
        avg_delta_days=("delta", lambda x: x.mean() / 86400),
        std_delta_days=("delta", lambda x: x.std() / 86400),
        session_accuracy=("session_correct", "sum"),
        session_total=("session_seen", "sum"),
    )
    feats["session_acc_rate"] = feats["session_accuracy"] / feats["session_total"].clip(1)
    feats.drop(columns=["session_accuracy", "session_total"], inplace=True)

    for col in ["vocab_breadth", "total_events", "avg_history_seen"]:
        feats[f"log_{col}"] = np.log1p(feats[col])

    combined = betas.join(feats, how="inner")
    combined.fillna(combined.median(numeric_only=True), inplace=True)
    return combined

def score_to_age(score: np.ndarray) -> np.ndarray:
    span = AGE_MAX - AGE_PEAK
    return AGE_PEAK + span * (1.0 - score)

def compute_learning_age(user_feats: pd.DataFrame) -> pd.DataFrame:
    uf = user_feats.copy()

    def minmax_data(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9)

    components = pd.DataFrame(index=uf.index)

    components["vocab"]     = minmax_data(uf["log_vocab_breadth"])
    components["accuracy"]  = minmax_data(uf["overall_accuracy"])
    components["effort"]    = minmax_data(uf["log_total_events"])
    components["frequency"] = minmax_data(-uf["avg_delta_days"])

    w = np.array([SCORE_WEIGHTS[k] for k in components.columns])
    score = components.values @ w

    # --- STRETCH SCORE to [0, 1] ---
    s_min, s_max = np.percentile(score, 5), np.percentile(score, 95)
    s_stretched = np.clip((score - s_min) / (s_max - s_min + 1e-9), 0, 1)

    # --- EXPONENTIAL DECAY MAPPING ---
    # High score (good learner) -> young age (20)
    # Low score (poor learner)  -> old age (60)
    # u = 0 means best learner (age=20), u = 1 means worst (age=60)
    u = 1.0 - s_stretched

    # Exponential: age = 20 + 40 * (e^(k*u) - 1) / (e^k - 1)
    # k controls steepness — higher k = more exponential
    #k = 3.0
    #age = AGE_PEAK + (60 - AGE_PEAK) * (np.exp(k * u) - 1) / (np.exp(k) - 1)

    k = 5.0
    age = AGE_PEAK + (50 - AGE_PEAK) * (np.exp(k * u) - 1) / (np.exp(k) - 1)


    uf["composite_score"] = np.round(s_stretched, 4)
    uf["learning_age"]    = np.round(age).astype(int)

    # bootstrap CI
    rng = np.random.default_rng(RANDOM_STATE)
    B = 200
    boots = np.zeros((len(uf), B))
    for i in range(B):
        noise = rng.normal(0, 0.05, size=components.shape)
        boots[:, i] = score_to_age((components.values + noise).clip(0, 1) @ w)

    uf["learning_age_lo"] = np.round(np.percentile(boots, 5,  axis=1)).astype(int)
    uf["learning_age_hi"] = np.round(np.percentile(boots, 95, axis=1)).astype(int)

    def bracket(a):
        for lo, hi, lbl in AGE_BRACKETS:
            if lo <= a <= hi:
                return lbl
        return AGE_BRACKETS[-1][2]

    uf["age_bracket"] = uf["learning_age"].apply(bracket)
    return uf




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    data_path = args.data or DATA_PATH
    sample    = args.sample or SAMPLE_SIZE
    out_dir   = Path(args.out or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df       = load_data(data_path, sample=sample)
    betas    = estimate_user_betas(df)
    feats    = build_user_features(df, betas)
    feats    = compute_learning_age(feats)



    # 2) minimal for app
    app_path = out_dir / "user_learning_age.csv"
    feats.reset_index()[[
        "user_id", "learning_age", "learning_age_lo", "learning_age_hi",
        "age_bracket", "composite_score"
    ]].to_csv(app_path, index=False)
    
    return feats


feats  = main()
    
import matplotlib.pyplot as plt

plt.figure()
plt.hist(feats["learning_age"], bins=30)
plt.xlabel("Learning Age")
plt.ylabel("Number of Users")
plt.title("Distribution of Learning Age")
plt.show()



