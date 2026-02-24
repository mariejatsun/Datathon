# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 13:54:56 2026

@author: janas
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"C:\Users\janas\Downloads\settles.acl16.learning_traces.13m.csv.gz",
    compression="gzip"
)

EPS = 1e-6


df = df[(df["delta"] > 0) & (df["history_seen"] > 0)].copy()
df["p_recall"] = df["p_recall"].clip(EPS, 1 - EPS)

df["log_p"]  = np.log(df["p_recall"])
df["log_d"]  = np.log(df["delta"])
df["log_hs"] = np.log(df["history_seen"])

MIN_N = 3000
rows = []

for uid, g in df.groupby("user_id"):
    if len(g) < MIN_N:
        continue

    X = g[["log_d", "log_hs"]].astype(float)
    X = sm.add_constant(X, has_constant="add")   # forceer intercept
    y = g["log_p"].astype(float)

    res = sm.OLS(y, X).fit()
    p = res.params

    beta0 = p.iloc[0]          # intercept (wat de naam ook is)
    beta1 = p["log_d"]
    beta2 = p["log_hs"]

    rows.append({
    "user_id":  uid,
    "beta0":    p.iloc[0],
    "beta1":    p["log_d"],
    "beta2":    p["log_hs"],
    "n":        len(g),
    "r2":       res.rsquared,
    "r2_adj":   res.rsquared_adj,
    "pval_log_d":  res.pvalues["log_d"],
    "pval_log_hs": res.pvalues["log_hs"],
    "rmse":     np.sqrt(res.mse_resid),
})

betas_df = pd.DataFrame(rows)
betas_df["beta_list"] = betas_df[["beta0","beta1","beta2"]].values.tolist()
print(betas_df.head())

betas_df.to_csv(
    r"betas_df_eval.csv",
    sep=";",
    index=False
)


# Are β’s unstable?
print(betas_df[["beta0","beta1","beta2"]].describe())

# check variance vs n
plt.scatter(betas_df["n"], betas_df["beta1"])
plt.xlabel("n observations")
plt.ylabel("beta1")
plt.show()

# Are effects statistically weak?
print((betas_df["pval_log_d"] < 0.05).mean())
print((betas_df["pval_log_hs"] < 0.05).mean())

# Is R² systematically tiny?
print(betas_df["r2"].describe())

plt.hist(betas_df["r2"], bins=50)
plt.show()