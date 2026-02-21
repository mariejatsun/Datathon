# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 13:54:56 2026

@author: janas
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv(r"C:\Users\janas\Downloads\settles.acl16.learning_traces.13m (1).csv.gz",
                 compression="gzip")

EPS = 1e-6


df = df[(df["delta"] > 0) & (df["history_seen"] > 0)].copy()
df["p_recall"] = df["p_recall"].clip(EPS, 1 - EPS)

df["log_p"]  = np.log(df["p_recall"])
df["log_d"]  = np.log(df["delta"])
df["log_hs"] = np.log(df["history_seen"])

MIN_N = 200
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

    rows.append({"user_id": uid, "beta0": beta0, "beta1": beta1, "beta2": beta2, "n": len(g)})

betas_df = pd.DataFrame(rows)
betas_df["beta_list"] = betas_df[["beta0","beta1","beta2"]].values.tolist()
print(betas_df.head())

betas_df.to_csv(
    r"C:\Users\janas\Downloads\betas_df.csv",
    sep=";",
    index=False
)