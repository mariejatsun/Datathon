# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 14:03:08 2026

@author: janas
"""

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

betas_df = pd.read_csv("betas_df.csv", sep=";")
betas_df = betas_df[
    (betas_df["beta0"] < 10) &
    (betas_df["beta1"] > -1.8)
]
X = betas_df[["beta0","beta1","beta2"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
betas_df["cluster"] = kmeans.fit_predict(X_scaled)



cluster_summary = betas_df.groupby("cluster")[["beta0","beta1","beta2"]].agg(["mean"])



plt.figure()
plt.scatter(betas_df["beta1"], betas_df["beta2"], c=betas_df["cluster"])
plt.xlabel("beta1 (time decay)"); plt.ylabel("beta2 (practice effect)")
plt.title("Clusters in (beta1,beta2)")
plt.show()

plt.figure()
plt.scatter(betas_df["beta0"], betas_df["beta1"], c=betas_df["cluster"])
plt.xlabel("beta0 (baseline)"); plt.ylabel("beta1 (time decay)")
plt.title("Clusters in (beta0,beta1)")
plt.show()

print(betas_df[["beta0","beta1","beta2"]].describe())