import pandas as pd
import matplotlib.pyplot as plt

betas_df = pd.read_csv('betas_df_eval.csv', delimiter=';')

print(betas_df[["r2", "r2_adj", "rmse"]].describe())

# What share of users have a decent fit?
print((betas_df["r2"] > 0.1).mean())   # fraction with R² > 0.1
print((betas_df["r2"] > 0.3).mean())   # fraction with R² > 0.3

# Are the betas significant for most users?
print((betas_df["pval_log_d"] < 0.05).mean())   # fraction where delta matters
print((betas_df["pval_log_hs"] < 0.05).mean())  # fraction where history matters
print((betas_df["beta1"] < 0).mean())  # should be close to 1.0
print((betas_df["beta2"] > 0).mean())  # should be close to 1.0