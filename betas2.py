import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# ----------------------------
# Load + clean data
# ----------------------------
df = pd.read_csv("learning_traces.13m.csv")

EPS = 1e-6

df = df[(df["delta"] > 0) & (df["history_seen"] > 0)].copy()
df["p_recall"] = df["p_recall"].clip(EPS, 1 - EPS)

# Log transforms
df["log_d"] = np.log(df["delta"])
df["log_hs"] = np.log(df["history_seen"])

# Logit transform
df["logit_p"] = np.log(df["p_recall"] / (1 - df["p_recall"]))

# ----------------------------
# Apply min_N = 200 constraint
# ----------------------------
MIN_N = 200
counts = df["user_id"].value_counts()
valid_users = counts[counts >= MIN_N].index

df_filt = df[df["user_id"].isin(valid_users)].copy()

print(f"Users kept: {len(valid_users)}")
print(f"Rows kept: {len(df_filt)}")

# ----------------------------
# Mixed Effects Model
# ----------------------------
print("Fitting mixed-effects model...")

# Recommended for speed:
#   random intercept + random slope for log_d
#   (log_hs slope is usually stable across users)
model = smf.mixedlm(
    "logit_p ~ log_d + log_hs",
    data=df_filt,
    groups=df_filt["user_id"],
    re_formula= "1"
)

res = model.fit(method="lbfgs")
print(res.summary())

# ----------------------------
# Extract user-level betas
# ----------------------------
fixed = res.fe_params

rows = []
for uid, re in res.random_effects.items():
    beta0 = fixed["Intercept"] + re.get("Intercept", 0.0)
    beta1 = fixed["log_d"] + re.get("log_d", 0.0)
    beta2 = fixed["log_hs"]  # no random slope for log_hs

    rows.append({
        "user_id": uid,
        "beta0": beta0,
        "beta1": beta1,
        "beta2": beta2
    })

betas_df = pd.DataFrame(rows)
print(betas_df.head())

betas_df.to_csv("betas_df_mixed.csv", sep=";", index=False)
print("Saved betas_df_mixed.csv")