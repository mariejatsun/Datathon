# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 11:22:37 2026

@author: janas
"""

import pandas as pd
import glob

folder = r"C:\Users\janas\OneDrive\Documenten\KU LEUVEN\DATATHON\Datathon\notif_data"
files = glob.glob(folder + "/*.parquet")



dfs = []

for f in files:
    print("Reading:", f)
    df_part = pd.read_parquet(f, engine='fastparquet')
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)


df["hours_since_start"] = df["datetime"] / 3600
df["time_bin"] = pd.qcut(df["hours_since_start"], q=4,
                         labels=["early","mid1","mid2","late"])
df.groupby("time_bin")["session_end_completed"].mean()



template_profile = (
    df.groupby("selected_template")["session_end_completed"]
      .mean()
      .sort_values(ascending=False)
)
print(template_profile)


lang_template_profile = (
    df.groupby(["ui_language", "selected_template"])["session_end_completed"]
      .mean()
      .unstack()
)
print(lang_template_profile)

time_template_profile = (
    df.groupby(["time_bin","selected_template"])
      ["session_end_completed"]
      .mean()
      .unstack()
)
print(time_template_profile)



################## visualisatie 
import seaborn as sns
import matplotlib.pyplot as plt

lang_template_profile = (
    df.groupby(["ui_language", "selected_template"])
      ["session_end_completed"]
      .mean()
      .unstack()
)

plt.figure()
sns.heatmap(lang_template_profile, annot=True)
plt.title("Notification Effectiveness per Language")
plt.ylabel("UI Language")
plt.xlabel("Template")
plt.show()


template_profile = (
    df.groupby("selected_template")["session_end_completed"]
      .mean()
      .sort_values(ascending=False)
)

template_profile.plot(kind="bar")
plt.title("Overall Template Performance")
plt.ylabel("Engagement Rate")
plt.show()