import pandas as pd

df1 = pd.read_csv("data/manual_set_new.csv")
df2 = pd.read_csv("data/manual_set_old_noleak.csv")

merged = pd.concat([df2, df1]).drop_duplicates(subset=["sentence"], keep="first")

merged.to_csv("data/manual_set_final.csv", index=False)
