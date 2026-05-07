import glob
import itertools
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

"""
In order:
Tatoeba
Tatoeba+Word
Tatoeba+Stress+Word
Tatoeba+NewsPH
Tatoeba+NewsPH+Word
Tatoeba+NewsPH+Stress
Tatoeba+NewsPH+Stress+Word
Tatoeba+NewsPH+Stress+Word (30%)
Tatoeba+NewsPH+Stress+Word (50%)
"""
CHECKPOINTS = [
    "890",
    "2415",
    "4207",
    "2928",
    "4354",
    "6399",
    "9670",
    "pruned_model30",
    "pruned_model50",
    "base",
    "epitran",
]
METRICS = ["per", "cer", "pfer"]
RESULTS_DIR = "results"

data = {}


def load_pkl(path):
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
        return obj


def get_checkpoint_pkl(checkpoint):
    pattern = f"{RESULTS_DIR}/*.pkl"
    matches = [p for p in glob.glob(pattern) if (checkpoint in p and "manual.pkl" in p)]
    return matches[0]


def pairwise_wilcoxon(data, metric):
    """Compute pairwise Wilcoxon signed-rank tests for a metric."""

    checkpoints = list(data.keys())
    n = len(checkpoints)
    stat_mat = np.full((n, n), np.nan)
    pval_mat = np.full((n, n), np.nan)

    for i, j in itertools.combinations(range(n), 2):
        x = data[checkpoints[i]][metric].dropna().values
        y = data[checkpoints[j]][metric].dropna().values

        res = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        stat_mat[i, j] = res.statistic
        pval_mat[i, j] = res.pvalue

    stat_df = pd.DataFrame(stat_mat, index=checkpoints, columns=checkpoints)
    pval_df = pd.DataFrame(pval_mat, index=checkpoints, columns=checkpoints)

    return stat_df, pval_df


for checkpoint in CHECKPOINTS:
    path = get_checkpoint_pkl(checkpoint)
    df = load_pkl(path)
    data[checkpoint] = df


def format_p_values(p):
    if pd.isna(p):
        return ""  # Keeps the diagonal clean
    if p < 0.001:
        return f"<.001***"
    if p < 0.01:
        return f"{p:.4f}**".replace("0.", ".", 1)
    if p < 0.05:
        return f"{p:.4f}*".replace("0.", ".", 1)
    if p < 0.10:
        return f"{p:.4f}†".replace("0.", ".", 1)  # Highlighting the 0.0632 case
    return f"{p:.4f}".replace("0.", ".", 1)


new_names = [
    "(1) Tatoeba",
    "(2) Tatoeba+Word",
    "(3) Tatoeba+Stress+Word",
    "(4) Tatoeba+NewsPH",
    "(5) Tatoeba+NewsPH+Word",
    "(6) Tatoeba+NewsPH+Stress",
    "(7) Tatoeba+NewsPH+Stress+Word",
    "(8) $\hookrightarrow$ 30\% prune ",
    "(9) $\hookrightarrow$ 50\% prune ",
    "(10) Base CharsiuG2P",
    "(11) Epitran",
]

summaries = []

for metric in METRICS:
    print(f"Performing pairwise Wilcoxon for {metric}...")
    stat_df, pval_df = pairwise_wilcoxon(data, metric)

    stat_df.index = stat_df.columns = pval_df.index = pval_df.columns = new_names

    out_stat = f"{RESULTS_DIR}/wilcoxon_{metric}_wstat.csv"
    out_pval = f"{RESULTS_DIR}/wilcoxon_{metric}_pvalue.csv"

    stat_df.to_csv(out_stat)
    pval_df.to_csv(out_pval)

    # Apply the formatting to the entire dataframe
    latex_ready_df = pval_df.map(format_p_values)
    formatted_stat = stat_df.map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")

    for i in range(len(latex_ready_df)):
        for j in range(i + 1, len(latex_ready_df)):
            latex_ready_df.iloc[j, i] = formatted_stat.iloc[i, j]

    # Export to LaTeX code
    print(
        latex_ready_df.to_latex(
            index=True,
            escape=False,
            column_format="l" + "c" * len(latex_ready_df.columns),
            caption="Pairwise Wilcoxon Test p-values",
            label="tab:wilcoxon_results",
        )
    )

    print(pval_df)
    print("-" * 40)

    summary = pd.DataFrame(
        {
            "pooled": 0,
            "mean": [data[chk][metric].mean() * 100 for chk in CHECKPOINTS],
            "std": [data[chk][metric].std() * 100 for chk in CHECKPOINTS],
        },
        index=new_names,
    )

    summaries.append(summary)


combined_df = pd.concat(summaries, axis=1)
print(f"Descriptive statistics for {metric.upper()}:")
print(combined_df.to_latex(index=True, float_format="%.2f"))
print("=" * 40)
