import csv
import sys
from collections import defaultdict

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix,
)

from src.utils.homographs import NON_HOMOGRAPHS_SET

VOWELS = "aeiouəʌɛɪʊ"
CLASSES = ["Malumay", "Malumi", "Mabilis", "Maragsa", "None", "Nonstandard"]

"""
    Malumay - 'buhay - stress on penult
    Malumi - 'pasoʔ - stress on penult + ends in glottal stop
    Mabilis - bu'hay - stress on ult
    Maragsa - pa'soʔ - stress on ult + ends in glottal stop
"""


def syllabicate(phonetic_word):
    """Splits a Tagalog IPA string into syllables"""

    tokens = list(phonetic_word)
    syllables, current = [], ""
    i = 0
    while i < len(tokens):
        ch = tokens[i]
        if ch == "'":
            if current:
                syllables.append(current)
            current = ch
            i += 1
            continue
        current += ch
        if ch in VOWELS:
            j = i + 1
            while j < len(tokens):
                ahead = tokens[j]
                if ahead == "'" or ahead in VOWELS:
                    break
                k = j + 1
                while k < len(tokens) and tokens[k] == "'":
                    k += 1
                if k < len(tokens) and tokens[k] in VOWELS:
                    break
                current += ahead
                j += 1
            i = j
            syllables.append(current)
            current = ""
            continue
        i += 1
    if current:
        syllables.append(current)
    return syllables


def classify_stress(phonetic_word):
    """Classifies a word into one of four stress classes"""

    if not phonetic_word or not isinstance(phonetic_word, str):
        return "None"  # default
    if "'" not in phonetic_word:
        return "None"

    has_glottal = phonetic_word.endswith("ʔ")
    syllables = syllabicate(phonetic_word)

    if not syllables:
        return "None"

    is_last_stressed = "'" in syllables[-1]
    is_penult_stressed = ("'" in syllables[-2]) if len(syllables) > 1 else False

    if is_last_stressed:
        return "Maragsa" if has_glottal else "Mabilis"
    elif is_penult_stressed:
        return "Malumi" if has_glottal else "Malumay"
    else:
        return "Nonstandard"


def compute_metrics(y_true, y_pred):
    """Calculates precision, recall, F1 per class + multiclass confusion matrix"""

    report = classification_report(
        y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0
    )
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=CLASSES)

    results = {}
    for i, cls in enumerate(CLASSES):
        tn, fp, fn, tp = mcm[i].ravel()
        results[cls] = {
            "Precision": report[cls]["precision"],
            "Recall": report[cls]["recall"],
            "F1": report[cls]["f1-score"],
            "Support": int(report[cls]["support"]),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
        }

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES, normalize="true")
    confusion_matrix_df = pd.DataFrame(
        cm,
        index=pd.Index(CLASSES, name="Actual"),
        columns=pd.Index(CLASSES, name="Predicted"),
    )

    return results, confusion_matrix_df


def run_evaluation(pkl_path: str):
    y_true, y_pred, table_rows = [], [], []

    df = pd.read_pickle(pkl_path)

    for row_idx, (_, row) in enumerate(df.iterrows(), 1):
        grapheme_words = str(row.get("sentence", "")).split()
        target_words = str(row.get("target", "")).split()
        pred_words = str(row.get("predicted", "")).split()

        for g, t, p in zip(grapheme_words, target_words, pred_words):
            # TODO: Can comment this out to include everything
            if g in NON_HOMOGRAPHS_SET:
                continue

            tc = classify_stress(t)
            pc = classify_stress(p)
            match = "yes" if tc == pc else "no"

            table_rows.append((row_idx, t, tc, p, pc, match))
            y_true.append(tc)
            y_pred.append(pc)

    metrics, confusion_matrix = compute_metrics(y_true, y_pred)
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = "Class"

    print("Metrics")
    print(metrics_df)

    print("\nMetrics (LaTeX)")
    print(metrics_df.to_latex(index=True, float_format="%.2f"))

    print("Confusion matrix")
    print(confusion_matrix)

    for y in range(num_classes := len(CLASSES)):
        for x in range(num_classes):
            val = confusion_matrix.iloc[y, x]
            print(f"{x} {y} {val:.2f}")

        print()


run_evaluation(
    "results/output_2026-04-03_00-54_tatoeba_newsph_stress_word_checkpoint-9670_manual.pkl"
)
