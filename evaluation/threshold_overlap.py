# threshold_overlap.py
# AyShCXR — films that touched threshold tuning, so evaluations can exclude them.
# by Subhrakant Sethi & Ayush Singh
#
# reports/tune_thresholds.py set each finding's "detected" threshold at a target
# specificity on 300 healthy NIH films, and scored up to 70 positives per finding
# for its sensitivity report. It drew both samples from ALL of
# data/nih_full_labels.csv, not from the training split, so some of those films
# are also NIH test films. Evaluations exclude them so that no reported result is
# measured on a film that helped set the operating point.
#
# The samples are reconstructed exactly: same CSV, same label list, same
# random_state=5. Positives are drawn for every NIH finding (the tuner skips a few
# it cannot diagnose), which can only exclude MORE films, never fewer.

import pandas as pd

NIH = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
       "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
       "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"]


def threshold_tuning_films(labels_csv="data/nih_full_labels.csv"):
    """Set of 'Image Index' values used by reports/tune_thresholds.py."""
    df = pd.read_csv(labels_csv)
    df["n"] = df[NIH].sum(axis=1)
    used = set(df[df.n == 0].sample(300, random_state=5)["Image Index"])
    for f in NIH:
        k = min(70, int((df[f] == 1).sum()))
        used |= set(df[df[f] == 1].sample(k, random_state=5)["Image Index"])
    return used


if __name__ == "__main__":
    used = threshold_tuning_films()
    with open("data/test_list.txt", encoding="utf-8") as f:
        test = {ln.strip() for ln in f if ln.strip()}
    print(f"films used in threshold tuning: {len(used)}; "
          f"of which in the NIH test split: {len(used & test)}")
