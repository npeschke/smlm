import pandas as pd


def get_tissue_label(s: pd.Series):
    col_a = "condensed"
    col_b = "aCasp3_signal"

    col_a_desc = "Condensed"
    col_b_desc = "aCasp3 "

    label = ""
    if not s[col_a]:
        label = f"Not "

    label += f"{col_a_desc} | {col_b_desc}"

    if s[col_b]:
        label += "pos."
    else:
        label += "neg."

    return label
