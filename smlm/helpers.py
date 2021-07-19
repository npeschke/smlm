import pandas as pd
import pathlib as pl


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


def stage_combination(n_stages: int):
    for first_stage in range(1, n_stages):
        yield first_stage, first_stage + 1


def merge_localizations_stages(data: pd.DataFrame, stages_df: pd.DataFrame, assure_cols: tuple = None):
    merged = data.merge(stages_df, on="file", how="left")

    if assure_cols is not None:
        for col in assure_cols:
            merged = merged.loc[~merged[col].isna()]

    return merged


def assure_dir(directory: pl.Path):
    if not directory.exists():
        directory.mkdir(parents=True)
