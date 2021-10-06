import numpy as np
import pandas as pd
import scipy.stats as scs

from . import helpers


def wilcoxon_rank_sums(comparison_column: str, n_stages: int, stage_method: str, data: pd.DataFrame):
    x_stages = []
    y_stages = []
    p_vals = []
    q_vals = []
    statistics = []
    for x_stage, y_stage in helpers.stage_combination(n_stages):
        x = data.loc[data[stage_method] == x_stage][comparison_column]
        y = data.loc[data[stage_method] == y_stage][comparison_column]

        statistic, p_val = scs.ranksums(x, y)

        # p_vals[f"{x_stage} - {y_stage}"] = p_val
        # q_vals[f"{x_stage} - {y_stage}"] = p_val * (n_stages - 1)
        x_stages.append(x_stage)
        y_stages.append(y_stage)

        statistics.append(statistic)

        p_vals.append(p_val)
        q_vals.append(p_val * (n_stages - 1))

    return pd.DataFrame({"stage_a": x_stages,
                         "stage_b": y_stages,
                         "statistic": statistics,
                         "p_val": p_vals,
                         "q_val": q_vals})


def localization_counts(localizations: pd.DataFrame, stages: pd.DataFrame,
                        file_label: str, n_loc_label: str,
                        method: str) -> (pd.DataFrame, pd.DataFrame):

    counts = get_loc_counts_by_file(localizations, file_label, n_loc_label)

    count_stages = stages.set_index(file_label).join(counts, on=file_label).reset_index()

    description = count_stages.groupby(method)[n_loc_label].describe()
    description = assign_sem(description)

    return description, count_stages


def assign_sem(data: pd.DataFrame, sd_col: str = "std", n_col: str = "count") -> pd.DataFrame:
    """
    Calculates SEM and appends it as a new column.

    :param data: Dataframe, ideally created by using the pd.DataFrame.describe() method.
    :param sd_col: Name of the column in data containing the standard deviation data
    :param n_col: Name of the column in data containing the number of observations
    :return: data with an additional sem column
    """
    return data.assign(sem=data[sd_col] / np.sqrt(data[n_col]))


def loc_counts_by_column(localizations: pd.DataFrame, column: str,
                         file_label: str = "file", n_loc_label: str = "n_localizations"):
    counts = get_loc_counts_by_file(localizations, file_label, n_loc_label)

    joined_counts = localizations.set_index(file_label).join(counts, on=file_label).reset_index()

    description = joined_counts.groupby(column)[n_loc_label].describe()

    description = assign_sem(description)

    return description


def get_loc_counts_by_file(localizations: pd.DataFrame,
                           file_label: str = "file",
                           n_loc_label: str = "n_localizations"):
    counts = localizations.value_counts(subset=[file_label], sort=False)
    counts.name = n_loc_label
    counts = counts.reset_index().set_index(file_label)
    return counts


def describe_hist_distribution(data: pd.DataFrame, desc_col: str, group: str = None) -> pd.DataFrame:
    if group is not None:
        result = data.groupby(group)[desc_col].describe()

    else:
        result = data[desc_col].describe()

    return result
