import pandas as pd
import scipy.stats as scs

import smlm.smlm.helpers as helpers


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

    counts = localizations.value_counts(subset=[file_label], sort=False)
    counts.name = n_loc_label

    counts = counts.reset_index().set_index(file_label)

    count_stages = stages.set_index(file_label).join(counts, on=file_label).reset_index()

    return count_stages.groupby(method)[n_loc_label].describe(), count_stages
