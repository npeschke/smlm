import pandas as pd
import scipy.stats as scs

import smlm.smlm.helpers as helpers


def wilcoxon_rank_sums(comparison_column: str, n_stages: int, stage_method: str, data: pd.DataFrame):
    p_vals = {}
    q_vals = {}
    for x_stage, y_stage in helpers.stage_combination(n_stages):
        x = data.loc[data[stage_method] == x_stage][comparison_column]
        y = data.loc[data[stage_method] == y_stage][comparison_column]

        statistic, p_val = scs.ranksums(x, y)

        p_vals[f"{x_stage} - {y_stage}"] = p_val
        q_vals[f"{x_stage} - {y_stage}"] = p_val * (n_stages - 1)

    return p_vals, q_vals


