import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_n_stages(method_name: str, stage_df: pd.DataFrame):
    return len(stage_df[method_name].unique())


# def get_n_stages(method: str, data: pd.DataFrame):
#     return data[[method]].nunique()


def plot_stage_histogram(plot_col: str, method: str, data: pd.DataFrame,
                         plot_label: str, method_label: str,
                         n_stages: int,
                         min_density: float = 1e-4, max_density: float = 1e0,
                         size: float = 5.0, dpi: int = 300,
                         stat: str = "probability"):
    fig, ax = plt.subplots(figsize=(size * 1.2, size), dpi=dpi)
    # data = data.loc[data[method] == 1]
    data = data.astype({method: str})
    ax = sns.histplot(x=plot_col, hue=method, data=data,
                      log_scale=(True, False),
                      hue_order=[str(i) for i in range(1, n_stages + 1)],
                      fill=False, element="step", stat=stat, common_norm=False, common_bins=False,
                      ax=ax)

    ax.set_xlabel(plot_label)
    ax.set_xlim(min_density, max_density)
    ax.legend_.set_title("Stage")

    fig.suptitle(method_label)

    return fig, ax


def plot_joint_fig(method: str, data: pd.DataFrame, stage_df: pd.DataFrame, cells_to_plot: list, labelling: str, method_display: str, dpi: int = 300):
    joint_columns = 3
    joint_rows = get_n_stages(method, stage_df)

    figure_base_size = 6.5
    # bins = 100

    color_vmin = 5e-6
    color_vmax = 1e-4

    voronoi_density_lims = (1e-4, 1e0)
    radius_lims = (0, 12000)

    joint_fig, joint_ax = plt.subplots(nrows=joint_rows, ncols=joint_columns,
                                       figsize=(joint_columns * figure_base_size * 1.2, joint_rows * figure_base_size),
                                       sharex="col", sharey="col", dpi=dpi)
    for stage_idx in range(joint_rows):
        stage_cells = data.loc[data[method] == stage_idx + 1]

        with sns.axes_style("white"):
            joint_ax[stage_idx][0] = sns.histplot(x="x", y="y", data=data.loc[data.file == cells_to_plot[stage_idx]],
                                                  ax=joint_ax[stage_idx][0],
                                                  stat="density", cmap=sns.color_palette("viridis", as_cmap=True),
                                                  binwidth=20, pthresh=0.37)

            joint_ax[stage_idx][0].set_frame_on(False)
            joint_ax[stage_idx][0].set_xticks([])
            joint_ax[stage_idx][0].set_ylabel(f"Stage {stage_idx + 1} (# Cells = {len(stage_df.loc[(stage_df.labelling == labelling) & (stage_df[method] == stage_idx + 1)])})")

        joint_ax[stage_idx][1] = sns.histplot(x="density", data=stage_cells, ax=joint_ax[stage_idx][1],
                                              log_scale=(True, False),
                                              fill=False, element="step", stat="probability")
        joint_ax[stage_idx][1].set_xlim(*voronoi_density_lims)
        joint_ax[stage_idx][1].set_xlabel(r"Voronoi Density $\left[\frac{1}{nm^2}\right]$")

        joint_ax[stage_idx][2] = sns.histplot(x="density", y="r", data=stage_cells, ax=joint_ax[stage_idx][2],
                                              log_scale=(True, False),
                                              cbar=True,
                                              stat="probability",
                                              cmap=sns.color_palette("viridis", as_cmap=True),
                                              vmin=color_vmin, vmax=color_vmax, thresh=color_vmin)
        joint_ax[stage_idx][2].set_xlabel(r"Voronoi Density $\left[\frac{1}{nm^2}\right]$")
        joint_ax[stage_idx][2].set_ylabel("Radius [nm]")
        joint_ax[stage_idx][2].set_xlim(*voronoi_density_lims)
        joint_ax[stage_idx][2].set_ylim(*radius_lims)
    joint_ax[0][0].set_title("Example Nucleus")
    joint_ax[0][1].set_title("Density Distribution")
    joint_ax[0][2].set_title("Radius/Density\nDistribution")
    joint_ax[0][0].set_yticks([])
    joint_fig.suptitle(method_display)
    joint_fig.tight_layout()
    return joint_fig
