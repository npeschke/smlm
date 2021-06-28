import pathlib as pl
import textwrap as tw

import cmasher
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_col
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_tick
import numpy as np
import pandas as pd
import seaborn as sns

from smlm.smlm import config as smlm_config


def get_n_stages(method_name: str, stage_df: pd.DataFrame):
    return len(stage_df[method_name].dropna().unique())


# def get_n_stages(method: str, data: pd.DataFrame):
#     return data[[method]].nunique()


def plot_stage_histogram(plot_col: str, method: str, data: pd.DataFrame,
                         plot_label: str, method_label: str,
                         n_stages: int,
                         # min_density: float = 1e-4, max_density: float = 1e0,
                         density_lims: tuple = (-4, 0),
                         size: float = 5.0, dpi: int = 300,
                         stat: str = "density"):
    fig, ax = plt.subplots(figsize=(size * 1.6, size), dpi=dpi)
    # data = data.loc[data[method] == 1]
    data = data.astype({method: str})
    ax = sns.histplot(x=plot_col, hue=method, data=data,
                      # log_scale=(True, False),
                      hue_order=[str(i) for i in range(1, n_stages + 1)],
                      palette=smlm_config.STAGE_CMAP,
                      binrange=density_lims,
                      fill=False, element="step", stat=stat, common_norm=False, common_bins=False,
                      ax=ax)

    ax.set_xlabel(plot_label)
    if stat == "density":
        ax.set_ylabel("Probability Density Function")
    # ax.set_xlim(min_density, max_density)
    ax.legend_.set_title("Stage")

    fig.tight_layout()
    # fig.suptitle(method_label)

    return fig, ax


def plot_joint_fig(method: str, data: pd.DataFrame, stage_df: pd.DataFrame, cells_to_plot: list, labelling: str, method_display: str, dpi: int = 300):
    joint_columns = 3
    joint_rows = get_n_stages(method, stage_df)

    figure_base_size = 6.5
    # bins = 100

    color_vmin = 1e-5
    color_vmax = 8e-4

    # cmap = sns.color_palette("viridis", as_cmap=True)
    cmap = cmasher.chroma

    voronoi_density_lims = (1e-4, 1e0)
    radius_lims = (0, 8000)

    joint_fig, joint_ax = plt.subplots(nrows=joint_rows, ncols=joint_columns,
                                       figsize=(joint_columns * figure_base_size * 1.2, joint_rows * figure_base_size),
                                       sharex="col", sharey="col", dpi=dpi)
    for stage_idx in range(joint_rows):
        stage_cells = get_stage_data(data, method, stage_idx + 1)

        with sns.axes_style("white"):
            joint_ax[stage_idx][0] = sns.histplot(x="x", y="y", data=data.loc[data.file == cells_to_plot[stage_idx]],
                                                  ax=joint_ax[stage_idx][0],
                                                  stat="density", cmap=cmasher.ember,
                                                  binwidth=30, pthresh=0.10)

            joint_ax[stage_idx][0].set_frame_on(False)
            joint_ax[stage_idx][0].set_xticks([])
            joint_ax[stage_idx][0].set_ylabel(f"Stage {stage_idx + 1} (# Cells = {len(stage_df.loc[(stage_df.labelling == labelling) & (stage_df[method] == stage_idx + 1)])})")

        joint_ax[stage_idx][1] = sns.histplot(x="density", data=stage_cells, ax=joint_ax[stage_idx][1],
                                              log_scale=(True, False),
                                              fill=False, element="step", stat="density")
        joint_ax[stage_idx][1].set_xlim(*voronoi_density_lims)
        joint_ax[stage_idx][1].set_xlabel(r"Voronoi Density $\left[\frac{1}{nm^2}\right]$")

        joint_ax[stage_idx][2] = plot_r_density_dist(data=stage_cells,
                                                     density_lim=voronoi_density_lims,
                                                     radius_lim=radius_lims,
                                                     cbar_min=color_vmin, cbar_max=color_vmax,
                                                     ax=joint_ax[stage_idx][2])

        # joint_ax[stage_idx][2] = sns.histplot(x="density", y="r", data=stage_cells, ax=joint_ax[stage_idx][2],
        #                                       log_scale=(True, False),
        #                                       cbar=True,
        #                                       stat="probability",
        #                                       cmap=cmap,
        #                                       vmin=color_vmin, vmax=color_vmax, thresh=color_vmin)
        # joint_ax[stage_idx][2].set_xlabel(r"Voronoi Density $\left[\frac{1}{nm^2}\right]$")
        # joint_ax[stage_idx][2].set_ylabel("Radius [nm]")
        # joint_ax[stage_idx][2].set_xlim(*voronoi_density_lims)
        # joint_ax[stage_idx][2].set_ylim(*radius_lims)
    joint_ax[0][0].set_title("Example Nucleus")
    joint_ax[0][1].set_title("Density Distribution")
    joint_ax[0][2].set_title("Radius/Density\nDistribution")
    joint_ax[0][0].set_yticks([])
    joint_fig.suptitle(method_display)
    joint_fig.tight_layout()
    return joint_fig


def get_stage_data(data, method, stage) -> pd.DataFrame:
    return data.loc[data[method] == stage]


def plot_binary_joint_fig(method: str, data: pd.DataFrame, stage_df: pd.DataFrame, cells_to_plot: list, labelling: str, method_display: str, dpi: int = 300):
    joint_columns = 3
    joint_rows = 2

    figure_base_size = 6.5
    # bins = 100

    color_vmin = 1e-5
    color_vmax = 3e-4

    # cmap = sns.color_palette("viridis", as_cmap=True)
    cmap = cmasher.chroma

    voronoi_density_lims = (1e-4, 1e0)
    radius_lims = (0, 8000)

    joint_fig, joint_ax = plt.subplots(nrows=joint_rows, ncols=joint_columns,
                                       figsize=(joint_columns * figure_base_size * 1.2, joint_rows * figure_base_size),
                                       sharex="col", sharey="col", dpi=dpi)
    for stage_idx in range(joint_rows):
        stage_cells = data.loc[data[method] == bool(stage_idx)]

        # with sns.axes_style("white"):
        #     joint_ax[stage_idx][0] = sns.histplot(x="x", y="y", data=data.loc[data.file == cells_to_plot[stage_idx]],
        #                                           ax=joint_ax[stage_idx][0],
        #                                           stat="density", cmap=cmasher.ember,
        #                                           binwidth=30, pthresh=0.10)
        #
        #     joint_ax[stage_idx][0].set_frame_on(False)
        #     joint_ax[stage_idx][0].set_xticks([])
        joint_ax[stage_idx][0].set_ylabel(f"{bool(stage_idx)} (# Cells = {len(stage_df.loc[(stage_df.labelling == labelling) & (stage_df[method] == bool(stage_idx))])})")

        joint_ax[stage_idx][1] = sns.histplot(x="density", data=stage_cells, ax=joint_ax[stage_idx][1],
                                              log_scale=(True, False),
                                              fill=False, element="step", stat="probability")
        joint_ax[stage_idx][1].set_xlim(*voronoi_density_lims)
        joint_ax[stage_idx][1].set_xlabel(r"Voronoi Density $\left[\frac{1}{nm^2}\right]$")

        joint_ax[stage_idx][2] = plot_r_density_dist(data=stage_cells,
                                                     density_lim=voronoi_density_lims,
                                                     radius_lim=radius_lims,
                                                     cbar_min=color_vmin, cbar_max=color_vmax,
                                                     ax=joint_ax[stage_idx][2])

        # joint_ax[stage_idx][2] = sns.histplot(x="density", y="r", data=stage_cells, ax=joint_ax[stage_idx][2],
        #                                       log_scale=(True, False),
        #                                       cbar=True,
        #                                       stat="probability",
        #                                       cmap=cmap,
        #                                       vmin=color_vmin, vmax=color_vmax, thresh=color_vmin)
        # joint_ax[stage_idx][2].set_xlabel(r"Voronoi Density $\left[\frac{1}{nm^2}\right]$")
        # joint_ax[stage_idx][2].set_ylabel("Radius [nm]")
        # joint_ax[stage_idx][2].set_xlim(*voronoi_density_lims)
        # joint_ax[stage_idx][2].set_ylim(*radius_lims)
    joint_ax[0][0].set_title("Example Nucleus")
    joint_ax[0][1].set_title("Density Distribution")
    joint_ax[0][2].set_title("Radius/Density\nDistribution")
    joint_ax[0][0].set_yticks([])
    joint_fig.suptitle(method_display)
    joint_fig.tight_layout()
    return joint_fig


def plot_r_density_dist_stages(data: pd.DataFrame,
                               method: str,
                               dpi: int = 300):
    cols = get_n_stages(method, data)
    fig, ax = plt.subplots(ncols=cols,
                           figsize=(smlm_config.FIG_BASE_SIZE * cols, smlm_config.FIG_BASE_SIZE),
                           sharex="col", sharey="row",
                           dpi=dpi)

    for col_idx in range(cols):
        stage = col_idx + 1
        ax[col_idx] = plot_r_density_dist(data=get_stage_data(data, method, stage),
                                          density_lim=smlm_config.DENSITY_LIM,
                                          radius_lim=smlm_config.RADIUS_LIM,
                                          cbar_min=smlm_config.CBAR_MIN,
                                          cbar_max=smlm_config.CBAR_MAX,
                                          ax=ax[col_idx])

    # for ax in ax[:-1]:
    #     ax.

    fig.suptitle("Radius/Density Distribution")
    fig.tight_layout()

    return fig, ax


def _sci_fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


def plot_r_density_dist_stages_separate(data: pd.DataFrame,
                                        method: str,
                                        result_dir: pl.Path,
                                        plot_col_radius: str = "r",
                                        plot_label_radius: str = "Radius [nm]",
                                        plot_col_density: str = "log_norm_density",
                                        plot_label_density: str = r"Voronoi Density $\left[\frac{1}{nm^2}\right]$",
                                        dpi: int = 300, ):
    n_stages = get_n_stages(method, data)
    separate_dir = result_dir.joinpath("r_density_plots")
    if not separate_dir.exists():
        separate_dir.mkdir()

    for stage_idx in range(n_stages):
        stage = stage_idx + 1
        fig, ax = plt.subplots(figsize=(smlm_config.FIG_BASE_SIZE, smlm_config.FIG_BASE_SIZE),
                               dpi=dpi)

        plot_r_density_dist(data=get_stage_data(data, method, stage),
                            plot_col_density=plot_col_density,
                            plot_col_radius=plot_col_radius,
                            density_lim=smlm_config.DENSITY_LIM,
                            radius_lim=smlm_config.RADIUS_LIM,
                            cbar_min=smlm_config.CBAR_MIN,
                            cbar_max=smlm_config.CBAR_MAX,
                            plot_label_density=plot_label_density,
                            plot_label_radius=plot_label_radius,
                            ax=ax)

        fig.tight_layout()

        fig.savefig(separate_dir.joinpath(f"stage_{stage}.png"))
        # break

    cbar_fig, cbar_ax = plt.subplots(figsize=(smlm_config.FIG_BASE_SIZE, smlm_config.FIG_BASE_SIZE),
                                     dpi=dpi)

    # cbar_fig.colorbar(
    #     mpl_cm.ScalarMappable(
    #         norm=mpl_col.Normalize(
    #             vmin=smlm_config.CBAR_MIN,
    #             vmax=smlm_config.CBAR_MAX
    #         ),
    #         cmap=smlm_config.SEQ_CMAP
    #     )
    # )

    # cbar_ax.yaxis.get_major_formatter().set_scientific(True)

    cbar_fig.colorbar(
        mpl_cm.ScalarMappable(
            norm=mpl_col.Normalize(
                vmin=smlm_config.CBAR_MIN,
                vmax=smlm_config.CBAR_MAX
            ),
            cmap=smlm_config.SEQ_CMAP
        ),
        format=mpl_tick.FuncFormatter(_sci_fmt)
    )

    cbar_fig.savefig(separate_dir.joinpath(f"cbar.svg"))


def plot_r_density_dist(data: pd.DataFrame,
                        density_lim: tuple, radius_lim: tuple,
                        cbar_min: float, cbar_max: float,
                        ax: plt.Axes,
                        plot_col_density: str = "density",
                        plot_col_radius: str = "r",
                        plot_label_radius: str = "Radius [nm]",
                        plot_label_density: str = r"Voronoi Density $\left[\frac{1}{nm^2}\right]$"):
    sns.histplot(x=plot_col_density, y=plot_col_radius, data=data, ax=ax,
                 # log_scale=(True, False),
                 cbar=False,
                 stat="density",
                 cmap=smlm_config.SEQ_CMAP,
                 vmin=cbar_min, vmax=cbar_max, thresh=cbar_min)
    ax.set_xlabel(plot_label_density)
    ax.set_ylabel(plot_label_radius)
    ax.set_xlim(*density_lim)
    ax.set_ylim(*radius_lim)

    return ax


def plot_localization_counts(loc_label, method, method_label, data, size, dpi):
    loc_count_fig, loc_count_ax = plt.subplots(figsize=(size * 1.2, size), dpi=dpi)
    sns.boxplot(x=method, y=loc_label, data=data, ax=loc_count_ax)

    loc_count_ax.set_ylabel("Number of Localizations")
    loc_count_ax.set_xlabel("Stage")
    loc_count_ax.set_title(f"Localization counts for {method_label}")

    loc_count_fig.tight_layout()

    return loc_count_fig, loc_count_ax


def plot_cell_density_hist(data: pd.DataFrame, fg_filename: str,
                           ax: plt.Axes,
                           plot_col: str = "log_density", stage_method: str = "manual_5_stage",
                           stages: tuple = None,
                           stat: str = "density",
                           min_density: float = -4, max_density: float = 0,
                           **kwargs):
    # Background
    bg_plot_data = data.loc[data.file != fg_filename]

    if stages is None:
        stages = _get_stages(bg_plot_data, stage_method)

    for stage in stages:
        for file in get_stage_file_names(bg_plot_data, stage, stage_method):
            ax = _plot_cell_density_hist(data=bg_plot_data, plot_col=plot_col,
                                         file=file, stage=stage, ax=ax,
                                         color=_get_tint(smlm_config.STAGE_COLORS[stage - 1]),
                                         max_density=max_density, min_density=min_density,
                                         stage_method=stage_method, stat=stat,
                                         **kwargs)

    # Foreground
    fg_plot_data = data.loc[data.file == fg_filename]
    fg_stage = fg_plot_data[stage_method].unique()
    assert len(fg_stage) == 1
    fg_stage = fg_stage[0]
    ax = _plot_cell_density_hist(data=fg_plot_data, plot_col=plot_col,
                                 file=fg_filename, stage=fg_stage, ax=ax,
                                 color=smlm_config.STAGE_COLORS[fg_stage - 1],
                                 max_density=max_density, min_density=min_density,
                                 stage_method=stage_method, stat=stat,
                                 linewidth=4,
                                 **kwargs)

    return ax


def _plot_cell_density_hist(data: pd.DataFrame, plot_col: str, file: str, stage: int, ax: plt.Axes,
                            color: str,
                            max_density: float, min_density: float,
                            stage_method: str, stat: str, **kwargs):
    # data[["file", "manual_5_stage"]].tail()
    ax = sns.histplot(x=plot_col,
                      data=data.loc[(data[stage_method] == stage) & (data["file"] == file)],
                      color=color,
                      binrange=(min_density, max_density),
                      fill=False, element="step", stat=stat, common_norm=False, common_bins=False,
                      ax=ax, **kwargs)
    return ax


def _get_tint(hex_str: str):
    hex_colors = tw.wrap(hex_str[1:], 2)
    int_colors = np.array([int(color, 16) for color in hex_colors])
    tint_colors = np.round(int_colors + (255 - int_colors) * 0.6)
    tint_hex_str = "".join([format(int(tint_color), "02x") for tint_color in np.nditer(tint_colors)])
    return f"#{tint_hex_str}"


def get_stage_file_names(data: pd.DataFrame, stage, stage_method: str):
    if type(stage) is int:
        stage = (stage,)
    file_names = data.loc[data[stage_method].isin(stage)].file.unique()
    return tuple(file_names)


def _get_stages(data: pd.DataFrame, stage_method: str):
    stages = data[stage_method].unique()
    return tuple(stages)


def plot_cell_vis(data: pd.DataFrame, filename: str, plot_col: str, ax: plt.Axes, density_lims: tuple, log_density_threshold: float = -2):
    plot_data = data.loc[data.file == filename]

    with sns.axes_style("white"):
        # sns.histplot(x="x", y="y", data=plot_data.loc[plot_data.log_density < log_density_threshold],
        #              ax=ax,
        #              stat="count", cmap=smlm_config.SEQ_CMAP,
        #              cbar=True,
        #              binwidth=30,
        #              pthresh=0.10, pmax=0.95)

        # sns.kdeplot(x="x", y="y", data=plot_data.loc[plot_data.log_density < log_density_threshold],
        #             fill=True)

        plot_scatter_localizations(plot_data.loc[plot_data.log_density < log_density_threshold], ax, plot_col, density_lims=density_lims, alpha=0.2)
        plot_scatter_localizations(plot_data.loc[plot_data.log_density >= log_density_threshold], ax, plot_col, density_lims=density_lims, alpha=1)

        ax.set_frame_on(False)
        ax.set_xticks([])


def plot_scatter_localizations(data, ax, plot_col, density_lims: tuple, **kwargs):
    alpha = 1
    if "alpha" in kwargs:
        alpha = kwargs["alpha"]

    sns.scatterplot(x="x", y="y", hue=plot_col,
                    data=data,
                    marker="2",  # edgecolor=None, linewidth=0,
                    alpha=alpha,
                    palette=smlm_config.SEQ_CMAP,
                    hue_norm=density_lims,
                    ax=ax)


def plot_cell_vis_density(data: pd.DataFrame, filename: str, plot_kwargs: dict, stages: tuple = None, result_dir: pl.Path = pl.Path.cwd()):
    # plot_col = "log_density"
    # plot_col = "log_norm_density"

    plot_col = plot_kwargs["plot_col"]
    min_density = plot_kwargs["min_density"]
    max_density = plot_kwargs["max_density"]
    vis_density_lims = plot_kwargs["vis_density_lims"]

    # min_density = -4
    # max_density = 0
    # min_density = -10
    # max_density = -4

    # vis_density_lims = (-3, -0.5)
    # vis_density_lims = (-8.5, -6.5)

    fig, ax = plt.subplots(ncols=2, figsize=(50, 25), dpi=200)

    plot_cell_vis(data=data, filename=filename, ax=ax[0], density_lims=vis_density_lims, plot_col=plot_col)
    plot_cell_density_hist(data=data, fg_filename=filename, ax=ax[1],
                           min_density=min_density, max_density=max_density,
                           stages=stages, plot_col=plot_col)

    stage = data.loc[data.file == filename].manual_5_stage.unique()
    assert len(stage) == 1
    stage = int(stage[0])

    fig.suptitle(f"Stage {stage} Nucleus {filename}")
    fig.tight_layout()
    fig.savefig(result_dir.joinpath(f"stage_{stage}_{filename.split('.')[0]}_vis_{plot_col}.png"))
    return filename


if __name__ == '__main__':
    _get_tint(smlm_config.STAGE_COLORS[0])
