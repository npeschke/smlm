import pathlib as pl

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial as spat
import seaborn as sns
import sklearn.cluster as skl_cluster

from . import analysis as san


def get_cluster_sizes(clusterer: hdbscan.HDBSCAN):
    """
    Counts the number of localizations in all clusters

    :param clusterer: the HDBSCAN object after fitting
    :return: dict with {cluster_id: size}
    """

    cluster_size = {}

    for cluster_id in clusterer.labels_:
        try:
            cluster_size[cluster_id] += 1
        except KeyError:
            cluster_size[cluster_id] = 0

    return cluster_size


def get_cluster_meta(clusterer: hdbscan.HDBSCAN, df_prefix: str):
    """
    Finds cluster metadata:
        - cluster size in number of localizations in cluster
        - persistence of the cluster as detailed in the HDBSCAN API

    :param clusterer: the HDBSCAN object after fitting
    :param df_prefix: prefix for the dataframe columns
    :return: pd.Dataframe with index cluster_id and size and persistence columns
    """

    cluster_meta = pd.DataFrame.from_dict(get_cluster_sizes(clusterer), orient="index")
    cluster_meta.columns = [f"{df_prefix}_cluster_size"]

    # persistence = pd.DataFrame(clusterer.cluster_persistence_, columns=["cluster_persistence"]).reset_index().rename(columns={"index": "cluster_id"})
    persistence = pd.DataFrame(clusterer.cluster_persistence_, columns=[f"{df_prefix}_cluster_persistence"])

    cluster_meta = cluster_meta.join(other=persistence, how="left")

    return cluster_meta


def get_dbscan_clustering(orte: pd.DataFrame, df_prefix: str, cluster_id_col: str):
    clusterer = skl_cluster.DBSCAN(11)
    clusterer.fit(orte[["x", "y"]])

    orte.insert(11, cluster_id_col, clusterer.labels_)
    orte, polygons = get_cluster_density(orte, df_prefix, cluster_id_col)
    return orte, clusterer, polygons


def get_hdbscan_clustering(orte, df_prefix: str, cluster_id_col: str):
    clusterer = hdbscan.HDBSCAN(core_dist_n_jobs=1)
    clusterer.fit(orte[["x", "y"]])

    orte.insert(11, cluster_id_col, clusterer.labels_)
    orte.insert(12, f"{df_prefix}_cl_prob", clusterer.probabilities_)

    cluster_meta = get_cluster_meta(clusterer, df_prefix)
    orte = orte.join(other=cluster_meta, on=cluster_id_col)

    orte, polygons = get_cluster_density(orte, df_prefix, cluster_id_col)
    # orte = orte.join(other=density_df, on=cluster_id_col)

    return orte, clusterer, cluster_meta, polygons


def get_cluster_density(orte: pd.DataFrame, df_prefix: str, cluster_id_col: str):
    # cluster_density = pd.DataFrame(columns=["cluster_id", "cluster_area", "cluster_density"])
    cluster_density = []
    polygons = []
    coordinate_cols = ["x", "y"]

    for cluster_id, cluster_df in orte.groupby(cluster_id_col):
        if cluster_id == -1 or cluster_df.shape[0] < 3:
            continue
        hull = spat.ConvexHull(cluster_df[coordinate_cols])
        # spat.convex_hull_plot_2d(hull)

        cluster_density.append([
            cluster_id,
            hull.volume,
            len(cluster_df) / hull.volume,
            np.sqrt(hull.volume / np.pi) * 2
        ])
        vertices = cluster_df.iloc[hull.vertices][coordinate_cols].to_numpy()
        polygons.append((cluster_id, vertices))

    density_df = pd.DataFrame(cluster_density, columns=[cluster_id_col,
                                                        f"{df_prefix}_cluster_area",
                                                        f"{df_prefix}_cluster_density",
                                                        f"{df_prefix}_cluster_diameter"])

    orte = orte.merge(density_df, on=cluster_id_col)

    return orte


def run(orte_path: pl.Path):
    # orte = san.analyze_orte(orte_path)
    orte = san.load_orte(orte_path)
    # orte = orte.sample(10000, replace=False)

    print(f"Starting clustering with {len(orte)} localizations")

    # tree = spat.cKDTree(orte[["x", "y"]])
    # clustering = scl.OPTICS(n_jobs=24)
    # clustering.fit(orte[["x", "y"]])

    orte = get_hdbscan_clustering(orte)

    print("Clustering completed, plotting...")

    # cl_size_fig, cl_size_ax = plt.subplots(figsize=(10, 10), dpi=300)
    cl_size_fig, cl_size_ax = plt.subplots(dpi=300)
    sns.scatterplot(x="x", y="y", hue="cluster_size", data=orte.dropna(),
                    marker=".", edgecolor=None, s=3, ax=cl_size_ax)
    cl_size_fig.tight_layout()
    cl_size_fig.show()

    cl_size_hist_fig, cl_size_hist_ax = plt.subplots(dpi=300)
    sns.histplot(x="cluster_size", data=orte.dropna(), ax=cl_size_hist_ax)
    cl_size_hist_fig.tight_layout()
    cl_size_hist_fig.show()

    print("end")


if __name__ == '__main__':
    pass
    # localization_path = pl.Path("../data/cut_cells/H2B_mCherry/2020_Nov13_SMLM_from_12th/1_0/merge_filter/cell3_s04_recon_1_0thre_merg_filt_cut.csv")
    # localization_path = pl.Path("../data/cut_cells/H2B_mCherry/2020_Nov13_SMLM_from_12th/1_0/merge_filter/cell2_s04_recon_1_0thre_merg_filt_cut.csv")
    # localization_path = pl.Path("../data/cut_cells/H2B_mCherry/2020_Jun_30_Stauro_LCI_SMLM/1_0/merge_filter/cell_5_thre_1_0_merge_filter.csv")

    # run(localization_path)
