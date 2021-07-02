import pathlib as pl

import hdbscan
import sklearn.cluster as skl_cluster
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


def get_cluster_meta(clusterer: hdbscan.HDBSCAN):
    """
    Finds cluster metadata:
        - cluster size in number of localizations in cluster
        - persistence of the cluster as detailed in the HDBSCAN API

    :param clusterer: the HDBSCAN object after fitting
    :return: pd.Dataframe with index cluster_id and size and persistence columns
    """

    cluster_meta = pd.DataFrame.from_dict(get_cluster_sizes(clusterer), orient="index")
    cluster_meta.columns = ["cluster_size"]

    # persistence = pd.DataFrame(clusterer.cluster_persistence_, columns=["cluster_persistence"]).reset_index().rename(columns={"index": "cluster_id"})
    persistence = pd.DataFrame(clusterer.cluster_persistence_, columns=["cluster_persistence"])

    cluster_meta = cluster_meta.join(other=persistence, how="left")

    return cluster_meta


def get_dbscan_clustering(orte):
    clusterer = skl_cluster.DBSCAN(10)
    clusterer.fit(orte[["x", "y"]])
    orte = orte.assign(dbscan_cl_id=clusterer.labels_)
    return orte, clusterer


def get_hdbscan_clustering(orte):
    clusterer = hdbscan.HDBSCAN(core_dist_n_jobs=1)
    clusterer.fit(orte[["x", "y"]])

    # tree_fig, tree_ax = plt.subplots(ncols=2, figsize=(10, 5), dpi=300)
    # tree_fig, tree_ax = plt.subplots(dpi=300)
    # clusterer.condensed_tree_.plot(axis=tree_ax[0])
    # clusterer.single_linkage_tree_.plot(axis=tree_ax, truncate_mode="lastp")
    # clusterer.single_linkage_tree_.plot(axis=tree_ax, truncate_mode="level")
    # tree_fig.tight_layout()
    # tree_fig.show()

    cluster_meta = get_cluster_meta(clusterer)
    orte = orte.assign(cluster_id=clusterer.labels_)
    orte = orte.assign(cluster_prob=clusterer.probabilities_)
    orte = orte.join(other=cluster_meta, on="cluster_id")
    return orte, clusterer, cluster_meta


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
