import pathlib as pl

import pandas as pd
import scipy.spatial as spat

from . import analysis as analysis
from . import clustering as cluster


class Orte(object):
    def __init__(self, orte_path: pl.Path):
        self.orte_path = orte_path
        self.orte_df = analysis.load_orte(self.orte_path)
        self.orte_df = analysis.analyze_orte(self.orte_df)

        self.orte_df, self.clusterer, self.cluster_meta = cluster.get_hdbscan_clustering(self.orte_df)

        cluster_density_df = self._get_cluster_density()

        self.cluster_meta = self.cluster_meta.join(cluster_density_df.set_index("cluster_id"), how="left")
        self.orte_df = self.orte_df.set_index("cluster_id").join(cluster_density_df.set_index("cluster_id"), how="left")

    def get_named_cluster_meta(self):
        named_cluster_meta = self.cluster_meta.assign(file=self.orte_path.name)
        named_cluster_meta = named_cluster_meta.reset_index().rename(columns={"index": "cluster_id"})
        return named_cluster_meta

    def _get_cluster_density(self):
        # cluster_density = pd.DataFrame(columns=["cluster_id", "cluster_area", "cluster_density"])
        cluster_density = []
        for cluster_id, cluster_df in self.orte_df.groupby("cluster_id"):
            hull = spat.ConvexHull(cluster_df[["x", "y"]])
            # spat.convex_hull_plot_2d(hull)

            cluster_density.append([cluster_id, hull.volume, len(cluster_df) / hull.volume])

        return pd.DataFrame(cluster_density, columns=["cluster_id", "cluster_area", "cluster_density"])


if __name__ == '__main__':
    localization_path = pl.Path("../data/cut_cells/H2B_mCherry/2020_Nov13_SMLM_from_12th/1_0/merge_filter/cell3_s04_recon_1_0thre_merg_filt_cut.csv")
    test = Orte(localization_path)

