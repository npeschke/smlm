import pathlib as pl

import numpy as np

from smlm.smlm import analysis as analysis
from smlm.smlm import clustering as cluster


class Orte(object):
    def __init__(self, orte_path: pl.Path, rotation_angle: float = None):
        self._hdbscan_prefix = "hdbscan"
        self._dbscan_prefix = "dbscan"
        self.hdbscan_cl_id_col = f"{self._hdbscan_prefix}_cluster_id"
        self.dbscan_cl_id_col = f"{self._dbscan_prefix}_cluster_id"

        self.orte_path = orte_path
        self._rotation_angle = rotation_angle

        self.orte_df = analysis.load_orte(self.orte_path)
        self._rotate_orte()
        self.orte_df, self.vor = analysis.analyze_orte(self.orte_df)

        self.orte_df, self.hdbscan_clusterer, self.hdbscan_cluster_meta, self.hdbscan_vertices = cluster.get_hdbscan_clustering(self.orte_df,
                                                                                                                                self._hdbscan_prefix,
                                                                                                                                self.hdbscan_cl_id_col)
        self.orte_df, self.dbscan_clusterer, self.dbscan_cluster_meta, self.dbscan_vertices = cluster.get_dbscan_clustering(self.orte_df,
                                                                                                                            self._dbscan_prefix,
                                                                                                                            self.dbscan_cl_id_col)

        # cluster_density_df = cluster.get_cluster_density(self.orte_df, self._hdbscan_prefix, self._hdbscan_cl_id_col)
        #
        # self.hdbscan_cluster_meta = self.hdbscan_cluster_meta.join(cluster_density_df.set_index("cluster_id"), how="left")
        # self.orte_df = self.orte_df.merge(cluster_density_df, how="left", on="cluster_id")

    def get_named_cluster_meta(self, cluster_method: str = "hdbscan"):

        if cluster_method == "dbscan":
            named_cluster_meta = self.dbscan_cluster_meta
            col_name = self.dbscan_cl_id_col

        elif cluster_method == "hdbscan":
            named_cluster_meta = self.hdbscan_cluster_meta
            col_name = self.hdbscan_cl_id_col

        else:
            raise NotImplementedError

        named_cluster_meta = named_cluster_meta.assign(file=self.orte_path.name)
        named_cluster_meta = named_cluster_meta.reset_index().rename(columns={"index": col_name})

        return named_cluster_meta

    def _rotate_orte(self):
        if self._rotation_angle is not None:
            rot_rad = (self._rotation_angle / 180.0) * np.pi

            x_coords = self.orte_df.x
            y_coords = self.orte_df.y

            rot_x = (x_coords * np.cos(rot_rad)) - (y_coords * np.sin(rot_rad))
            rot_y = (x_coords * np.sin(rot_rad)) + (y_coords * np.cos(rot_rad))

            rot_shift_x = self._shift(rot_x)
            rot_shift_y = self._shift(rot_y)

            self.orte_df.x = rot_shift_x
            self.orte_df.y = rot_shift_y

    @staticmethod
    def _shift(coordinates, offset: float = 50):
        if min(coordinates) < 0:
            return coordinates + abs(min(coordinates)) + offset

        else:
            return coordinates


if __name__ == '__main__':
    localization_path = pl.Path("../../data/cut_cells/H2B_mCherry/2020_Jun_30_Stauro_LCI_SMLM/1_0/merge_filter/cell_5_thre_1_0_merge_filter.csv")
    test = Orte(localization_path, 180)

    print("completed")
