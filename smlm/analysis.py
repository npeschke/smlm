import pathlib as pl
import pandas as pd
import numpy as np

# import multiprocessing as mp

import smlm.smlm.voronoi as vor
import smlm.smlm.polar as pol
# from . import voronoi as vor
# from . import polar as pol


def get_result_dir(orte_path):
    result_dir = pl.Path("results").joinpath("/".join(orte_path.parts[1:-1]))
    if not result_dir.exists():
        result_dir.mkdir(parents=True)

    return result_dir


def analyze_orte(orte: pd.DataFrame):
    # orte = load_orte(orte_path)
    points = np.array([orte.x, orte.y]).T
    voronoi_density, voronoi = vor.get_voronoi_density(points)
    norm_voronoi_density = voronoi_density / len(voronoi_density)
    log_voronoi_density = np.log10(voronoi_density)
    log_norm_voronoi_density = np.log10(norm_voronoi_density)
    center_of_mass = pol.center_of_mass(points)
    polar_points = pol.to_polar(points, new_center=center_of_mass, angle_offset=0)
    orte.insert(5, "r", polar_points[:, 0])
    orte.insert(6, "phi", polar_points[:, 1])
    orte.insert(7, "density", voronoi_density)
    orte.insert(8, "norm_density", norm_voronoi_density)
    orte.insert(9, "log_density", log_voronoi_density)
    orte.insert(10, "log_norm_density", log_norm_voronoi_density)

    return orte.dropna(), voronoi


def load_orte(orte_path: pl.Path):
    with orte_path.open("r") as f:
        orte = pd.read_csv(f, header=0, names=["id", "frame", "x", "y", "sigma", "intensity",
                                               "offset", "bkgstd", "chi2", "uncertainty",
                                               "detections"])
    orte = orte.assign(file=orte_path.name)
    orte = orte.assign(path=orte_path)
    if "Control" in str(orte_path):
        orte = orte.assign(control=True)
    else:
        orte = orte.assign(control=False)
    return orte


if __name__ == '__main__':
    # test_path = pl.Path("../data/cut_cells/Sytox_Orange/2020_26_06_Control/1_0/merge_filter/Control_cell_4_recon_1_0_thre_merge_filter_cut.csv")
    # analyze_orte(test_path)

    # labelling = "Sytox_Orange"
    labelling = "H2B_mCherry"
    thunderstorm_param_1 = "1_0"
    thunderstorm_param_2 = "merge_filter"

    data_dir = pl.Path("../../data/cut_cells")

    orte_paths = [path for path in data_dir.glob(f"{labelling}/*/{thunderstorm_param_1}/{thunderstorm_param_2}/*.csv")]
    labelling_csv = pl.Path("data/cut_cells/stage_labelling.csv")

    orte_dfs = []

    # with mp.Pool(processes=1) as p:
        # with mp.Pool(processes=min(len(orte_paths), mp.cpu_count())) as p:
        # orte_dfs = p.map(analyze_orte, orte_paths[::-1], chunksize=1)

    for this_orte_path in orte_paths:
        orte_dfs.append(analyze_orte(load_orte(this_orte_path)))

    print(orte_dfs)
