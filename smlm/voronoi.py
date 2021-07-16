import collections as coll
import pathlib as pl

import numpy as np
import pandas as pd
import scipy.io as spio
import scipy.spatial as spat


def polygon_density(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_voronoi(coord_x, coord_y):
    points = list(zip(coord_x, coord_y))
    # Get Voronoi points
    vor = spat.Voronoi(points)

    x_patch, y_patch, _ = get_full_patches(vor)

    # This code that gets the multi lines that run indefinitely
    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)

    line_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            line_segments.append([(vor.vertices[i, 0], vor.vertices[i, 1]),
                                  (far_point[0], far_point[1])])

    x_vor_ls, y_vor_ls = [], []
    for region in line_segments:

        x1, y1 = [], []
        for i in region:
            x1.append(i[0])
            y1.append(i[1])

        x_vor_ls.append(np.array(x1))
        y_vor_ls.append(np.array(y1))

    # Removing patches that were added multiple times.
    patches = pd.DataFrame(data={"x_patch": list(coll.OrderedDict((tuple(x), x) for x in x_patch).values()),
                                 "y_patch": list(coll.OrderedDict((tuple(x), x) for x in y_patch).values())})
    patches = patches.assign(n_vertices=patches.x_patch.apply(len))

    # x_patch = list(coll.OrderedDict((tuple(x), x) for x in x_patch).values())
    # y_patch = list(coll.OrderedDict((tuple(x), x) for x in y_patch).values())
    x_vor_ls = list(coll.OrderedDict((tuple(x), x) for x in x_vor_ls).values())
    y_vor_ls = list(coll.OrderedDict((tuple(x), x) for x in y_vor_ls).values())

    # return x_patch, y_patch, x_vor_ls, y_vor_ls
    return patches, x_vor_ls, y_vor_ls


def get_full_patches(vor):
    region_idxs = []
    x_patch, y_patch = [], []
    x1_patch, y1_patch = [], []
    # The Voronoi has 2 parts. The actual patches and the unbounded lines (that run  #indefinitely)
    for region_idx, region in enumerate(vor.regions):
        if -1 not in region and len(region) >= 3:
            x1_patch, y1_patch = [], []
            for i in region:
                x1_patch.append(vor.vertices[i][0])
                y1_patch.append(vor.vertices[i][1])

            x_patch.append(np.array(x1_patch))
            y_patch.append(np.array(y1_patch))
            region_idxs.append(region_idx)
    return x_patch, y_patch, region_idxs


def get_voronoi_df(orte_file_path: pl.Path):
    if orte_file_path.suffix == ".mat":
        orte = spio.loadmat(str(orte_file_path))["Orte"]
        df, x_vor_ls, y_vor_ls = get_voronoi(orte[:, 1], orte[:, 2])
    elif orte_file_path.suffix == ".csv":
        with orte_file_path.open("r") as f:
            orte = pd.read_csv(f)

        df, x_vor_ls, y_vor_ls = get_voronoi(orte["x [nm]"], orte["y [nm]"])
    else:
        raise NotImplementedError(f"Orte file must be of type .mat or type .csv - You specified {orte_file_path.suffix}")

    df = df.assign(sample_name=orte_file_path.name)

    if "Control" in orte_file_path.name:
        df = df.assign(is_control=True)
    else:
        df = df.assign(is_control=False)

    # df = df.assign(density=lambda x: polygon_density(x.x_patch, x.y_patch))
    df["area"] = df.apply(lambda x: 0.5 * np.abs(np.dot(x["x_patch"], np.roll(x["y_patch"], 1)) -
                                                 np.dot(x["y_patch"], np.roll(x["x_patch"], 1))), axis=1)
    df["density"] = 1 / df["area"]
    return df


def get_voronoi_density(points):
    vor = spat.Voronoi(points)

    # noinspection PyTypeChecker
    x_patch, y_patch, region_idxs = get_full_patches(vor)
    patches = pd.DataFrame({"x_patch": x_patch, "y_patch": y_patch}, index=region_idxs)

    # noinspection PyTypeChecker
    patches["area"] = patches.apply(lambda row: 0.5 * np.abs(np.dot(row["x_patch"], np.roll(row["y_patch"], 1)) - np.dot(row["y_patch"], np.roll(row["x_patch"], 1))), axis=1)
    patches = patches.assign(density=1 / patches.area)

    # noinspection PyUnresolvedReferences
    return patches.reindex(vor.point_region).density.reset_index(drop=True), vor

    # 0.5 * np.abs(np.dot(x_patch, np.roll(y_patch, 1)) - np.dot(y_patch, np.roll(x_patch, 1)))


if __name__ == '__main__':
    # orte_path = pl.Path("data/revised/sytox_orange/2020_25_06_Stauro/3_merged_filtered/Stauro_cell_1_recon_1_0_thre_merge_filter.mat")
    # orte = spio.loadmat(orte_path)["Orte"]
    #
    # coordinates = orte[:, 1:3]

    # treated_orte_dir = pl.Path("data/revised/sytox_orange/2020_25_06_Stauro/3_merged_filtered/")
    # control_orte_dir = pl.Path("data/revised/sytox_orange/2020_25_06_Control/3_merged_filtered/")
    #
    # orte_list = list(treated_orte_dir.glob("*.mat"))
    # orte_list.extend(list(control_orte_dir.glob("*.mat")))

    treated_orte_dir = pl.Path("data/cut_cells/Sytox_Orange/2020_25_06_Stauro/1_0/merge_filter")
    control_orte_dir = pl.Path("data/cut_cells/Sytox_Orange/2020_26_06_Control/1_0/merge_filter")

    orte_list = list(treated_orte_dir.glob("*.csv"))
    orte_list.extend(list(control_orte_dir.glob("*.csv")))

    patches_df = pd.DataFrame()

    patches_list = []

    patches_list.append(get_voronoi_df(orte_list[0]))

    # with mp.Pool(processes=min(len(orte_list), mp.cpu_count())) as p:
    # with mp.Pool(processes=1) as p:
    #     patches_list = p.map(get_voronoi_df, orte_list, chunksize=1)

    for patch_df in patches_list:
        patches_df = patches_df.append(patch_df)


    # for orte_file_path in orte_dir.glob("*.mat"):
    #     get_voronoi_df()
    #     patches_df.append(patch_df)

    # # x_patch, y_patch, x_vor_ls, y_vor_ls = get_voronoi(orte[:, 1], orte[:, 2])
    # patch_df, x_vor_ls, y_vor_ls = get_voronoi(orte[:, 1], orte[:, 2])
    # patch_df = patch_df.assign(sample=orte_path.name)
    # # patch_df = patch_df.assign(density=lambda x: polygon_density(x.x_patch, x.y_patch))
    # patch_df["area"] = patch_df.apply(lambda x: 0.5 * np.abs(np.dot(x["x_patch"], np.roll(x["y_patch"], 1)) - np.dot(x["y_patch"], np.roll(x["x_patch"], 1))), axis=1)
    # patch_df["density"] = 1/patch_df["area"]

    # density = []
    # for idx in range(len(x_patch)):
    #     density.append(polygon_density(x_patch[idx], y_patch[idx]))

    # density_hist_fig, density_hist_ax = plt.subplots()
    # plot_density = patch_df.loc[(patch_df.density > np.quantile(patch_df.density, 0.5)) & (patch_df.density < np.inf)].density
    # sns.distplot(plot_density, ax=density_hist_ax)
    # density_hist_fig.show()

    # density_pair_fig, density_pair_ax = plt.subplots()
    # pairplot_df = patches_df.loc[patches_df.density < np.inf].pivot(columns="sample", values="density")
    # pairplot = sns.pairplot(pairplot_df, height=5, aspect=1)
    # pairplot.fig.tight_layout()
    # pairplot.fig.show()

    # density_comp_fig, density_comp_ax = plt.subplots()
    





    # source_vor = bmods.ColumnDataSource(dict(xs=patch_df.x_patch, ys=patch_df.y_patch, density=patch_df.density))
    # source_vor_ls = bmods.ColumnDataSource(dict(xs=x_vor_ls, ys=y_vor_ls))
    #
    # bkplt.output_file("voronoi.html", mode="inline")
    #
    # plot = bkplt.figure(plot_width=1000, plot_height=1000, match_aspect=True)
    #
    # mapper = bktansf.linear_cmap(field_name="density", palette="Plasma256", low=np.quantile(patch_df.density, 0.93), high=np.quantile(patch_df.density, 0.01))
    #
    # # density = polygon_density(x_patch, y_patch)
    #
    # # plot patches for the Voronoi
    # glyph_vor = plot.patches('xs', 'ys', source=source_vor, alpha=1, line_width=0, fill_color=mapper, line_color='black')
    #
    # # # plot the boundary lines
    # glyph_ls = plot.multi_line('xs', 'ys', source=source_vor_ls, alpha=1, line_width=1, line_color='black')
    #
    # # color_bar = bmods.ColorBar(color_mapper=mapper['transform'], width=8, location=(0, 0))
    # # color_bar = bmods.ColorBar(color_mapper=mapper["transform"], ticker=bmods.LogTicker(), location=(0, 0))
    # color_bar = bmods.ColorBar(color_mapper=mapper["transform"], location=(0, 0))
    # plot.add_layout(color_bar, 'right')
    #
    # bkplt.show(plot)

# fig.show()
