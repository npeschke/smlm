import seaborn as sns
import cmasher

# sns.set_style("darkgrid")
# sns.set_theme(font="sans-serif")
# mpl.rcParams["font.size"] = 14.0


# COLORS
# SNS_BLUE = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
SNS_BLUE = "#4c72b0"

# APOPTOSIS_RED = (1.0, 0.0, 0.0)
# APOPTOSIS_RED = "#ff0000"
APOPTOSIS_RED = "#c44e52"

STAGE1_BLUE = "#1f77b4"
# STAGE1_BLUE = "#0173B2"

STAGE2_MAGENTA = "#ff3dc8"
# STAGE2_MAGENTA = "#ff3dc8"


STAGE3_CYAN = "#1cd5ea"
STAGE4_RED = "#d62728"
STAGE5_PURPLE = "#9467bd"
# STAGE6_BROWN = "#654321"
# STAGE_COLORS = [STAGE1_BLUE, STAGE2_MAGENTA, STAGE3_CYAN, STAGE4_RED, STAGE5_PURPLE]

# STAGE_COLORS = ["#0173B2", "#CC78BC", "#029e73", "#d62728", "#ca9161"]

# ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9']


# CMAPS
BIN_CMAP = sns.color_palette([SNS_BLUE, APOPTOSIS_RED])
STAGE_CMAP = sns.color_palette(n_colors=5)
STAGE_COLORS = STAGE_CMAP.as_hex()

SEQ_CMAP = cmasher.chroma
CLUSTER_POLY_CMAP = cmasher.ember

# Figure Parameters
FIG_BASE_SIZE = 6.5
FIG_TYPE = "svg"
DPI = 300

# Limits
# DENSITY_LIM = (1e-4, 1e0)
DENSITY_LIM = (-10, -5)
RADIUS_LIM = (0, 8000)

# CBAR_MIN = 1e-5
CBAR_MIN = 0.00001
# CBAR_MAX = 3e-4
CBAR_MAX = 0.001


# Statistics
HIST_STAT = "density"

# Labels
STAGE_METHOD = "manual_5_stage"
TISSUE_HUE = "aCasp3_signal"
LOG_NORM_DENS_COL = "log_norm_density"
LOG_NORM_DENS_LABEL = r"$\log_{Voronoi\,Density}\;\left[\log\left(\frac{1}{nm^2}\right)\right]$"
PDF = "Probability Density Function"




