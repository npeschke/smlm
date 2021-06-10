# import matplotlib.colors as mpl_col
import seaborn as sns
# import cmasher

# COLORS
# SNS_BLUE = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
SNS_BLUE = "#4c72b0"

# APOPTOSIS_RED = (1.0, 0.0, 0.0)
APOPTOSIS_RED = "#ff0000"

# cmap = cmasher.get_sub_cmap(sns.color_palette(as_cmap=True), 0, 1,)

# CMAPS
# BIN_CMAP = mpl_col.LinearSegmentedColormap.from_list(name="apoptosis", colors=[SNS_BLUE, APOPTOSIS_RED], N=2)
BIN_CMAP = sns.color_palette([SNS_BLUE, APOPTOSIS_RED])
