import seaborn as sns
import cmasher

# COLORS
# SNS_BLUE = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
SNS_BLUE = "#4c72b0"

# APOPTOSIS_RED = (1.0, 0.0, 0.0)
APOPTOSIS_RED = "#ff0000"

# CMAPS
BIN_CMAP = sns.color_palette([SNS_BLUE, APOPTOSIS_RED])

SEQ_CMAP = cmasher.chroma

# Figure Parameters
FIG_BASE_SIZE = 6.5

# Limits
DENSITY_LIM = (1e-4, 1e0)
RADIUS_LIM = (0, 8000)

CBAR_MIN = 1e-5
CBAR_MAX = 3e-4
