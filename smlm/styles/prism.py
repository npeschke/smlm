import matplotlib as mpl
import seaborn as sns


def prism_style():
    sns.set_style(None)
    mpl.rcParams["font.size"] = 14.0
    # mpl.rcParams["font.family"] = "serif"
    # sns.set(font="Amiri")
    mpl.rcParams["font.sans-serif"] = "Arial"
    # mpl.rcParams["font.family"] = "Arial"
    # sns.set_style(rc={'font.family': 'sans-serif', "font.sans-serif": "Arial"})
    # sns.set_style(None, {"font.sans-serif": "Arial",
    #                         # "mathtext.default": "regular",
    #                         # "mathtext.it": "Arial",
    #                         # "mathtext.rm": "Arial",
    #                         # "mathtext.sf": "Arial",
    #                         # "mathtext.cal": "Arial",
    #                         # "mathtext.tt": "Arial",
    #                         # "mathtext.fallback": None,
    #
    #                         "axes.spines.top": False,
    #                         "axes.spines.right": False,
    #
    #                         })

    mpl.rcParams["figure.facecolor"] = "white"

    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False

    _linewidth = 2.5
    _size_multiplier = 3

    mpl.rcParams["axes.linewidth"] = _linewidth

    mpl.rcParams["xtick.major.width"] = _linewidth
    mpl.rcParams["ytick.major.width"] = _linewidth

    mpl.rcParams["xtick.major.size"] = _size_multiplier * _linewidth
    mpl.rcParams["ytick.major.size"] = _size_multiplier * _linewidth
