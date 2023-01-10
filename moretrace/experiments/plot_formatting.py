import matplotlib.pyplot as plt
from cycler import cycler


def preformat_plots():
    # Set font size
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rcParams.update({'font.size': 22})

    # Set color cycle to
    # Midnight Blue, Belize Hole, Nephritis, Pomegranate
    # (see https://flatuicolors.com/palette/defo for all colors)
    color_cycler = cycler('color', ['#2c3e50', '#2980b9', '#27ae60', '#c0392b'])
    plt.rcParams.update({'axes.prop_cycle': color_cycler})


def postformat_plots(aspect=1.0, legend=True):
    # Make axes square
    ax = plt.gca()
    ax.set_aspect(1.0 / (aspect * ax.get_data_ratio()))
    height_in = 6
    width_in = aspect * height_in
    plt.gcf().set_size_inches(width_in, height_in)

    if legend:
        plt.legend(loc='best', frameon=False, framealpha=0.0, fontsize=16)

    # Turn off top/right borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout(pad=0.02)
