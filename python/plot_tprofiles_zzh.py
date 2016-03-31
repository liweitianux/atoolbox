# -*- coding: utf-8 -*-
#
# Weitian LI
# 2015-09-11
#

"""
Plot a list of *temperature profiles* in a grid of subplots with Matplotlib.
"""

import matplotlib.pyplot as plt


def plot_tprofiles(tplist, nrows, ncols,
        xlim=None, ylim=None, logx=False, logy=False,
        xlab="", ylab="", title=""):
    """
    Plot a list of *temperature profiles* in a grid of subplots of size
    nrow x ncol. Each subplot is related to a temperature profile.
    All the subplots share the same X and Y axes.
    The order is by row.

    The tplist is a list of dictionaries, each of which contains all the
    necessary data to make the subplot.

    The dictionary consists of the following components:
    tpdat = {
        "name": "NAME",
        "radius": [[radius points], [radius errors]],
        "temperature": [[temperature points], [temperature errors]],
        "radius_model": [radus points of the fitted model],
        "temperature_model": [
            [fitted model value],
            [lower bounds given by the model],
            [upper bounds given by the model]
        ]
    }

    Arguments:
        tplist - a list of dictionaries containing the data of each
                 temperature profile.
                 Note that the length of this list should equal to nrows*ncols.
        nrows  - number of rows of the subplots
        ncols  - number of columns of the subplots
        xlim   - limits of the X axis
        ylim   - limits of the Y axis
        logx   - whether to set the log scale for X axis
        logy   - whether to set the log scale for Y axis
        xlab   - label for the X axis
        ylab   - label for the Y axis
        title  - title for the whole plot
    """
    assert len(tplist) == nrows*ncols, "tplist length != nrows*ncols"
    # All subplots share both X and Y axes.
    fig, axarr = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    # Set title for the whole plot.
    if title != "":
        fig.suptitle(title)
    # Set xlab and ylab for each subplot
    if xlab != "":
        for ax in axarr[-1, :]:
            ax.set_xlabel(xlab)
    if ylab != "":
        for ax in axarr[:, 0]:
            ax.set_ylabel(ylab)
    for ax in axarr.reshape(-1):
        # Set xlim and ylim.
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        # Set xscale and yscale.
        if logx:
            ax.set_xscale("log", nonposx="clip")
        if logy:
            ax.set_yscale("log", nonposy="clip")
    # Decrease the spacing between the subplots and suptitle
    fig.subplots_adjust(top=0.94)
    # Eleminate the spaces between each row and column.
    fig.subplots_adjust(hspace=0, wspace=0)
    # Hide X ticks for all subplots but the bottom row.
    plt.setp([ax.get_xticklabels() for ax in axarr[:-1, :].reshape(-1)],
            visible=False)
    # Hide Y ticks for all subplots but the left column.
    plt.setp([ax.get_yticklabels() for ax in axarr[:, 1:].reshape(-1)],
            visible=False)
    # Plot each temperature profile in the tplist
    for i, ax in zip(range(len(tplist)), axarr.reshape(-1)):
        tpdat = tplist[i]
        # Add text to display the name.
        # The text is placed at (0.95, 0.95), i.e., the top-right corner,
        # with respect to this subplot, and the top-right part of the text
        # is aligned to the above position.
        ax_pois = ax.get_position()
        ax.text(0.95, 0.95, tpdat["name"],
                verticalalignment="top", horizontalalignment="right",
                transform=ax.transAxes, color="black", fontsize=10)
        # Plot data points
        if isinstance(tpdat["radius"][0], list) and \
                len(tpdat["radius"]) == 2 and \
                isinstance(tpdat["temperature"][0], list) and \
                len(tpdat["temperature"]) == 2:
            # Data points have symmetric errorbar
            ax.errorbar(tpdat["radius"][0], tpdat["temperature"][0],
                    xerr=tpdat["radius"][1], yerr=tpdat["temperature"][1],
                    color="black", linewidth=1.5, linestyle="None")
        else:
            ax.plot(tpdat["radius"], tpdat["temperature"],
                    color="black", linewidth=1.5, linestyle="None")
        # Plot model line and bounds band
        if isinstance(tpdat["temperature_model"][0], list) and \
                len(tpdat["temperature_model"]) == 3:
            # Model data have bounds
            ax.plot(tpdat["radius_model"], tpdat["temperature_model"][0],
                    color="blue", linewidth=1.0)
            # Plot model bounds band
            ax.fill_between(tpdat["radius_model"],
                    y1=tpdat["temperature_model"][1],
                    y2=tpdat["temperature_model"][2],
                    color="gray", alpha=0.5)
        else:
            ax.plot(tpdat["radius_model"], tpdat["temperature_model"],
                    color="blue", linewidth=1.5)
    return (fig, axarr)

#  vim: set ts=4 sw=4 tw=0 fenc=utf-8 ft=python: #
