import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from ..helper_functions import goto_folder, get_view
from .radvis_functions import radvis_temperature, radvis_column_density

plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def _stylize_plot(ax, color="white"):
    # Remove axes
    ax.xaxis.label.set_visible(False); ax.yaxis.label.set_visible(False)
    ax.set_yticklabels([]); ax.set_yticks([])
    ax.set_xticklabels([]); ax.set_xticks([])
    
    # Create scale bar
    # We should normalize the distances to the edges
    plot_size = np.abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    bar_length = int(25 * (plot_size // 250))
    bar_length_normalized = bar_length / (np.abs(ax.get_xlim()[1] - ax.get_xlim()[0]))

    ax.hlines(0.07, 0.91-bar_length_normalized/2, 0.91+bar_length_normalized/2, color="white", linestyles="solid", linewidths=5, transform=ax.transAxes)
    ax.text(0.91, 0.05, str(bar_length)+" au", ha="center", va='top', color="white", transform=ax.transAxes, fontsize=30)

def plot_radvis_temperature(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, convolve=False, beam_kernel=None, verbose=1, lognorm=False, vmin=None, vmax=None, ax=None, xlim=None, ylim=None, save=False):
    path = goto_folder(isink, iout)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)
    fname = "temperature-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)

    temperature = radvis_temperature(isink=isink, iout=iout, resolution=resolution, width=width, dz=dz, inclination=inclination, rotangle=rotangle, convolve=convolve, beam_kernel=beam_kernel, verbose=verbose)

    if ax is None: 
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    if vmin is None:
        vmin = temperature.min()
    if vmax is None:
        vmax = temperature.max()

    if vmax < temperature.max() and vmin > temperature.min():
        extend = "both"
    elif vmin > temperature.min():
        extend = "min"
    elif vmax < temperature.max():
        extend = "max"
    else:
        extend = "neither"

    if convolve:
        cb_label = "Convolved Density-weighted Temperature [K]"
    else:
        cb_label = "Density-weighted Temperature [K]"


    X, Y = [np.linspace(-width //2, width // 2, resolution) for _ in range(2)]
    if convolve:
        im = ax.pcolormesh(X, Y, temperature, cmap="viridis")
    else:
        if lognorm:
            im = ax.pcolormesh(X, Y, temperature, cmap="viridis", norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        else:
            im = ax.pcolormesh(X, Y, temperature, cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, pad=0, orientation="horizontal", location="top", extend=extend)
    cbar.set_label(cb_label, size=30)
    cbar.ax.tick_params(labelsize=25)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    _stylize_plot(ax)

    if save:
        # Save the figure
        print("Outputting image plot as .png")
        plt.savefig(path+"/saved_plots/TemperatureMap/"+fname+".png", bbox_inches="tight")

def plot_radvis_column_density(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, convolve=False, beam_kernel=None, verbose=1, vmin=None, vmax=None, ax=None, xlim=None, ylim=None, save=False):
    path = goto_folder(isink, iout)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)
    fname = "coldens-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)

    coldens = radvis_column_density(isink=isink, iout=iout, resolution=resolution, width=width, dz=dz, inclination=inclination, rotangle=rotangle, convolve=convolve, beam_kernel=beam_kernel, verbose=verbose)
    log_coldens = np.log10(coldens)

    if ax is None: 
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    if vmin is None:
        vmin = log_coldens.min()
    if vmax is None:
        vmax = log_coldens.max()

    if vmax < log_coldens.max() and vmin > log_coldens.min():
        extend = "both"
    elif vmin > log_coldens.min():
        extend = "min"
    elif vmax < log_coldens.max():
        extend = "max"
    else:
        extend = "neither"

    if convolve:
        cb_label = "Convolved Column Density [$\\log(\\Sigma / \\mathrm{g\\;cm^{-2}})$]"
    else:
        cb_label = "Column Density [$\\log(\\Sigma / \\mathrm{g\\;cm^{-2}})$]"

    X, Y = [np.linspace(-width //2, width // 2, resolution) for _ in range(2)]

    im = ax.pcolormesh(X, Y, log_coldens, cmap="cividis", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, pad=0, orientation="horizontal", location="top", extend=extend)
    cbar.set_label(cb_label, size=30)
    cbar.ax.tick_params(labelsize=25)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    _stylize_plot(ax)

    if save:
        # Save the figure
        print("Outputting image plot as .png")
        plt.savefig(path+"/saved_plots/ColumnDensity/"+fname+".png", bbox_inches="tight")

#def plot_radvis_velocities(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, verbose=1, vmin=None, vmax=None, ax=None, xlim=None, ylim=None, save=False):
#    path = goto_folder(isink, iout)
#    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)
#    fname = "velocities-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)
#
#    velocities = radvis_velocities(isink=isink, iout=iout, resolution=resolution, width=width, dz=dz, inclination=inclination, rotangle=rotangle, verbose=verbose)
#
#    if ax is None: 
#        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
#
#    X, Y = [np.linspace(-width //2, width // 2, resolution) for _ in range(2)]
#
#    if vmin is None:
#        vmin = velocities.min()
#    if vmax is None:
#        vmax = velocities.max()
#
#    if vmax < velocities.max() and vmin > velocities.min():
#        extend = "both"
#    elif vmin > velocities.min():
#        extend = "min"
#    elif vmax < velocities.max():
#        extend = "max"
#    else:
#        extend = "neither"
#
#    im = ax.pcolormesh(X, Y, velocities, cmap="RdBu_r", vmin=vmin, vmax=vmax)
#    cbar = plt.colorbar(im, ax=ax, pad=0, orientation="horizontal", location="top", extend=extend)
#    cbar.set_label("Line-of-sight velocity [km/s]", size=20)
#    cbar.ax.tick_params(labelsize=14)
#    
#    if xlim is not None:
#        ax.set_xlim(xlim[0], xlim[1])
#    if ylim is not None:
#        ax.set_ylim(ylim[0], ylim[1])
#
#    _stylize_plot(ax)
#
#    if save:
#        # Save the figure
#        print("Outputting image plot as .png")
#        plt.savefig(path+"/saved_plots/ColumnDensity/"+fname+".png", bbox_inches="tight")