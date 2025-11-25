import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from ..helper_functions import goto_folder, get_view
from .radvis_functions import radvis_temperature, radvis_column_density, radvis_velocities

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
    bar_length = int(25 + 25 * (plot_size // 250))
    end_point = 4/5 * plot_size//2

    ax.hlines(-375/500 * plot_size//2, end_point - bar_length, end_point, color=color, linestyles="solid", linewidths=3)
    ax.text(end_point - bar_length/2, -375/500 * plot_size//2 -plot_size//2*20/500, str(bar_length)+" AU", ha="center", va='top', color=color, fontsize=20)

def plot_radvis_temperature(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, verbose=1, vmin=None, vmax=None, ax=None, xlim=None, ylim=None, save=False):
    path = goto_folder(isink, iout)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)
    fname = "temperature-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)

    temperature = radvis_temperature(isink=isink, iout=iout, resolution=resolution, width=width, dz=dz, inclination=inclination, rotangle=rotangle, verbose=verbose)

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

    X, Y = [np.linspace(-width //2, width // 2, resolution) for _ in range(2)]

    im = ax.pcolormesh(X, Y, temperature, cmap="viridis", norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(im, ax=ax, pad=0, orientation="horizontal", location="top", extend=extend)
    cbar.set_label("Density-weighted Temperature [K]", size=20)
    cbar.ax.tick_params(labelsize=14)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    _stylize_plot(ax)

    if save:
        # Save the figure
        print("Outputting image plot as .png")
        if not ax: plt.savefig(path+"/saved_plots/TemperatureMap/"+fname+".png", bbox_inches="tight")

def plot_radvis_column_density(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, verbose=1, vmin=None, vmax=None, ax=None, xlim=None, ylim=None, save=False):
    path = goto_folder(isink, iout)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)
    fname = "coldens-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)

    coldens = radvis_column_density(isink=isink, iout=iout, resolution=resolution, width=width, dz=dz, inclination=inclination, rotangle=rotangle, verbose=verbose)
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

    X, Y = [np.linspace(-width //2, width // 2, resolution) for _ in range(2)]

    im = ax.pcolormesh(X, Y, log_coldens, cmap="cividis", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, pad=0, orientation="horizontal", location="top", extend=extend)
    cbar.set_label("Column Density [$\\log(\\Sigma / \\mathrm{g\\;cm^{-2}})$]", size=20)
    cbar.ax.tick_params(labelsize=14)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    _stylize_plot(ax)

    if save:
        # Save the figure
        print("Outputting image plot as .png")
        if not ax: plt.savefig(path+"/saved_plots/ColumnDensity/"+fname+".png", bbox_inches="tight")

def plot_radvis_velocities(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, verbose=1, vmin=None, vmax=None, ax=None, xlim=None, ylim=None, save=False):
    path = goto_folder(isink, iout)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)
    fname = "velocities-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)

    velocities = radvis_velocities(isink=isink, iout=iout, resolution=resolution, width=width, dz=dz, inclination=inclination, rotangle=rotangle, verbose=verbose)

    if ax is None: 
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    X, Y = [np.linspace(-width //2, width // 2, resolution) for _ in range(2)]

    if vmin is None:
        vmin = velocities.min()
    if vmax is None:
        vmax = velocities.max()

    if vmax < velocities.max() and vmin > velocities.min():
        extend = "both"
    elif vmin > velocities.min():
        extend = "min"
    elif vmax < velocities.max():
        extend = "max"
    else:
        extend = "neither"

    im = ax.pcolormesh(X, Y, velocities, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, pad=0, orientation="horizontal", location="top", extend=extend)
    cbar.set_label("Line-of-sight velocity [km/s]", size=20)
    cbar.ax.tick_params(labelsize=14)
    
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    _stylize_plot(ax)

    if save:
        # Save the figure
        print("Outputting image plot as .png")
        if not ax: plt.savefig(path+"/saved_plots/ColumnDensity/"+fname+".png", bbox_inches="tight")