import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from ..helper_functions import goto_folder, get_view
from .radvis_functions import radvis_temperature, radvis_column_density, radvis_velocities

plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def _stylize_plot(ax, width, color="white"):
    # Remove axes
    ax.xaxis.label.set_visible(False); ax.yaxis.label.set_visible(False)
    ax.set_yticklabels([]); ax.set_yticks([])
    ax.set_xticklabels([]); ax.set_xticks([])
    
    # Create scale bar
    # We should normalize the distances to the edges
    bar_length = int(25 + 25 * (width // 250))
    end_point = 4/5 * width//2

    ax.hlines(-375/500 * width//2, end_point - bar_length, end_point, color=color, linestyles="solid", linewidths=3)
    ax.text(end_point - bar_length/2, -375/500 * width//2 -width//2*20/500, str(bar_length)+" AU", ha="center", va='top', color=color, fontsize=20, weight="heavy")

def plot_radvis_temperature(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, verbose=1, vmin=None, vmax=None, ax=None, save=False):
    path = goto_folder(isink, iout)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)
    fname = "temperature-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)

    temperature = radvis_temperature(isink, iout, resolution, width, dz, inclination=None, rotangle=None, verbose=1)

    if ax is None: 
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    if vmin is None:
        vmin = temperature.min()

    X, Y = [np.linspace(-width //2, width // 2, resolution) for _ in range(2)]

    im = ax.pcolormesh(X, Y, temperature, cmap="viridis", norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar(im, ax=ax, label="Density-weighted Temperature [K]", location="top", orientation="horizontal", pad=0)
    _stylize_plot(ax, width)

    if save:
        # Save the figure
        print("Outputting image plot as .png")
        if not ax: plt.savefig(path+"/saved_plots/TemperatureMap/"+fname+".png", bbox_inches="tight")

def plot_radvis_column_density(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, verbose=1, vmin=None, vmax=None, ax=None, save=False):
    path = goto_folder(isink, iout)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)
    fname = "coldens-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)

    coldens = radvis_column_density(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, verbose=1)

    if ax is None: 
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    X, Y = [np.linspace(-width //2, width // 2, resolution) for _ in range(2)]

    im = ax.pcolormesh(X, Y, np.log10(coldens), cmap="cividis", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Column Density [$\\log(\\Sigma / \\mathrm{g\\;cm^{-2}})$]", location="top", orientation="horizontal", pad=0)
    _stylize_plot(ax, width)

    if save:
        # Save the figure
        print("Outputting image plot as .png")
        if not ax: plt.savefig(path+"/saved_plots/ColumnDensity/"+fname+".png", bbox_inches="tight")

def plot_radvis_velocities(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, verbose=1, vmin=None, vmax=None, ax=None, save=False):
    path = goto_folder(isink, iout)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)
    fname = "velocities-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)

    velocities = radvis_velocities(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, verbose=1)

    if ax is None: 
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    X, Y = [np.linspace(-width //2, width // 2, resolution) for _ in range(2)]

    im = ax.pcolormesh(X, Y, velocities, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Line-of-sight velocity [km/s]", location="top", orientation="horizontal", pad=0)
    _stylize_plot(ax, width)

    if save:
        # Save the figure
        print("Outputting image plot as .png")
        if not ax: plt.savefig(path+"/saved_plots/ColumnDensity/"+fname+".png", bbox_inches="tight")