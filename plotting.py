import os, sys
sys.path.insert(0,'/groups/astro/troels/python')
sys.path.insert(0,'/groups/astro/troels/python/sfrimann')
sys.path.insert(0,'/groups/astro/troels/python/sigurd')
import pyradmc3d as pyrad
from radmc3dPy_SSJ import image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse
from matplotlib.offsetbox import AuxTransformBox, AnchoredOffsetbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Import functions from main and data generation
from .main import *
from .data_generation import *

# NEW plot functions

def plot_single_image(isink, iout, npix, wav, sizeau, setthreads, view=None, inclination=None, rotangle=None, dpc=None, vmax=None, log=False, ax=None, verbose=1):
    '''Create/load an image at a single wavelength and plot it.'''

    path = goto_folder(isink, iout)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)
    
    img = single_wavelength_image(isink=isink, iout=iout, npix=npix, wav=wav, sizeau=sizeau, setthreads=setthreads, 
                                  inclination=inclination, rotangle=rotangle, dpc=dpc, nostar=True, verbose=verbose)
    
    if not ax: # Create a figure if not supplied
        fig, ax = plt.subplots(1,1, figsize=(8,10))

    if log:
        cb_label = "$\\log(I_\\nu / \\mathrm{max}(I_\\nu))$"
        plot_img = np.log10(img.image / img.image.max())
    else:
        if dpc:
            cb_label = "[mJy/px]"
            plot_img = img.image*1e3
        else:
            cb_label = "[$\\mathrm{erg/s/cm^2/Hz/ster}$]"
            plot_img = img.image

    if not vmax:
        vmax = plot_img.max()
        extend="neither"
    if vmax < plot_img.max():
        extend="max"
    else:
        extend="neither"

    im = ax.imshow(plot_img, cmap="magma", vmax=vmax, extent=(-sizeau/2,sizeau/2,-sizeau/2,sizeau/2))
    plt.colorbar(im, ax=ax, label=cb_label, pad=0, orientation="horizontal", location="top", extend=extend)

    # Remove axes
    ax.xaxis.label.set_visible(False); ax.yaxis.label.set_visible(False)
    ax.set_yticklabels([]); ax.set_yticks([])
    ax.set_xticklabels([]); ax.set_xticks([])
    
    # Create scale bar
    # We should normalize the distances to the edges
    bar_length = 25 + 25 * (sizeau // 250)
    end_point = 4/5 * sizeau//2

    ax.hlines(-375/500 * sizeau//2, end_point - bar_length, end_point, color="white", linestyles="solid", linewidths=3)
    ax.text(end_point - bar_length/2, -375/500 * sizeau//2 -sizeau//2*20/500, str(bar_length)+" AU", ha="center", va='top', color="white", fontsize=20, weight="heavy")
    ax.text(0, 49/50 * sizeau//2, "$\\lambda = "+str(np.round(wav,2))+"$µm", ha="center", va="top", color="white", fontsize=18)

    if not ax: 
        if verbose: print("Outputting image plot as .png")
        plt.savefig(path+"/saved_plots/SingleWav/image-"+view_str+"-singlewav-"+str(sizeau)+"au-"+str(int(wav))+"mu.png", bbox_inches="tight")

def plot_moment_map(isink, iout, npix, sizeau, setthreads, iline, widthkms, linenlam, imolspec=1, moment=0, view=None, inclination=None, rotangle=None, 
                    dpc=None, beam=None, verbose=1, nostar=True, casa=False, writefits=False, antennalist=None, totaltime=None, threshold="4mJy", niter=5000, 
                    vclip=None, ax=None, xlim=None, ylim=None):
    path = goto_folder(isink, iout)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)

    molecules = np.loadtxt(path+"/lines.inp", skiprows=2, dtype=str)[:,0]
    molecule_name = molecules[imolspec-1] # The imolspec is 1-indexed
    print_name = np.loadtxt(path+"/molecule_"+molecule_name+".inp", dtype=str, max_rows=2)[1]

    # Now we need to find the transition (which energy levels we move through)
    # Read how many transitions there are
    ntrans = np.loadtxt(path+"/molecule_"+molecule_name+".inp", skiprows=49, max_rows=1)
    # Load in the transition values [1-indexed as well]
    transitions = np.loadtxt(path+"/molecule_"+molecule_name+".inp", skiprows=51, max_rows=int(ntrans), usecols=(0,1,2), dtype=int)
    printtrans = transitions[iline-1] - 1
    
    mmap, img = create_moment_map(isink=isink, iout=iout, npix=npix, sizeau=sizeau, setthreads=setthreads, iline=iline, widthkms=widthkms, linenlam=linenlam, 
                                  imolspec=imolspec, moment=moment, inclination=inclination, rotangle=rotangle, dpc=dpc, beam=beam, verbose=verbose, 
                                  nostar=nostar, casa=casa, writefits=writefits, antennalist=antennalist, totaltime=totaltime, threshold=threshold, niter=niter, return_img=True)

    # Appropriated from radmc3dPy
    if moment == 0:
        cmap = "Spectral_r"
        if dpc:
            mmap = mmap * 1e3 # Turn to mJy :)
            if casa or beam:
                cb_label = "[mJy/beam $\\times$ km/s]"
            else:
                cb_label = "[mJy/px $\\times$ km/s]"
        else:
            cb_label = '[erg/s/cm$^2$/Hz/ster*km/s]'
    if moment == 1:
        cmap = "RdBu_r"
        cb_label = 'Velocity [km/s]'
    if moment > 1:
        powex = str(moment)
        cmap = "Spectral_r"
        cb_label = r'v$^' + powex + '$ [(km/s)$^' + powex + '$]'

    if not ax: # Create a figure if not supplied
        fig, ax = plt.subplots(1,1, figsize=(8,10))

    if casa:
        # Add beam
        aux_tr_box = AuxTransformBox(ax.transData)
        aux_tr_box.add_artist(Ellipse((0,0), img.beam[0] * img.sizepix_x/unit.AU.to(unit.cm), img.beam[1] * img.sizepix_y/unit.AU.to(unit.cm), img.beam[2], color="black"))
        box = AnchoredOffsetbox(child=aux_tr_box, loc="lower left", frameon=True)
        ax.add_artist(box)

        # Mask unwanted values
        if moment > 0:
            # We need to figure out which values to mask
            mmap0 = create_moment_map(isink=isink, iout=iout, npix=npix, sizeau=sizeau, setthreads=setthreads, iline=iline, widthkms=widthkms, linenlam=linenlam, 
                                     imolspec=imolspec, moment=0, inclination=inclination, rotangle=rotangle, dpc=dpc, beam=beam, verbose=0, nostar=nostar, 
                                     casa=casa, writefits=writefits, antennalist=antennalist, totaltime=totaltime, threshold=threshold, niter=niter, return_img=False)
            mmap = np.ma.masked_where(mmap0 < 3*img.rms, mmap)
        else:
            mmap = np.ma.masked_where(mmap < 3*img.rms, mmap)

    if beam and not casa:
        # Add beam
        beam_x = beam[0] / np.sqrt(2*np.log(2)); beam_y = beam[1] / np.sqrt(2*np.log(2)) # Beam radius w = FWHM/sqrt(2*ln(2))
        aux_tr_box = AuxTransformBox(ax.transData)
        aux_tr_box.add_artist(Ellipse((0,0), beam_x / (sizeau/npix / dpc), beam_y / (sizeau/npix / dpc), color="black"))
        box = AnchoredOffsetbox(child=aux_tr_box, loc="lower left", frameon=True)
        ax.add_artist(box)

    # Clipping the vmin and vmax
    if isinstance(vclip, tuple):
        if len(vclip) != 2:
            msg = 'Wrong shape in vclip. vclip should be a two element tuple with (clipmin, clipmax)'
            raise ValueError(msg)
        else:
            vmin = vclip[0]; vmax = vclip[1]
            # Set the colorbar extend
            if vclip[1] < mmap.max():
                cbar_ex = "max"
            elif vclip[0] > mmap.min():
                cbar_ex = "min"
            elif vclip[0] > mmap.min() and vclip[1] < mmap.max():
                cbar_ex = "both"
            else:
                cbar_ex = "neither"
            mmap = mmap.clip(vclip[0], vclip[1])
    else:
        cbar_ex = "neither"
        # ADDITION: Let's make the min and max the same value
        if moment == 1:
            if np.abs(mmap.max()) > 0 and np.abs(mmap.max()) > np.abs(mmap.min()):
                vmin = -mmap.max(); vmax = mmap.max()
            elif np.abs(mmap.max()) > 0 and np.abs(mmap.max()) < np.abs(mmap.min()):
                vmin = mmap.min(); vmax = np.abs(mmap.min())
        else:
            vmin = None; vmax = None

    if casa: 
        plot_origin = "lower"
    else:
        plot_origin = None

    im = ax.imshow(mmap, extent=(-sizeau/2,sizeau/2,-sizeau/2,sizeau/2), cmap=cmap, vmin=vmin, vmax=vmax, origin=plot_origin)
    plt.colorbar(im, ax=ax, label=cb_label, pad=0, orientation="horizontal", location="top", extend=cbar_ex)

    # Remove axes
    ax.xaxis.label.set_visible(False); ax.yaxis.label.set_visible(False)
    ax.set_yticklabels([]); ax.set_yticks([])
    ax.set_xticklabels([]); ax.set_xticks([])

    # Cut the image and give the size needed for the scale bar
    if xlim is not None: 
        ax.set_xlim(xlim[0], xlim[1])
        plot_size = np.abs(xlim[1] - xlim[0])
    else: plot_size = sizeau
    if ylim is not None: ax.set_ylim(ylim[0], ylim[1])

    # Create scale bar
    # We should normalize the distances to the edges
    bar_length = 25 + 25 * (plot_size // 250)
    end_point = 4/5 * plot_size//2

    # Pick color for overlay
    if casa or moment>0:
        text_color = "black"
    else:
        text_color = "white"

    ax.hlines(-375/500 * plot_size//2, end_point - bar_length, end_point, color=text_color, linestyles="solid", linewidths=3)
    ax.text(end_point - bar_length/2, -375/500 * plot_size//2 -plot_size//2*20/500, str(bar_length)+" AU", ha="center", va='top', color=text_color, fontsize=20, weight="heavy")
    ax.text(0, 49/50 * plot_size//2, print_name+" J="+str(printtrans[1])+"-"+str(printtrans[2])+" transition", ha="center", va="top", color=text_color, fontsize=18)

    if ax == None: 
        if verbose: print("Outputting image plot as .png")
        plt.savefig(path+"/saved_plots/MomentMaps/moment-"+str(moment)+"-map-"+molecule_name+"-"+view_str+"-"+str(sizeau)+"au-transition"+str(iline)+".png", bbox_inches="tight")

def plot_sed(isink, iout, npix, sizeau, setthreads, dpc, view=None, inclination=None, rotangle=None, sed_points=20, subtract_isrf=True, verbose=1, plot_planck=True, ax=None):
    path = goto_folder(isink, iout)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)
    
    freqs, fluxes = create_sed(isink=isink, iout=iout, npix=npix, sizeau=sizeau, setthreads=setthreads, dpc=dpc, view=view, inclination=inclination, 
                             rotangle=rotangle, sed_points=sed_points, subtract_isrf=subtract_isrf, verbose=verbose)
    reduced_flux = freqs * unit.Hz * fluxes * unit.Jy
    c_mum = 299792458 * 1e6 # mum / s
    if not ax: 
        fig, ax = plt.subplots(1, 1, figsize=(8,6))

    ax.loglog(c_mum/freqs, reduced_flux.to(unit.erg/unit.s/unit.cm**2), label="Synthetic SED")

    Tbol_SED = flux2Tbol(fluxes,freq=freqs) * unit.K

    # Plot the corresponding Planck function
    if plot_planck:
        conv = 18700000000000.0**2 / (140 * cnst.pc.cgs.value)**2. * 800**2 # steradian to pixel times pixel area
        planck = 2 * cnst.h * (freqs*unit.Hz)**3 / cnst.c**2 / (np.exp(cnst.h * (freqs*unit.Hz) / cnst.k_B / Tbol_SED) - 1) * conv

        ax.plot(c_mum/freqs, (freqs*unit.Hz*planck).to(unit.erg/unit.s/unit.cm**2), label="$\\nu B_\\nu (T_\\mathrm{bol})$")

    ax.set_xlabel("Wavelength [$\\mu$m]", fontsize=20, y=0.01)
    ax.set_ylabel("$\\nu F_\\nu$ [erg/s/cm${}^2$]", fontsize=20, x=0.01)
    legend = ax.legend(loc="upper right")
    ax.grid()

    # Add Tbol text in a box centered below the legend, just a bit closer
    legend_box = legend.get_window_extent(ax.figure.canvas.get_renderer())
    bbox_axes = ax.transAxes.inverted().transform(legend_box)
    # Center x under the legend, y just below it
    x = (bbox_axes[0][0] + bbox_axes[1][0]) / 2
    y = bbox_axes[0][1] - 0.02
    ax.text(
        x, y,
        "$T_\\mathrm{bol}=$ "+str(np.round(Tbol_SED,2)),
        ha='center', va='top', color="gray", fontsize=14,
        transform=ax.transAxes,
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3')
    )

def plot_column_density(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, vclip=None, draw_star=False, verbose=1, ax=None):
    path = goto_folder(isink, iout)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)

    if vclip == None: vclip = (None,None) # Make sure it's a tuple
    if not ax: 
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    coldens = calc_column_density(isink=isink, iout=iout, resolution=resolution, width=width, dz=dz, 
                                  inclination=inclination, rotangle=rotangle, verbose=verbose)

    im = ax.imshow(np.log10(coldens), cmap = 'cividis', vmin=vclip[0], vmax=vclip[1], extent=(-width/2,width/2,-width/2,width/2))
    cbar = plt.colorbar(im, ax=ax, location="top", orientation="horizontal", pad=0)
    cbar.set_label("$\\log_{10} \\Sigma \: [\\mathrm{g/cm^2}]$", size=20)

    ax.xaxis.label.set_visible(False); ax.yaxis.label.set_visible(False)
    ax.set_yticklabels([]); ax.set_yticks([])
    ax.set_xticklabels([]); ax.set_xticks([])

    # Create scale bar
    # We should normalize the distances to the edges
    bar_length = 25 + 25 * (width // 250)
    end_point = 4/5 * width//2

    ax.hlines(-375/500 * width//2, end_point - bar_length, end_point, color="white", linestyles="solid", linewidths=3)
    ax.text(end_point - bar_length/2, -375/500 * width//2 -width//2*20/500, str(bar_length)+" AU", ha="center", va='top', color="white", fontsize=20, weight="heavy")
    if draw_star: ax.scatter(0, 0, s=100, marker="*",c="white",edgecolors='black')

    # Save the figure
    print("Outputting image plot as .png")
    if not ax: plt.savefig(path+"/saved_plots/ColumnDensity/column-density-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)+".png", bbox_inches="tight")

def plot_velocities(isink, iout, resolution, width, dz, view="face-on", vclip=None, draw_star=False, ax=None):
    path = goto_folder(isink, iout)

    if vclip == None: vclip = (None,None)
    if not ax: 
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    velocities = calc_velocities(isink=isink, iout=iout, resolution=resolution, width=width, dz=dz, view=view)

    if view == "face-on":
        plot_img = np.rot90(velocities, k=1)
        cmap = 'Spectral_r'
        cb_label = 'Midplane Speed [km s$^{-1}]$'
    elif view == "edge-on":
        plot_img = np.fliplr(velocities)
        cmap = "RdBu_r"
        cb_label = '$v_\\mathrm{\\phi,proj}$ [km s$^{-1}]$'

    im = ax.imshow(plot_img, cmap = cmap, vmin = vclip[0], vmax = vclip[1], extent=(-width/2,width/2,-width/2,width/2), origin="lower")
    cbar = plt.colorbar(im, ax=ax, location="top", orientation="horizontal", pad=0)
    cbar.set_label(cb_label, size=20)

    ax.xaxis.label.set_visible(False); ax.yaxis.label.set_visible(False)
    ax.set_yticklabels([]); ax.set_yticks([])
    ax.set_xticklabels([]); ax.set_xticks([])

    # Create scale bar
    # We should normalize the distances to the edges
    bar_length = 25 + 25 * (width // 250)
    end_point = 4/5 * width//2

    ax.hlines(-375/500 * width//2, end_point - bar_length, end_point, color="white", linestyles="solid", linewidths=3)
    ax.text(end_point - bar_length/2, -375/500 * width//2 -width//2*20/500, str(bar_length)+" AU", ha="center", va='top', color="white", fontsize=20, weight="heavy")
    if draw_star: ax.scatter(0, 0, s=100, marker="*",c="white",edgecolors='black')

    print("Outputting image plot as .png")
    plt.savefig(path+"/saved_plots/Velocities/velocities-"+view+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)+".png", bbox_inches="tight")

#def plot_optical_depth(isink, iout, npix, wav=None, sizeau=1000, setthreads=4, view=None, inclination=None, rotangle=None,
#                       imolspec=1, iline=None, widthkms=None, linenlam=None, nostar=True, save=True, show=True, create_fig=True):
#    '''
#    Plot the optical depth weighted by the intensity. 
#    If contour == True, the contour is plotted on top of the moment 0 map unless a single wavelength is requested.
#    '''
#    path = goto_folder(isink, iout)
#    if inclination is not None and rotangle is not None: # Make the string for the image
#        print("Inclination and rotation angle given, overriding 'view' argument, if given.")
#        view = "incl"+str(inclination)+"-angle"+str(rotangle)
#
#    # Get the optical depth
#    tau_img = trace_tau(isink, iout, npix=npix, wav=wav, sizeau=sizeau, setthreads=setthreads, view=view, inclination=inclination, rotangle=rotangle,
#                        imolspec=imolspec, iline=iline, widthkms=widthkms, linenlam=linenlam, nostar=nostar)
#    
#    # We want to weigh the optical depth with the intensity
#    if np.count_nonzero([iline, widthkms, linenlam]) == 3:
#        # Let's read in which molecule is of interest here
#        molecules = np.loadtxt("lines.inp", skiprows=2, dtype=str)[:,0]
#        molecule_name = molecules[imolspec-1] # The imolspec is 1-indexed
#
#        img = molecular_lines_image(isink=isink, iout=iout, npix=npix, sizeau=sizeau, setthreads=setthreads,
#                                    imolspec=imolspec, iline=iline, widthkms=widthkms, linenlam=linenlam, 
#                                    view=view, inclination=inclination, rotangle=rotangle, nostar=nostar)
#
#    elif np.count_nonzero([iline, widthkms, linenlam]) < 3 and np.count_nonzero([iline, widthkms, linenlam]) > 0:
#        raise ValueError("To do molecular line images, 'iline', 'widthkms' and 'linenlam' must all be given.")
#    else:
#        img = single_wavelength_image(isink=isink, iout=iout, npix=npix, wav=wav, sizeau=sizeau, setthreads=setthreads, view=view, 
#                                      inclination=inclination, rotangle=rotangle, nostar=True)
#    print("Loading in the image to do the intensity weighting...")
#
#    # Get the total intensity
#    Itot = img.image.sum(axis=2)
#    
#    # Weighted tau
#    tt = (tau_img.image*img.image).sum(axis=2) / Itot
#
#    # Rotate the image since for some reason it needs to do that???
#    tt = np.rot90(tt, k=1, axes=(0,1))
#
#    if create_fig: fig = plt.figure(figsize=(8,6))
#    plt.imshow(tt, extent=(-500,500,-500,500))
#    plt.xlabel("X [AU]"); plt.ylabel("Y [AU]")
#    plt.colorbar(label="Optical depth $\\tau$")
#
#    # Now we need to find the transition (which energy levels we move through)
#    # Read how many transitions there are
#    ntrans = np.loadtxt(path+"/molecule_"+molecule_name+".inp", skiprows=49, max_rows=1)
#    # Load in the transition values [1-indexed as well]
#    transitions = np.loadtxt(path+"/molecule_"+molecule_name+".inp", skiprows=51, max_rows=int(ntrans), usecols=(0,1,2), dtype=int)
#    printtrans = transitions[iline-1] - 1
#    print_name = np.loadtxt(path+"/molecule_"+molecule_name+".inp", dtype=str, max_rows=2)[1]
#
#    if "ice" in molecule_name:
#         plt.text(475, -475, f"{print_name} J = {printtrans[1]}-{printtrans[2]} transition (with freeze-out)", color="white", fontsize=15, ha="right")
#    else:
#         plt.text(475, -475, f"{print_name} J = {printtrans[1]}-{printtrans[2]} transition", color="white", fontsize=15, ha="right")
#
#    plt.tight_layout()
#    print("Outputting image as .png")
#    if save:
#        if np.count_nonzero([iline, widthkms, linenlam]) == 3:
#            plt.savefig(path+"/saved_plots/TauMap/optdepthmap-"+molecule_name+"-"+view+"-"+str(sizeau)+"au-transition"+str(iline)+"-widthkms"+str(widthkms)+"-lines"+str(linenlam)+".png", bbox_inches="tight")
#        else:
#            plt.savefig(path+"/saved_plots/TauMap/optdepthmap-"+view+"-"+str(sizeau)+"au-"+str(int(wav))+"mu.png", bbox_inches="tight")
#    if not show:
#        plt.close()
#
#    return tt # If you want to make your own plot

def viewpoint_comparison_plot(isink, iout, npix=800, wav=None, sizeau=1000, setthreads=4, imolspec=1, iline=None, widthkms=None, linenlam=None, 
                              moment=0, dpc=None, beam=None, verbose=1, nostar=True, casa=False, writefits=False, alma_config=None, threshold="1e-6Jy", 
                              niter=500000, pbcor=True, tracetau=False, log=None, vclip=None):
    path = goto_folder(isink, iout)

    # Convert to tuple if only 1 value is given
    if not isinstance(vclip, (tuple, np.ndarray, list)):
        vclip = (0, vclip)

    molecules = np.loadtxt(path+"/lines.inp", skiprows=2, dtype=str)[:,0]
    molecule_name = molecules[imolspec-1] # The imolspec is 1-indexed

    fig, ax = plt.subplots(1, 3, figsize=(14, 8))
    views = ["face-on", "edge-on-A", "edge-on-B"]

    # Should check whether multi wavelength or single
    if tracetau:
        # Check first if we should trace tau 
        print("No.")
    elif np.count_nonzero([iline, widthkms, linenlam]) == 3: # If all inputs given, do molecular image
        for i in range(len(views)):
            plot_moment_map(isink=isink, iout=iout, npix=npix, sizeau=sizeau, setthreads=setthreads, iline=iline, widthkms=widthkms, linenlam=linenlam, 
                            imolspec=imolspec, moment=moment, view=views[i], dpc=dpc, beam=beam, verbose=verbose, nostar=nostar, casa=casa, writefits=writefits, 
                            alma_config=alma_config, threshold=threshold, niter=niter, pbcor=pbcor, vclip=vclip, ax=ax[i])
            
        plt.subplots_adjust(wspace=0.01)
        plt.savefig(path+"/saved_plots/ViewComparison/moment-"+str(moment)+"-map-"+molecule_name+"-viewcomp-"+str(sizeau)+"au-transition"+str(iline)+".png", bbox_inches="tight")
    elif np.count_nonzero([iline, widthkms, linenlam]) < 3 and np.count_nonzero([iline, widthkms, linenlam]) > 0:
        raise ValueError("To plot moment maps, 'iline', 'widthkms' and 'linenlam' must all be given.")
    else: # Do single wavelength image
        for i in range(len(views)):
            plot_single_image(isink=isink, iout=iout, npix=npix, wav=wav, sizeau=sizeau, setthreads=setthreads, view=views[i], dpc=dpc, log=log, ax=ax[i], verbose=verbose, vmax=vclip[1])

        plt.subplots_adjust(wspace=0.01)
        plt.savefig(path+"/saved_plots/ViewComparison/image-singlewav-viewcomp-"+str(sizeau)+"au-"+str(int(wav))+"mu.png", bbox_inches="tight")

def channel_map(isink, iout, npix, sizeau, setthreads, iline, widthkms, linenlam, imolspec=1, view=None, inclination=None, rotangle=None, 
                nostar=True, save=True):
    '''
    Plots a channel map of a multi-wavelength image.
    Colorbar functionality may be dodgy, so returns the last axis element so one can manually add colorbar.
    '''
    path = goto_folder(isink, iout)
    if inclination is not None and rotangle is not None: # Make the string for the image
        print("Inclination and rotation angle given, overriding 'view' argument, if given.")
        view = "incl"+str(inclination)+"-angle"+str(rotangle)


    molecules = np.loadtxt(path+"/lines.inp", skiprows=2, dtype=str)[:,0]
    molecule_name = molecules[imolspec-1] # The imolspec is 1-indexed
    print_name = np.loadtxt(path+"/molecule_"+molecule_name+".inp", dtype=str, max_rows=2)[1]

    # Now we need to find the transition (which energy levels we move through)
    # Read how many transitions there are
    ntrans = np.loadtxt(path+"/molecule_"+molecule_name+".inp", skiprows=49, max_rows=1)
    # Load in the transition values [1-indexed as well]
    transitions = np.loadtxt(path+"/molecule_"+molecule_name+".inp", skiprows=51, max_rows=int(ntrans), usecols=(0,1,2), dtype=int)
    printtrans = transitions[iline-1] - 1

    img = molecular_lines_image(isink=isink, iout=iout, npix=npix, sizeau=sizeau, setthreads=setthreads, iline=iline, widthkms=widthkms, 
                                linenlam=linenlam, imolspec=imolspec, view=view, inclination=inclination, rotangle=rotangle, nostar=nostar)
    nu0 = img.freq[img.nfreq//2]
    v_kms = (cnst.c * (nu0 - img.freq) / nu0).to(unit.km/unit.s).value

    # Build the figure based on the number of lines probed
    if linenlam <= 9: n = 3
    elif linenlam <= 16: n = 4
    elif linenlam <= 25: n = 5
    elif linenlam <= 36: n = 6
    elif linenlam <= 49: n = 7
    else:
        raise ValueError("linenlam exceeds the recommended number of maps.")

    fig, ax = plt.subplots(n,n, figsize=(16,16))
    ax = ax.flatten()

    for i in range(len(ax)):
        if i < img.nfreq:
            # Calculate the brightness temperature of the image
            Tb = cnst.h.cgs.value * img.freq[i] / cnst.k_B.cgs.value * 1/np.log(1 + 2 * cnst.h.cgs.value * img.freq[i]**3 / np.copy(img.image[:,:,i]) / cnst.c.cgs.value**2)

            plot = ax[i].imshow(Tb, cmap="Spectral_r", origin="lower", vmin=0, vmax=100, extent=(-sizeau/2,sizeau/2,-sizeau/2,sizeau/2))
            ax[i].text(-475/500 * sizeau//2, 490/500 * sizeau//2 ,str(np.round(v_kms[i],2)) + " km/s", va="top", ha="left", color="white", size=18)
            # Remove axes
            ax[i].xaxis.label.set_visible(False); ax[i].yaxis.label.set_visible(False)
            ax[i].set_yticklabels([]); ax[i].set_yticks([])
            ax[i].set_xticklabels([]); ax[i].set_xticks([])

            # Create scale bar
            # We should normalize the distances to the edges
            bar_length = 25 + 25 * (sizeau // 250)
            end_point = 4/5 * sizeau//2

            ax[i].hlines(-375/500 * sizeau//2, end_point - bar_length, end_point, color="white", linestyles="solid", linewidths=3)
            ax[i].text(end_point - bar_length/2, -375/500 * sizeau//2 -sizeau//2*20/500, str(bar_length)+" AU", ha="center", va='top', color="white", fontsize=18, weight="heavy")
        else:
            fig.delaxes(ax[i])
    plt.subplots_adjust(wspace=0.01, hspace=0.01, top=0.88)
    # compute combined horizontal span of the top row and place colorbar centered above it
    left = min(a.get_position().x0 for a in ax[:n])
    right = max(a.get_position().x1 for a in ax[:n])
    top = max(a.get_position().y1 for a in ax[:n]) + 0.01
    cbar_ax = fig.add_axes([left, top, right - left, 0.015])
    cbar = fig.colorbar(plot, cax=cbar_ax, orientation="horizontal", extend="max")
    cbar.set_label("Brightness Temperature [K]", size=20)
    cbar_ax.xaxis.set_label_position('top')
    cbar_ax.xaxis.set_ticks_position('top')

    fig.suptitle("Channel Map "+print_name+" J="+str(printtrans[1])+"-"+str(printtrans[2])+" transition", size=30)
    plt.savefig(path+"/saved_plots/ChannelMaps/channel-map-"+molecule_name+"-transition"+str(iline)+"-"+str(linenlam)+"lines-"+str(sizeau)+"au.png", bbox_inches="tight", dpi=300)

#def comparison_plot(isink, iout, npix, sizeau, setthreads, iline, widthkms,
#                    linenlam, moment, mol1=1, mol2=3, view='face-on', nostar=True, vclip=None, 
#                    overplot_tau=False, show=True):
#    '''
#    SHOULD BE REWORKED **FIXME**
#    Plot the two different molecules side-by-side in a plot with the same colorbar :)
#    '''
#
#    fig, ax = plt.subplots(1 , 2, figsize=(12,5))
#
#    plt.sca(ax[0])
#    plot_moment_map(isink, iout, npix=npix, sizeau=sizeau, setthreads=setthreads, imolspec=mol1, iline=iline, widthkms=widthkms, 
#                    linenlam=linenlam, moment=moment, view=view, nostar=nostar, create_fig=False, vclip=vclip, overplot_tau=overplot_tau, save=False)
#    cb = ax[0].images[-1].colorbar
#    cb.remove()
#    plt.sca(ax[1])
#    plot_moment_map(isink, iout, npix=npix, sizeau=sizeau, setthreads=setthreads, imolspec=mol2, iline=iline, widthkms=widthkms, 
#                    linenlam=linenlam, moment=moment, view=view, nostar=nostar, create_fig=False, vclip=vclip, overplot_tau=overplot_tau, save=False)
#    ax[1].yaxis.label.set_visible(False)
#    ax[1].set_yticklabels([]); ax[1].set_yticks([])
#    cb = ax[1].images[-1].colorbar
#    cb.ax.set_position([cb.ax.get_position().x0 - 0.02, cb.ax.get_position().y0, 
#                        cb.ax.get_position().width, cb.ax.get_position().height])
#    fig.subplots_adjust(wspace=0, hspace=0)
#    plt.tight_layout(pad=0.000001)
#    # Save the figure in the desired folder
#    # Get the molecules we're comparing
#    path = goto_folder(isink, iout)
#    molecules = np.loadtxt(path+"/lines.inp", skiprows=2, dtype=str)[:,0]
#    mol_name1 = molecules[mol1-1] # The imolspec is 1-indexed
#    mol_name2 = molecules[mol2-1] # The imolspec is 1-indexed
#
#    if overplot_tau:
#        plt.savefig(path+"/saved_plots/MolComparison/moment-"+str(moment)+"-map-comparison-"+str(mol_name1)+"-and-"+str(mol_name2)+"-"+view+"-"+str(sizeau)+"au-transition"+str(iline)+"withtau.png", bbox_inches="tight", dpi=300)
#    else:
#        plt.savefig(path+"/saved_plots/MolComparison/moment-"+str(moment)+"-map-comparison-"+str(mol_name1)+"-and-"+str(mol_name2)+"-"+view+"-"+str(sizeau)+"au-transition"+str(iline)+".png", bbox_inches="tight", dpi=300)
#    if not show: plt.close("all")