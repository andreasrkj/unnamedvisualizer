import os
import numpy as np
import astropy.constants as cnst
import astropy.units as unit
import matplotlib.pyplot as plt
from scipy.integrate import simps
from radmc3dPy_SSJ import image
from ..helper_functions import goto_folder, check_folders, get_view
from .image_generation import create_image

def make_3d_datacube(isink, iout, npix, sizeau, setthreads, dpc, view=None, inclination=None, rotangle=None, sed_points=20, verbose=1):
    # Generate the 3D datacube used for SED and Tbol maps
    path = goto_folder(isink, iout)
    check_folders(path)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)

    fname = "3d-datacube-"+view_str+"-npix"+str(npix)+"-"+str(sizeau)+"au-"+str(sed_points)+"pts"

    # Create wavelength spectrum
    light_speed = 299792458 * 1e6 # mum/s
    freqs = np.logspace(np.log10(30), np.log10(30000), sed_points) * 1e9 # Hz
    wavs = light_speed/freqs

    # If the datacube doesn't exist we should load it in!
    if not os.path.exists(path+"/saved_values/"+fname+".npy"):
        for i in range(len(freqs)):
            create_image(isink, iout, npix=npix, wav=wavs[i], sizeau=sizeau, setthreads=setthreads,
                            inclination=inclination, rotangle=rotangle)
            
            img = image.readImage(fname=path+"/image.out")

            # If a distance is given, we'd like to convert the units to Jy/px
            # Conversion from erg/s/cm/cm/ster to Jy/pixel (from radmc3dPy)
            if verbose: print("Distance to source given. Converting flux unit to Jy/px...")
            conv = img.sizepix_x * img.sizepix_y / (dpc * cnst.pc.cgs.value)**2. * 1e23
            img.image *= conv

            # We want to save the image grid in an array
            if i == 0:
                img_stack = np.copy(img.image)
            else:
                img_stack = np.dstack([img_stack, img.image])
            
        np.save(path+"/saved_values/"+fname+".npy", img_stack)
    else:
        if verbose: print("The image cube has already been generated. Loading...")
        img_stack = np.load(path+"/saved_values/"+fname+".npy")
    
    return freqs, img_stack.transpose((1,0,2))

def flux2Tbol(flux,wav=None,freq=None):
    """
    Calculate bolometric Temperature (as defined in Myers et. al. (1993)).
    (Stolen) "Appropriated" from pyradmc3d.
    
    Parameters:
        flux: Flux in units of Jy.
        wav (optional):  Wavelength in microns.
        freq (optional):  Frequency in Hz.
    
    Returns:
        Tbol (ndarray): Bolometric temperature calculated as an integral of flux over wavelength.
    """
    
    c = cnst.c.to('um/s').value # speed of light in um/s
    
    # ------- Check inputs
    if freq is None:
        if wav is None:
            raise ValueError("wavelength or frequency must be given")
        else:
            freq = c/wav
    
    if (np.diff(freq) < 0).any(): # some elements are not increasing
        if (np.diff(freq) < 0).all(): # all elements are not increasing
            freq = freq[::-1]
            flux = flux[::-1]
        else:
            raise ValueError("frequency array must consist of consecutive values")
    
    # ------- Calculate Tbol
    return 1.25e-11 * simps(freq*flux,x=freq) / simps(flux,x=freq) # (Eq. 1 and 2)

def create_sed(isink, iout, npix, sizeau, setthreads, dpc, view=None, inclination=None, rotangle=None, sed_points=20, subtract_isrf=True, verbose=1):
    '''
    Create the Spectral Energy Distribution (SED) for the given sink and output

    Parameters:
        isink (int): The sink ID
        iout (int): The snapshot ID
        npix (int): Number of pixels on the rectangular images
        sizeau (float): Diameter of the image in au
        setthreads (int): Number of threads for RADMC-3D to use for image generation
        dpc (float, optional): Distance to source in pc
        view (string, optional): Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination (float, optional): The inclination of the viewed image
        rotangle (float, optional): The clockwise rotation around the z-axis of the system
        sed_points (int, default=20): How many data points to make for the SED
        subtract_isrf (bool, default=True): Whether to subtract the interstellar radiation field (background emission)
        verbose (bool, default=1): Report task activity
        

    Returns:
        freqs, fluxes (ndarray, ndarray) The frequencies used and the flux calculated at the given frequencies
    
    '''
    # Check if all the folders, that need to exist, do exist
    path = goto_folder(isink, iout)
    check_folders(path)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)
        
    fname = "sed-"+view_str+"-"+str(sizeau)+"au"

    # Handling of different cases
    # if SED exists but not ISRF subtracted
    if os.path.exists(path+"/saved_values/"+fname+".dat") and subtract_isrf:
        if verbose: print("The SED exists but is not ISRF subtracted. Performing subtraction and re-saving...")
        # Load non-subtracted ISRF
        freqs, fluxes = np.loadtxt(path+"/saved_values/"+fname+".dat", unpack=True)
        light_speed = 299792458 * 1e6 # mum/s
        wavs = light_speed/freqs

        # Perform subtraction
        if verbose: print("Subtracting ISRF from calculated fluxes...")

        # Load in the ISRF
        isrf = np.loadtxt(path+"/external_source.inp")

        # Read off the length of the isrf
        datapts = int(isrf[1])

        isrf_wav = isrf[2:datapts+2]
        intensity = isrf[datapts+2:] * 18700000000000.0**2 / (140 * cnst.pc.cgs.value)**2. * 1e23 # Convert to Jy/px
        isrf_flux = intensity * npix**2
        isrf_flux_interp = np.interp(wavs, isrf_wav, isrf_flux) # intensity taken over the image area, interpolated to SED datapoints (Jy)

        # We can now subtract from the calculated fluxes (which are in the same unit Jy)
        fluxes -= isrf_flux_interp

        # Now we save the frequencies and fluxes to a file for easy access
        np.savetxt(path+"/saved_values/"+fname+"-isrf_subtracted.dat", np.transpose([freqs, fluxes]))

    # Check if the values for the SED already exist
    elif os.path.exists(path+"/saved_values/"+fname+".dat") or os.path.exists(path+"/saved_values/"+fname+"-isrf_subtracted.dat"):
        if verbose: print("The SED for this configuration already exists. Loading it in...")
        # Load in the SED data from the folder
        if subtract_isrf:
            freqs, fluxes = np.loadtxt(path+"/saved_values/"+fname+"-isrf_subtracted.dat", unpack=True)
        else:
            freqs, fluxes = np.loadtxt(path+"/saved_values/"+fname+".dat", unpack=True)
    else: # Create SED
        if verbose: print("The SED for this configuration doesn't exist. Creating...")
        # Call the 3D datacube function to see if the datacube exists, otherwise make it
        freqs, img_stack = make_3d_datacube(isink, iout, npix=npix, sizeau=sizeau, setthreads=setthreads, dpc=dpc, inclination=inclination, rotangle=rotangle, 
                                            sed_points=sed_points, verbose=verbose)
        light_speed = 299792458 * 1e6 # mum/s
        wavs = light_speed/freqs

        # Let's make a flux array by summing over the image planes in the (npix,npix,sed_points) array
        fluxes = img_stack.sum(axis=0).sum(axis=0)
        
        if subtract_isrf: # Now we check whether we want to subtract the ISRF
            if verbose: print("Subtracting ISRF from calculated fluxes...")
            # Load in the ISRF
            isrf = np.loadtxt(path+"/external_source.inp")
            # Read off the length of the isrf
            datapts = int(isrf[1])

            isrf_wav = isrf[2:datapts+2]
            intensity = isrf[datapts+2:] * 18700000000000.0**2 / (140 * cnst.pc.cgs.value)**2. * 1e23 # Convert to Jy/px
            isrf_flux = intensity * npix**2
            isrf_flux_interp = np.interp(wavs, isrf_wav, isrf_flux) # intensity taken over the image area, interpolated to SED datapoints (Jy)

            # We can now subtract from the calculated fluxes (which are in the same unit Jy)
            fluxes -= isrf_flux_interp

            # Now we save the frequencies and fluxes to a file for easy access
            np.savetxt(path+"/saved_values/"+fname+"-isrf_subtracted.dat", np.transpose([freqs, fluxes]))
        else:
            np.savetxt(path+"/saved_values/"+fname+".dat", np.transpose([freqs, fluxes]))

    # Now we can output the frequencies and the fluxes
    return freqs, fluxes

def plot_sed(isink, iout, npix, sizeau, setthreads, dpc, view=None, inclination=None, rotangle=None, sed_points=20, subtract_isrf=True, verbose=1, plot_planck=True, ax=None, save=False):
    '''
    Create and plot the Spectral Energy Distribution (SED) for the given sink and output

    Parameters:
        isink (int): The sink ID
        iout (int): The snapshot ID
        npix (int): Number of pixels on the rectangular images
        sizeau (float): Diameter of the image in au
        setthreads (int): Number of threads for RADMC-3D to use for image generation
        dpc (float, optional): Distance to source in pc
        view (string, optional): Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination (float, optional): The inclination of the viewed image
        rotangle (float, optional): The clockwise rotation around the z-axis of the system
        sed_points (int, default=20): How many data points to make for the SED
        subtract_isrf (bool, default=True): Whether to subtract the interstellar radiation field (background emission)
        verbose (bool, default=1): Report task activity
        plot_planck (bool, default=True): Overplot the Planck spectrum corresponding to the SED bolometric temperature
        ax (matplotlib.axes.Axes class, default=None): Supply a matplotlib axis to plot in. If not supplied, an Axes class will be generated automatically.
        save (bool, default=False): Whether to output the generated plot as a .png file    
    '''
    path = goto_folder(isink, iout)
    check_folders(path)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)
    
    if subtract_isrf:
        fname = "sed-"+view_str+"-"+str(sizeau)+"au-isrf_subtracted"
    else:
        fname = "sed-"+view_str+"-"+str(sizeau)+"au"

    freqs, fluxes = create_sed(isink=isink, iout=iout, npix=npix, sizeau=sizeau, setthreads=setthreads, dpc=dpc, view=view, inclination=inclination, 
                             rotangle=rotangle, sed_points=sed_points, subtract_isrf=subtract_isrf, verbose=verbose)
    reduced_flux = freqs * unit.Hz * fluxes * unit.Jy
    c_mum = 299792458 * 1e6 # mum / s
    if ax is None: 
        fig, ax = plt.subplots(1, 1, figsize=(8,6))
    else:
        if save: print("Note, you've supplied a matplotlib axis while setting 'save' = True, this may create a weird-looking plot in the .png, if part of subplot")

    ax.loglog(c_mum/freqs, reduced_flux.to(unit.erg/unit.s/unit.cm**2), '.--', label="Synthetic SED")

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

    if save:
        if verbose: print("Outputting image plot as .png")
        plt.savefig(path+"/saved_plots/"+fname+".png", bbox_inches="tight")

def create_Tbol_map(isink, iout, npix, sizeau, setthreads, dpc, view=None, inclination=None, rotangle=None, sed_points=20, verbose=1):
    # Check if all the folders, that need to exist, do exist
    path = goto_folder(isink, iout)
    check_folders(path)
    _, inclination, rotangle = get_view(view, inclination, rotangle, verbose)
    
    # Load in the 3D data cube for calculating bolometric temperature
    freqs, img_stack = make_3d_datacube(isink, iout, npix=npix, sizeau=sizeau, setthreads=setthreads, dpc=dpc, inclination=inclination, rotangle=rotangle, 
                                        sed_points=sed_points, verbose=verbose)

    # Make the bolometric temperature map
    Tbol_map = flux2Tbol(img_stack, freq=freqs)

    return Tbol_map

def plot_Tbol_map(isink, iout, npix, sizeau, setthreads, dpc, view=None, inclination=None, rotangle=None, sed_points=20, verbose=1, vmin=None, vmax=None, xlim=None, ylim=None, ax=None, save=False):
    # Check if all the folders, that need to exist, do exist
    path = goto_folder(isink, iout)
    check_folders(path)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)

    save_name = "Tbol-"+view_str+"-npix"+str(npix)+"-"+str(sizeau)+"au-"+str(sed_points)+"pts"

    if ax is None: # Create a figure if not supplied
        fig, ax = plt.subplots(1,1, figsize=(8,10))
    else:
        if save: print("Note, you've supplied a matplotlib axis while setting 'save' = True, this may create a weird-looking plot in the .png")

    # Generate the bolometric temperature
    Tbol_map = create_Tbol_map(isink, iout, npix=npix, sizeau=sizeau, setthreads=setthreads, dpc=dpc, inclination=inclination, rotangle=rotangle, 
                               sed_points=sed_points, verbose=verbose)

    # Plot the bolometric temperature
    if vmin is None: vmin = Tbol_map.min()
    if vmax is None: vmax = Tbol_map.max()

    if vmin > Tbol_map.min():
        extend = "min"
    elif vmax < Tbol_map.max():
        extend="max"
    elif (vmin > Tbol_map.min()) and (vmin < Tbol_map.max()):
        extend="both"
    else:
        extend = "neither"

    im = ax.imshow(Tbol_map, extent=(-sizeau/2,sizeau/2,-sizeau/2,sizeau/2), cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    cbar = plt.colorbar(im, ax=ax, pad=0, orientation="horizontal", location="top", extend=extend)
    cbar.set_label("Bolometric Temperature [K]", size=20)
    cbar.ax.tick_params(labelsize=14)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.xaxis.label.set_visible(False); ax.yaxis.label.set_visible(False)
    ax.set_yticklabels([]); ax.set_yticks([])
    ax.set_xticklabels([]); ax.set_xticks([])
    
    # Create scale bar
    # We should normalize the distances to the edges
    plot_size = np.abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    bar_length = int(25 + 25 * (plot_size // 250))
    end_point = 4/5 * plot_size//2

    ax.hlines(-375/500 * plot_size//2, end_point - bar_length, end_point, color="white", linestyles="solid", linewidths=3)
    ax.text(end_point - bar_length/2, -375/500 * plot_size//2 -plot_size//2*20/500, str(bar_length)+" AU", ha="center", va='top', color="white", fontsize=20)

    if save: 
        print("Outputting image plot as .png")
        plt.savefig(path+"/saved_plots/TemperatureMap/"+save_name+".png", bbox_inches="tight")