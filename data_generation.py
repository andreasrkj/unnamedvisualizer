# Import functions necessary to run
import os, sys
import pickle
from .sink_config import sink_dirs
#from .imageclass import casaImageClass
import astropy.constants as cnst
import astropy.units as unit
import astropy.io.fits as fits
from scipy.integrate import simps
from astropy.convolution import convolve_fft, Gaussian2DKernel
sys.path.insert(0,'/groups/astro/troels/python')
sys.path.insert(0,'/groups/astro/troels/python/sfrimann')
sys.path.insert(0,'/groups/astro/troels/python/sigurd')
sys.path.insert(0,"/groups/astro/andreask/python")
import pyradmc3d as pyrad
from radmc3dPy_SSJ import image
from radmc3dPy_SSJ import analyze


import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

import nexus

# Import CASA functions
sys.path.insert(0,'/lustre/hpc/software/astro/casa/casa-6.6.1-17-pipeline-2024.1.0.8/lib/py/lib/python3.8/site-packages/')
import casatasks
from casatools import synthesisutils
su = synthesisutils()

# Import functions from main
from .main import *

#### -----------------------------------
#### Non-image functions (QoL functions)
#### -----------------------------------

# Classes to load in image data
class casaImageClass:
    def __init__(self, isink, iout, image_name, dpc, antennalist=None):
        path = goto_folder(isink, iout)
        # Load in the data and assign keywords
        if antennalist is not None:
            # Transform given image name to simalma image name
            if len(antennalist) > 1:
                config_str = "combined"+"_".join(['alma.cycle7.8.cfg', 'alma.cycle7.5.cfg']).replace("alma.cycle","").replace(".cfg","")
            else:
                config_str = antennalist[0]
            data, header = fits.getdata(path+"/saved_fits/simalma_"+config_str+"_"+image_name+".fits", header=True)
            # Load in the beam and RMS value if CASA image
            stats = get_stats(path, image_name, antennalist)
            self.beam = (header["BMAJ"]/header["CDELT1"], header["BMIN"]/header["CDELT2"], header["BPA"]) # beam size in px
            self.rms = stats["rms"]
        else:
            data, header = fits.getdata(path+"/saved_fits/"+image_name+".fits", header=True)
        if header["NAXIS3"] > 1: # If multi-wavelength
            self.image = data[0,:,::-1,:].transpose((1,2,0)) # Aligned with the axis that radmc3dPy loads in with
        else:
            self.image = data[0,0,:,:].transpose(1,0)
        # Assign header keywords
        self.x = (np.arange(1,header["NAXIS1"]+1) - header["CRPIX1"]) * np.abs(header["CDELT1"]) * np.pi/180 * dpc * unit.pc.to(unit.cm)
        self.y = (np.arange(1,header["NAXIS2"]+1) - header["CRPIX2"]) * np.abs(header["CDELT2"]) * np.pi/180 * dpc * unit.pc.to(unit.cm)
        self.nx = len(self.x)
        self.ny = len(self.x)
        self.sizepix_x = np.abs(header["CDELT1"] * np.pi/180 * dpc * unit.pc.to(unit.cm))
        self.sizepix_y = np.abs(header["CDELT2"] * np.pi/180 * dpc * unit.pc.to(unit.cm))
        self.freq = np.linspace(start=header["CRVAL3"], stop=(header["NAXIS3"]-1)*header["CDELT3"]+header["CRVAL3"], num=header["NAXIS3"])
        self.nfreq = len(self.freq)
        self.wav = (cnst.c / (self.freq * unit.Hz)).to(unit.micron).value
        self.nwav = len(self.wav)

# Class for saving information to create moment maps with same functions as without CASA
#class casaImageClass:
#    def __init__(self, casa_image, x, y, nx, ny, sizepix_x, sizepix_y, nfreq, freqs, beam, rms):
#        self.image = casa_image
#        self.x = x
#        self.y = y
#        self.nx = nx
#        self.ny = ny
#        self.sizepix_x = sizepix_x
#        self.sizepix_y = sizepix_y
#        self.nfreq = nfreq
#        self.freq = freqs
#        self.beam = beam # tuple of (bmaj, bmin, bpa)
#        self.rms = rms            

def _perpendicular_vector(v):
    """
    Compute a unit vector perpendicular to the input vector.

    Parameters:
        v: 3-dimensional normalized vector [x,y,z].
    
    Returns:
        Unit vector perpendicular to the given vector (ndarray).
    """
    if v[2] == 0:
        vperp = np.array([-v[1], v[0], 0])
    else:
        vperp = np.array([1.0, 1.0, -1.0 * (v[0] + v[1]) / v[2]])
    return vperp / np.sqrt(np.sum(vperp**2))

#def get_sink_data(isink, iout):
#    if os.path.exists("saved_values/sink_data.npz"):
#        print("The data for this sink is already saved. Loading...")
#        sink_data = np.loadtxt("saved_values/sink_data.npz")
#    else:
#    
#    return sink_data

# Find the coordinate basis for the system
def calc_coord_basis(isink, iout):
    '''
    Calculate the coordinate basis for the protostellar system, assuming.
    
    Parameters:
        isink: The sink ID.
        iout: The snapshot ID.
    
    Returns:
        The spin vector z-axis (ndarray) as well as two plane vectors plane_vector1 (ndarray) and plane_vector2 (ndarray).
    '''
    path = goto_folder(isink, iout)

    # If the isink is given as a float "isink.level", we should get the level and sink from that
    # FIXME

    # First we check if the data file already exists
    if os.path.exists(path+"/saved_values/coordinate_basis.dat"):
        print("Coordinate basis already generated, loading from file...")
        spin_vector, plane_vector1, plane_vector2 = np.loadtxt(path+"/saved_values/coordinate_basis.dat", unpack=True)
    else:
        check_folders(path) # Check if folder exists
        print("Coordinate basis doesn't exist for this file, generating...")
        # Load the data from the "master" folder
        ramses = nexus.dataclass()
        # Grab the data folder
        if isinstance(isink, float):
            sink_id, level = str(isink).split(".")
            sink_id = int(sink_id); level = int(level)
            print(f"You have specified isink as a float, interpreted as sink ID {sink_id} with max level {level}")
            error_msg = f"The data directory for sink {sink_id} with max level {level} hasn't been configured. Please specify the data directory in 'sink_config.py'." # Only necessary if not configured!
        else:
            sink_id = isink
            error_msg = f"The data directory for sink {sink_id} hasn't been configured. Please specify the data directory in 'sink_config.py'." # Only necessary if not configured!
        try:
            datadir = sink_dirs[str(isink)]
        except:
            raise ValueError(error_msg)

        ramses.load(snap = iout, io = 'RAMSES', path = datadir, sink_id=sink_id, verbose=1, dtype = 'float64')
        # Calculate the new vector basis from the angular momentum vector
        ramses.recalc_L(r = 100)
        # We grab the vector pointing "north" (direction to view disk face-on)
        spin_vector = np.array(ramses.L)

        spin_vector = spin_vector / np.sqrt(np.sum(spin_vector**2))
        plane_vector1 = _perpendicular_vector(spin_vector)
        plane_vector2 = np.cross(spin_vector,plane_vector1)
        coord_basis = np.array([spin_vector, plane_vector1, plane_vector2])

        np.savetxt(path+"/saved_values/coordinate_basis.dat", coord_basis.T)
        print("Coordinate basis created!")
    return spin_vector, plane_vector1, plane_vector2 # z, x, y

def get_projang(v):
    '''
    Calculates the inclination and phi keywords for RADMC-3D.

    Parameters:
        v: The vector that should point towards the viewer.

    Returns:
        inclination (float) and phi (float) parameters for RADMC-3D.
    '''
    incl = np.rad2deg(np.arccos(v[2]))
    phi  = 270 - np.rad2deg(np.arctan2(v[1],v[0]))
    return incl, phi

def get_posang(north_vector, projection_vector):
    if projection_vector[2] == 0:
        yimg = np.array([0, 0, 1])
    else:
        yimg = np.array([-projection_vector[0], -projection_vector[1], 
                         (projection_vector[0]**2+projection_vector[1]**2)/projection_vector[2]])
        if projection_vector[2] < 0:
            yimg = -yimg
    yimg = yimg/np.linalg.norm(yimg)
    ximg = np.cross(yimg, projection_vector)
    ximg = ximg/np.linalg.norm(ximg)
    xnorth = np.dot(ximg, north_vector)
    ynorth = np.dot(yimg, north_vector)
    #print(north_vector)
    #print(projection_vector)
    #print(yimg)
    return 90-np.rad2deg(np.arctan2(ynorth, xnorth))

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

#def OLD_convolve_beam(img, beam, dpc, return_gaussian=False):
#    '''
#    Returns calculated image convolved with a 2D Gaussian beam.
#    If 'return_gaussian' is set to True, returns 2D gaussian function instead of image.
#    '''
#    # If beam (in FWHM) is given (must be array of length 2)
#    if isinstance(beam, tuple) and len(beam) == 2:
#        # Convert beam FWHM (in arcsec) to standard deviation in AU
#        beam_x = beam[0] * dpc / np.sqrt(8 * np.log(2)) # FWHM [''] * physical size [AU/''] / FWHM-to-sigma
#        beam_y = beam[1] * dpc / np.sqrt(8 * np.log(2)) # FWHM [''] * physical size [AU/''] / FWHM-to-sigma
#
#        # Define 2D gaussian beam (defined in AU)
#        gauss = Gaussian2D(amplitude=1/(2*np.pi*beam_x*beam_y), x_mean = 0, y_mean = 0, x_stddev=beam_x, y_stddev=beam_y)
#        x_, y_ = np.meshgrid(img.x/unit.AU.to(unit.cm),img.y/unit.AU.to(unit.cm))
#
#        convolved_img = np.empty_like(img.image)
#        for i in range(img.image.shape[2]):
#            convolved_img[:,:,i] = fftconvolve(img.image[:,:,i], gauss(x_, y_), mode="same")
#        
#        # Add it to the image class to return to user
#        img.image = convolved_img
#        if return_gaussian:
#            return gauss
#        else:
#            return img
#    else:
#        raise ValueError("'beam' must be a tuple of length 2: (FWHM_x, FWHM_y)")

def convolve_beam(img, beam, dpc):
    '''
    Convolves the given image with a Gaussian beam.

    Parameters:
        img: radmc3dPy.image.radmc3dImage class
        beam (tuple):  FWHM beam in arcseconds (beam_x, beam_y)
        dpc (float):  Distance to source in pc

    Returns:
        Calculated image convolved with a 2D Gaussian beam.
    '''
    # If beam (in FWHM) is given (must be array of length 2)
    if isinstance(beam, tuple) and len(beam) == 2:
        # Convert beam FWHM (in arcsec) to standard deviation in AU
        beam_x = beam[0] * dpc / np.sqrt(8 * np.log(2)) # FWHM [''] * physical size [AU/''] / FWHM-to-sigma
        beam_y = beam[1] * dpc / np.sqrt(8 * np.log(2)) # FWHM [''] * physical size [AU/''] / FWHM-to-sigma

        # Define 2D Gaussian kernel
        gauss_kernel = Gaussian2DKernel(beam_x, beam_y)

        convolved_img = np.empty_like(img.image)
        for i in range(img.image.shape[2]):
            convolved_img[:,:,i] = convolve_fft(img.image[:,:,i], gauss_kernel)
        
        # Add it to the image class to return to user
        img.image = convolved_img
        return img
    else:
        raise ValueError("'beam' must be a tuple of length 2: (FWHM_x, FWHM_y)")

def get_casa_project_name(image_name):
    '''
    Get the CASA project file name.

    Parameters:
        image_name (string):  Image name generated for the calculated image

    Returns:
        project_name (string): CASA project name
    
    '''
    project_name = image_name
    replist = ["image-", "transition", "widthkms", "lines", "-"]
    repwith = ["", "t", "w", "l", ""]
    for i in range(len(replist)):
        project_name = project_name.replace(replist[i], repwith[i])
    return project_name
    

#def run_casa(image_name, npix, alma_config, path, verbose=1, threshold="1e-6Jy", niter=500000):
#    '''
#    Run the CASA simobserve and simanalyze pipeline.
#
#    Parameters:
#        image_name (string): Image name generated for the calculated image
#        npix (int): Number of pixels on the rectangular images
#        alma_config (string): ALMA configuration file to use with CASA
#        path (string): The path to the snapshot
#        verbose (bool, default=1): Report task activity
#        threshold (string, default="1e-6Jy"):  The threshold to which the image should be cleaned
#        niter (int, default=500000): The number of iterations the cleaning algorithm will run before forcefully stopping.
#    '''
#    # Create the CASA directory
#    try:
#        project_name = get_casa_project_name(image_name)
#        root = os.getcwd()
#        os.chdir(path+"/casa_projects/")
#        if verbose: print("Executing simobserve command...")
#        casatasks.simobserve(project = project_name, skymodel = "/groups/astro/andreask/radmc/"+path+"/saved_fits/"+image_name+".fits",
#                setpointings = True, direction = "J2000 04h04m43.07s 26d18m56.4s", obsmode = "int", totaltime = "1800s", antennalist = alma_config+".cfg", thermalnoise = "")
#        if verbose: print("Executing simanalyze command... Using threshold "+threshold+" for "+str(niter)+" iterations...")
#
#        casatasks.simanalyze(project = project_name, image = True, vis = project_name+"."+alma_config+".ms", imsize = [npix,npix], niter = niter, threshold=threshold, 
#                                weighting="natural", analyze = True, showuv = False, showresidual = True, showconvolved = True, graphics = "both", verbose = True, overwrite = True)
#        os.chdir(root)
#        if verbose: print("Succesfully ran the commands!")
#    except:
#        os.chdir(root)
#        raise ValueError("Could not create CASA image. See CASA log error message for details...")

def run_simalma(image_name, path, antennalist=['alma.cycle7.8.cfg', 'alma.cycle7.5.cfg'], totaltime=['15h', '3h'], pwv=0.5, threshold="4mJy", niter=5000, verbose=True):
    project_name = get_casa_project_name(image_name)
    root = os.getcwd()

    data, header = fits.getdata(path+"/saved_fits/"+image_name+".fits", header=True)

    map_x = header["NAXIS1"] * header["CDELT1"] * 3600; map_y = header["NAXIS2"] * header["CDELT2"] * 3600

    try:
        os.chdir(path+"/casa_projects")
        casatasks.simalma(project=project_name, dryrun=False, skymodel="../saved_fits/"+image_name+".fits", setpointings=True, integration="4500s", mapsize=[str(map_x)+'arcsec',str(map_y)+'arcsec'],
                          antennalist=antennalist, hourangle='transit', totaltime=totaltime, pwv=pwv, image=True, imsize=[header["NAXIS1"], header["NAXIS2"]], niter=niter, 
                          threshold=threshold, graphics='file', verbose=True, overwrite=True)
        os.chdir(root)
    except:
        os.chdir(root)
        raise OSError("simalma failed. See CASA logs for details.")

    # After running we should export to FITS file
    if verbose: print("Exporting CASA files to .fits")
    if len(antennalist) > 1:
        if verbose: print("Outputting combined antenna image as FITS...")
        config_str = "_".join(['alma.cycle7.8.cfg', 'alma.cycle7.5.cfg']).replace("alma.cycle","").replace(".cfg","")
        casatasks.exportfits(imagename=path+"/casa_projects/"+project_name+"/"+project_name+".concat.image.pbcor", 
                             fitsimage=path+"/saved_fits/simalma_combined"+config_str+"_"+image_name+".fits")
    else:
        if verbose: print(f"Outputting image for configuration {antennalist[0]}")
        casatasks.exportfits(imagename=path+"/casa_projects/"+project_name+"/"+project_name+"."+antennalist[0]+".image.pbcor", 
                             fitsimage=path+"/saved_fits/simalma_"+antennalist[0]+"_"+image_name+".fits")

def get_stats(path, image_name, antennalist):
    '''
    Get the statistical information about the generated image from CASA.

    Parameters:
        image_name: (string) Image name generated for the calculated image
        alma_config: (string) ALMA configuration file to use with CASA
        path: (string) The path to the snapshot
        verbose: (bool, default=1) Report task activity
        pbcor: (bool) Whether to export the primary beam corrected image.
    
    Returns:
        stats (dict) - image statistics computed over given axes
    '''
    project_name = get_casa_project_name(image_name)
    if len(antennalist) > 1:
        # Output the combined image
        stats = casatasks.imstat(imagename=path+"/casa_projects/"+project_name+"/"+project_name+".concat.image.pbcor")
    else:
        # Output the single image
        stats = casatasks.imstat(imagename=path+"/casa_projects/"+project_name+"/"+project_name+"."+str(antennalist[0]).replace(".cfg","")+".noisy.image.pbcor")
    return stats

#def export_fits(image_name, alma_config, path, verbose, pbcor):
#    '''
#    Export CASA project to FITS file.
#
#    Parameters:
#        image_name: (string) Image name generated for the calculated image
#        alma_config: (string) ALMA configuration file to use with CASA
#        path: (string) The path to the snapshot
#        verbose: (bool, default=1) Report task activity
#        pbcor: (bool) Whether to export the primary beam corrected image.
#    '''
#    if verbose: print("Exporting CASA files to .fits")
#    project_name = get_casa_project_name(image_name)
#    if pbcor:
#        if verbose: print("Outputting primary beam corrected image as FITS...")
#        casatasks.exportfits(imagename=path+"/casa_projects/"+project_name+"/"+project_name+"."+alma_config+".image.pbcor", 
#                             fitsimage=path+"/saved_fits/simanalyze_"+alma_config+"_"+image_name+"pbcor.fits")
#    else:
#        if verbose: print("Outputting image as FITS without correcting for primary beam...")
#        casatasks.exportfits(imagename=path+"/casa_projects/"+project_name+"/"+project_name+"."+alma_config+".image", 
#                             fitsimage=path+"/saved_fits/simanalyze_"+alma_config+"_"+image_name+".fits")
#    if verbose: print("Finished exporting CASA files to .fits!")
#
#def get_stats(image_name, alma_config, path, pbcor):
#    '''
#    Get the statistical information about the generated image from CASA.
#
#    Parameters:
#        image_name: (string) Image name generated for the calculated image
#        alma_config: (string) ALMA configuration file to use with CASA
#        path: (string) The path to the snapshot
#        verbose: (bool, default=1) Report task activity
#        pbcor: (bool) Whether to export the primary beam corrected image.
#    
#    Returns:
#        stats (dict) - image statistics computed over given axes
#    '''
#    project_name = get_casa_project_name(image_name)
#    if pbcor: stats = casatasks.imstat(imagename=path+"/casa_projects/"+project_name+"/"+project_name+"."+alma_config+".image.pbcor")
#    else: stats = casatasks.imstat(imagename=path+"/casa_projects/"+project_name+"/"+project_name+"."+alma_config+".image")
#    return stats

def get_view(view=None, inclination=None, rotangle=None, verbose=1):
    '''
    Get the necessary information to generate or load an image made with this package.

    Parameters:
        view: (string, optional) Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination: (float, optional) The inclination of the viewed image
        rotangle: (float, optional) The clockwise rotation around the z-axis of the system
        verbose: (bool, default=1) Report task activity

    Returns:
        view_str (string) for file handling, inclination (float), rotangle (float) for image generation
    '''
    view_error_msg = "View must be either string: 'face-on', 'edge-on-A' or 'edge-on-B', or provide floats: 'inclination' and 'rotangle'."
    # Check whether view or inclination+rotangle is given.
    # If both are given, inclination+rotangle override view
    # Should return inclination, rotangle and the "view" string for saving...

    # Check first if we need to turn a "view" into inclination+rotangle
    if isinstance(view, str):
        if inclination and rotangle:
            if verbose: print("Both 'view', 'inclination' and 'rotangle' keywords given. 'view' keyword is ignored.")
            if (inclination, rotangle) == (0,0):
                view_str = "face-on"
            elif (inclination, rotangle) == (90,0):
                view_str = "edge-on-A"
            elif (inclination, rotangle) == (90,90):
                view_str = "edge-on-B"
        elif view.lower() == "face-on":
            #pv = spin_vector #LEGACY
            #nv = plane_vector2 #LEGACY
            inclination = 0
            rotangle = 0

            view_str = "face-on"
        elif view.lower() == "edge-on-a":
            #pv = - plane_vector2 #LEGACY
            #nv = spin_vector #LEGACY
            inclination = 90
            rotangle = 0

            view_str = "edge-on-A"
        elif view.lower() == "edge-on-b":
            #pv = plane_vector1 #LEGACY
            #nv = spin_vector #LEGACY
            inclination = 90
            rotangle = 90

            view_str = "edge-on-B"
    elif inclination is not None and rotangle is not None:
        inclination = inclination % 360
        rotangle = rotangle % 360
        if (inclination, rotangle) == (0,0):
                view_str = "face-on"
        elif (inclination, rotangle) == (90,0):
            view_str = "edge-on-A"
        elif (inclination, rotangle) == (90,90):
            view_str = "edge-on-B"
        else:
            # If inclination and rotangle given, we should name it
            view_str = "incl"+str(inclination)+"-angle"+str(rotangle)
    else:
        raise ValueError(view_error_msg)
    
    return view_str, inclination, rotangle

def calc_view_vectors(isink, iout, view=None, inclination=None, rotangle=None):
    '''
    Calculate the coordinate system for the given viewing parameters of the system.

    Parameters:
        isink (int):  The sink ID
        iout (int):  The snapshot ID
        view (string, optional): Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination (float, optional): The inclination of the viewed image
        rotangle (float, optional): The clockwise rotation around the z-axis of the system

    Returns:
        (east_vector, north_vector, normal_vector) (ndarrays): The x-y image plane and the vector normal to that plane needed for image generation
    '''
    ## load spin vector and coordinate basis for the disk from snapshot
    #spin_vector, plane_vector1, plane_vector2 = calc_coord_basis(isink, iout)
#
    ## Use face-on as the zero point
    ## In the old notation pv = vector that points towards the "camera", nv = vector that points up in the image
#
    ## Let's check if the "view" keyword has been given, otherwise we should use inclination and rotangle
    #_, inclination, rotangle = get_view(view, inclination, rotangle)
    #    
#
    ## Calculate based on the given inclination and rotation...
    #incl_rad = np.deg2rad(inclination)
    #angl_rad = np.deg2rad(rotangle)
#
    ## Since the vector pointing towards us is the spin vector, we want to rotate the pv1-pv2 plane 
    ## We first rotate in the plane_vector1-plane_vector2 axis
    #rot_plane_vector1 = plane_vector1 * np.cos(angl_rad) + plane_vector2 * np.sin(angl_rad)
    #rot_plane_vector2 = np.cross(spin_vector, rot_plane_vector1)
#
    ## Then we incline the system, creating a new projection vector and north vector
    ## We want to rotate in the L-p2 plane. We calculate the rotated plane by
    #east_vector = rot_plane_vector1
    #north_vector = np.cos(incl_rad) * rot_plane_vector2 + np.sin(incl_rad) * spin_vector
    #normal_vector = np.cross(east_vector, north_vector)
#
    #return east_vector, north_vector, normal_vector # x,y,z

    # load spin vector and coordinate basis for the disk from snapshot
    spin_vector, x_vector, y_vector = calc_coord_basis(isink, iout) # # z, x, y

    # Use face-on as the zero point
    # In the old notation pv = vector that points towards the "camera", nv = vector that points up in the image

    # Let's check if the "view" keyword has been given, otherwise we should use inclination and rotangle
    _, inclination, rotangle = get_view(view, inclination, rotangle)
        

    # Calculate based on the given inclination and rotation...
    incl_rad = np.deg2rad(inclination)
    angl_rad = np.deg2rad(rotangle)

    # Since the vector pointing towards us is the spin vector, we want to rotate the pv1-pv2 plane 
    # We first rotate in the plane_vector1-plane_vector2 axis
    rot_x = x_vector * np.cos(angl_rad) + y_vector * np.sin(angl_rad)
    rot_y = -x_vector * np.sin(angl_rad) + y_vector * np.cos(angl_rad)

    # Then we incline the system, creating a new projection vector and north vector
    # We want to rotate in the L-p2 plane. We calculate the rotated plane by
    east_vector = rot_x
    north_vector = rot_y * np.cos(incl_rad) + spin_vector * np.sin(incl_rad)
    normal_vector = -rot_y * np.sin(incl_rad) + spin_vector * np.cos(incl_rad)

    return east_vector, north_vector, normal_vector # x,y,z

def run_osyris2dslab(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, verbose=1):
    view_str, inclination, rotangle = get_view(view=view, inclination=inclination, rotangle=rotangle, verbose=verbose)
    path = goto_folder(isink, iout)
    filename = "osyris2dslab-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)+".dat"

    # Check if the file exists
    if os.path.exists(path+"/saved_values/"+filename):
        if verbose: print("The data for this configuration already exists. Loading...")
        with open(path+"/saved_values/"+filename, 'rb') as fp:
            osyris_data = pickle.load(fp)
    else:
        if verbose: print("The data for this configuration doesn't exist already. Generating...")
        # Load in RADMC-3D data
        os.chdir(path)

        grid = analyze.readGrid()
        data = analyze.readData(gdens=True, dtemp=True, ispec='co', grid=grid)
        temp = data.dusttemp.flatten()

        os.chdir("../../")

        # Load in RAMSES data
        ramses = nexus.dataclass()
        if isinstance(isink, float):
            sink_id, level = str(isink).split(".")
            sink_id = int(sink_id); level = int(level)
            print(f"You have specified isink as a float, interpreted as sink ID {sink_id} with max level {level}")
            error_msg = f"The data directory for sink {sink_id} with max level {level} hasn't been configured. Please specify the data directory in 'sink_config.py'." # Only necessary if not configured!
        else:
            sink_id = isink
            # Level for max resolution - hardcoded FIXME
            level = 20
            error_msg = f"The data directory for sink {sink_id} hasn't been configured. Please specify the data directory in 'sink_config.py'." # Only necessary if not configured!
        try:
            datadir = sink_dirs[str(isink)]
        except:
            raise ValueError(error_msg)
        print(datadir)

        ramses.load(snap = iout, io = 'RAMSES', path = datadir, sink_id=sink_id, verbose=verbose, dtype = 'float64')

        # Recalculate the coordinates and velocities
        ramses.calc_trans_xyz()
        ramses.vx, ramses.vy, ramses.vz = ramses.trans_vrel

        # Define minimum resolution for simulation
        dx_max = (4 * unit.pc / 2**level).to(unit.cm).value

        # Convert RADMC-3D and RAMSES data to integer coordinates

        coords_cm = ramses.rel_xyz * (4 * unit.pc).to(unit.cm) # (3 * ncells) [x,y,z]
        radmc3d_coords = np.array([grid.x[grid.isLeaf], grid.y[grid.isLeaf], grid.z[grid.isLeaf]]) # (3 * ncells) [x,y,z] - already in cm

        icoords_ramses = np.round(coords_cm.value / dx_max).astype(np.int64)
        icoords_radmc = np.round(radmc3d_coords / dx_max).astype(np.int64)

        # Assign ID to each cell in RAMSES data
        idx = np.arange(0, len(icoords_ramses[0,:]))

        # Create dictionary and fill each integer coordinate with a corresponding RAMSES ID
        cell_dict = {}

        if verbose: print("Creating hash table for RADMC-3D and RAMSES coordinates...")
        for i in range(icoords_ramses.shape[1]):
            cell_dict[tuple(icoords_ramses[:,i])] = idx[i]

        # Add a temperature array to the RAMSES class and fill it out 
        ramses.T = np.empty_like(ramses.mhd["d"])

        for i in range(icoords_radmc.shape[1]):
            ientry = cell_dict[tuple(icoords_radmc[:,i])]
            ramses.T[ientry] = temp[i]

        # We calculate the density-weighted temperature too
        ramses.Td = ramses.mhd["d"] * ramses.T

        # Run Osyris2Dslab
        east_vector, north_vector, normal_vector = calc_view_vectors(isink, iout, inclination=inclination, rotangle=rotangle)

        custom_viewpoint = {'new_x': east_vector, 
                            'new_y': north_vector, 
                            'view_vector': normal_vector}

        if verbose: print("Creating the 2D slab with Osyris. This might take a while...")
        ramses.osyris2Dslab(variables = ['d', 'Td', 'T', 'vx', 'vy', 'vz'], 
                        viewpoint=custom_viewpoint, 
                        resolution=resolution, 
                        view = width, 
                        dz = dz,
                        weights=[None, None, 'mass', 'mass', 'mass', 'mass'])

        # After running we now want to save all the calculated osyris2Dslab values (those are of interest to us)
        # But first we want to convert them to CGS units (except "T" which is already in K)
        ramses.osyris_ivs["data1"]["d"]  *= ramses.d_cgs # g/cm^3
        ramses.osyris_ivs["data1"]["Td"] *= ramses.d_cgs # g/cm^3 * K
        ramses.osyris_ivs["data1"]["vx"] *= ramses.v_cgs # cm/s
        ramses.osyris_ivs["data1"]["vy"] *= ramses.v_cgs # cm/s
        ramses.osyris_ivs["data1"]["vz"] *= ramses.v_cgs # cm/s

        # Rename it for easy access
        osyris_data = ramses.osyris_ivs["data1"]

        # Save using pickle
        with open(path+"/saved_values/"+filename, 'wb') as fp:
            pickle.dump(osyris_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    return osyris_data

#### ------------------------
#### Image creation functions
#### ------------------------

# Create RADMC image
def create_image(isink, iout, npix=800, wav=None, sizeau=1000, setthreads=4, view=None, 
                 inclination=None, rotangle=None, imolspec=1, iline=None, widthkms=None, linenlam=None, 
                 nostar=True, tracetau=False):
    '''
    This function calls the RADMC-3D create image function, for either a single wavelength, 
    a transition and for a given width around it, or traces the optical depth and outputs a '.out' file.

    Parameters:
        isink: (int) The sink ID
        iout: (int) The snapshot ID
        npix: (int) Number of pixels on the rectangular images
        wav: (float) Wavelength of the image in micron - Ignore if doing line imaging
        sizeau: (float) Diameter of the image in au
        setthreads: (int) Number of threads for RADMC-3D to use for image generation
        view: (string, optional) Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination: (float, optional) The inclination of the viewed image
        rotangle: (float, optional) The clockwise rotation around the z-axis of the system
        imolspec: (int, default=1) Molecule to use from 'lines.inp' file
        iline: (int, optional) Line transition index
        widthkms: (float, optional) Width of the frequency axis of the channel maps
        linenlam: (int, optional) Number of wavelengths to calculate images at
        nostar: (bool, default=True) If True the calculated images will not contain stellar emission
        tracetau: (bool, default=False) If True returns the traced optical depth instead of emission
    '''
    try:
        # Log the current working directory before we move around
        org_path = os.getcwd()

        east_vector, north_vector, normal_vector = calc_view_vectors(isink, iout, view=view, inclination=inclination, rotangle=rotangle)

        incl, phi      = get_projang(normal_vector)
        position_angle = get_posang(north_vector, normal_vector)
        
        # Check if we're in the necessary folder
        path = goto_folder(isink, iout)
        check_in_folder(path)

        # Check if the inputs for molecular transitions are given
        if np.count_nonzero([iline, widthkms, linenlam]) == 3: # If all inputs given, do molecular image
            print(f"Creating molecular line images for molecule {imolspec} transition {iline} with {linenlam} different wavelengths at a width of {widthkms} km/s")
            image.makeImage(npix=npix, incl=incl, phi=phi, sizeau=sizeau, setthreads=setthreads, posang=position_angle, 
                            imolspec=imolspec, iline=iline, widthkms=widthkms, linenlam=linenlam, nostar=nostar, tracetau=tracetau,
                            doppcatch=True, exe = '/lustre/astro/troels/radmc3d/bin/radmc3d')
        
        elif np.count_nonzero([iline, widthkms, linenlam]) < 3 and np.count_nonzero([iline, widthkms, linenlam]) > 0:
            raise ValueError("To do molecular line images, 'iline', 'widthkms' and 'linenlam' must all be given.")
        
        else: # Do single wavelength image
            image.makeImage(npix=npix, incl=incl, phi=phi, wav=wav, sizeau=sizeau,
                            setthreads=setthreads, posang=position_angle, nostar=nostar, tracetau=tracetau,
                            doppcatch=True, exe = '/lustre/astro/troels/radmc3d/bin/radmc3d')
            
        # Leave the folder
        os.chdir(org_path)
    except: # Go to original directory if program fails.
        print("Program failed?")
        os.chdir(org_path)
        

def single_wavelength_image(isink, iout, npix, wav, sizeau, setthreads, view=None, inclination=None, rotangle=None, dpc=None, nostar=True,
                            writefits=False, casa=False, antennalist=None, totaltime=None, threshold="4mJy", niter=5000, verbose=1):
    '''
    This function calls the create image function for a single wavelength.

    Parameters:
        isink: (int) The sink ID
        iout: (int) The snapshot ID
        npix: (int) Number of pixels on the rectangular images
        wav: (float) Wavelength of the image in micron - Ignore if doing line imaging
        sizeau: (float) Diameter of the image in au
        setthreads: (int) Number of threads for RADMC-3D to use for image generation
        view: (string, optional) Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination: (float, optional) The inclination of the viewed image
        rotangle: (float, optional) The clockwise rotation around the z-axis of the system
        dpc: (float) Distance to source in pc
        nostar: (bool, default=True) If True the calculated images will not contain stellar emission
        tracetau: (bool, default=False) If True returns the traced optical depth instead of emission

    Returns:
        radmc3dPy.image.radmc3dImage class
    '''
    # Check if all the folders, that need to exist, do exist
    path = goto_folder(isink, iout)
    check_folders(path)

    # Check if CASA is given the correct number of pixels
    if casa:
        optimal_npix = su.getOptimumSize(npix)
        if optimal_npix != npix: print("For the cleaning algorithm, CASA demands 'npix' is factorisable by 2,3,5 only. Setting 'npix' = "+str(optimal_npix))
        npix = optimal_npix
        if verbose: print("Set new 'npix' = "+str(npix))

    if casa and not writefits:
        if verbose: print("Notice: 'writefits' must be set to 'True' to make a CASA image. Setting 'writefits' = True")
        writefits = True

    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)

    # Construct the file name
    fname = "image-"+view_str+"-npix"+str(npix)+"-singlewav-"+str(sizeau)+"au-"+str(int(wav))+"mu"

    # Next if it already existed, we'll check if we've already created the image before
    if not os.path.exists(path+"/saved_images/"+fname+".out"):
        if verbose: print("An image of this configuration doesn't exist. Creating it...")
        create_image(isink, iout, npix=npix, wav=wav, sizeau=sizeau, setthreads=setthreads, nostar=nostar, 
                     inclination=inclination, rotangle=rotangle)
        # We know that the image is saved 'image.out', but we want to move it and rename it
        # Check if the folder "saved_images" exists
        shutil.move(path+"/image.out", path+"/saved_images/"+fname+".out")
    else:
        if verbose: print("An image for this configuration already exists. Loading...")
    # Load the image using RADMC-3D
    if (casa and not os.path.exists(path+"/saved_fits/"+fname+".fits")) or not casa:
        img = image.readImage(fname=path+"/saved_images/"+fname+".out")

        # Export to fits file for easy usage
        if writefits and dpc:
            img.writeFits(fname=path+"/saved_fits/"+fname+".fits", dpc=dpc, coord="04h04m43.08s 26d18m56.12s", casa=True, nu0=img.freq[0])
        elif writefits and dpc is None:
            raise ValueError("FITS file not created. Missing argument 'dpc'.")
        
        if not casa:
            # Fix orientation of image
            img.image = np.rot90(img.image, k=1)

            # If a distance is given, we'd like to convert the units to Jy/px
            # Conversion from erg/s/cm/cm/Hz/ster to Jy/pixel (from radmc3dPy)
            if dpc:
                if verbose: print("Distance to source given. Converting flux unit to Jy/px...")
                conv = img.sizepix_x * img.sizepix_y / (dpc * cnst.pc.cgs.value)**2. * 1e23
                img.image *= conv
    
    if casa:
        if verbose: print("Creating CASA image...")
        if len(antennalist) > 1:
            config_str = "combined"+"_".join(['alma.cycle7.8.cfg', 'alma.cycle7.5.cfg']).replace("alma.cycle","").replace(".cfg","")
        else:
            config_str = antennalist[0]

        # If it exists we load it. Otherwise run the simalma command
        if os.path.exists(path+"/saved_fits/simalma_"+config_str+"_"+fname+".fits"): 
            print("The requested image has already been generated. Loading...")
        else: 
            run_simalma(fname, path, antennalist=antennalist, totaltime=totaltime, pwv=0.5, threshold=threshold, niter=niter, verbose=verbose)
        
        # Load it in using our class
        img = casaImageClass(isink, iout, fname, dpc, antennalist)

    return img

def molecular_lines_image(isink, iout, npix, sizeau, setthreads, iline, widthkms, linenlam, imolspec=1, view=None, inclination=None, 
                          rotangle=None, dpc=None, beam=None, nostar=True, writefits=False, casa=False, antennalist=None, totaltime=None,
                          threshold="4mJy", niter=5000, verbose=1):
    '''
    This function calls the create image function for an image with multiple wavelengths around a given molecular line transition..

    Parameters:
        isink: (int) The sink ID
        iout: (int) The snapshot ID
        npix: (int) Number of pixels on the rectangular images
        wav: (float) Wavelength of the image in micron - Ignore if doing line imaging
        sizeau: (float) Diameter of the image in au
        setthreads: (int) Number of threads for RADMC-3D to use for image generation
        iline: (int, optional) Line transition index
        widthkms: (float, optional) Width of the frequency axis of the channel maps
        linenlam: (int, optional) Number of wavelengths to calculate images at
        imolspec: (int, default=1) Molecule to use from 'lines.inp' file
        view: (string, optional) Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination: (float, optional) The inclination of the viewed image
        rotangle: (float, optional) The clockwise rotation around the z-axis of the system
        dpc: (float, optional) Distance to source in pc
        beam: (tuple, optional) FWHM beam in arcseconds (beam_x, beam_y)
        nostar: (bool, default=True) If True the calculated images will not contain stellar emission
        writefits: (bool, default=False) Convert the RADMC-3D image to a .fits file
        casa: (bool, default=False) Whether to put the image through the CASA simobserve+simanalyze pipeline. Automatically sets `writefits` = True
        antennalist: (string or list, optional) ALMA configuration file(s) to use with simalma.
        totaltime: (string or list, optional) Total integration time for each ALMA configuration. Must be same length as `antennalist`
        threshold: (string, default="0.1mJy") The threshold to which the image should be cleaned
        niter: (int, default=10000) The number of iterations the cleaning algorithm will run before forcefully stopping.
        verbose: (bool, default=1) Report task activity

    Returns:
        radmc3dPy.image.radmc3dImage class
    '''
    # Error handling
    if beam and casa:
        raise ValueError("'beam' is not needed to run CASA, please either specify 'beam' to convolve with a Gaussian beam, or set 'casa' = True to run the CASA pipeline.")
    # Check if CASA is given the correct number of pixels
    if casa:
        optimal_npix = su.getOptimumSize(npix)
        if optimal_npix != npix: print("For the cleaning algorithm, CASA demands 'npix' is factorisable by 2,3,5 only. Setting 'npix' = "+str(optimal_npix))
        npix = optimal_npix
        if verbose: print("Set new 'npix' = "+str(npix))
    if casa and not writefits:
        if verbose: print("Notice: 'writefits' must be set to 'True' to make a CASA image. Setting 'writefits' = True")
        writefits = True

    # Check if all the folders, that need to exist, do exist
    path = goto_folder(isink, iout)
    check_folders(path)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)

    # Let's read in which molecule is of interest here
    molecules = np.loadtxt(path+"/lines.inp", skiprows=2, dtype=str)[:,0]
    molecule_name = molecules[imolspec-1] # The imolspec is 1-indexed

    # Save fname for later use
    fname = "image-"+molecule_name+"-"+view_str+"-npix"+str(npix)+"-"+str(sizeau)+"au-transition"+str(iline)+"-widthkms"+str(widthkms)+"-lines"+str(linenlam)

    # Next if it already existed, we'll check if we've already created the image before
    if not os.path.exists(path+"/saved_images/"+fname+".out"):
        if verbose: print("An image of this configuration doesn't exist. Creating it...")
        create_image(isink, iout, npix=npix, sizeau=sizeau, setthreads=setthreads, inclination=inclination, rotangle=rotangle,
                     imolspec=imolspec, iline=iline, widthkms=widthkms, linenlam=linenlam, nostar=nostar)
        # We know that the image is saved 'image.out', but we want to move it and rename it
        shutil.move(path+"/image.out", path+"/saved_images/"+fname+".out")
    else:
        if verbose: print("An image for this configuration already exists. Loading...")

    # We want to load in the image file in case the output isn't casa or if the FITS file for casa doesn't exist
    if (casa and not os.path.exists(path+"/saved_fits/"+fname+".fits")) or not casa:
        img = image.readImage(fname=path+"/saved_images/"+fname+".out")

        # Now that we have the image loaded in, we want to know whether we should perform operations on it...
        # Save the fits file?
        if writefits and dpc:
            if verbose: print("Writing image to fits file...")
            if img.nfreq % 2 == 0:
                nu0 = img.freq[img.nfreq//2-1]/2 + img.freq[img.nfreq//2]/2
            else:
                nu0 = img.freq[img.nfreq//2]

            bandwidth = np.abs(img.freq[0] - img.freq[1]) * 1e-6 # Calculate the bandwidth (assumed the same between freqs)
            if os.path.exists(path+"/saved_fits/"+fname+".fits"):
                if verbose: print("FITS file already exists. If you want to overwrite, please delete this.")
            else: img.writeFits(path+"/saved_fits/"+fname+".fits", dpc=dpc, coord="04h04m43.07s 26d18m56.4s", bandwidthmhz=bandwidth, casa=True, nu0=nu0)
        elif writefits and not dpc:
            raise ValueError("FITS file not created. Missing argument 'dpc'.")
        
        # If a distance is given, we'd like to convert the units to Jy/px
        # Conversion from erg/s/cm/cm/ster to Jy/pixel (from radmc3dPy)
        if dpc:
            if verbose: print("Distance to source given. Converting flux unit to Jy/px...")
            conv = img.sizepix_x * img.sizepix_y / (dpc * cnst.pc.cgs.value)**2. * 1e23
            img.image *= conv

        # Fix the orientation of the image for plotting purposes
        img.image = np.rot90(img.image, k=1)

        # Convolve with beam?
        if beam: 
            if verbose: print("Convolving image with given beam...")
            img = convolve_beam(img, beam, dpc)

    if casa:
        if verbose: print("Creating CASA image...")
        if len(antennalist) > 1:
            config_str = "combined"+"_".join(['alma.cycle7.8.cfg', 'alma.cycle7.5.cfg']).replace("alma.cycle","").replace(".cfg","")
        else:
            config_str = antennalist[0]

        # If it exists we load it. Otherwise run the simalma command
        if os.path.exists(path+"/saved_fits/simalma_"+config_str+"_"+fname+".fits"): 
            print("The requested image has already been generated. Loading...")
        else: 
            run_simalma(fname, path, antennalist=antennalist, totaltime=totaltime, pwv=0.5, threshold=threshold, niter=niter, verbose=verbose)
        
        # Load it in using our class
        img = casaImageClass(isink, iout, fname, dpc, antennalist)

        # If one of the FITS file already exists, it should be possible to load in 
        #if os.path.exists(path+"/saved_fits/simanalyze_"+alma_config+"_"+fname+".fits") and not pbcor:
        #    if verbose: print("The CASA fits file already exists. Loading it in...")
        #elif os.path.exists(path+"/saved_fits/simanalyze_"+alma_config+"_"+fname+"pbcor.fits") and pbcor:
        #    if verbose: print("The CASA fits file already exists. Loading it in...")
        ## This should work cause if one of the FITS files exists, there should be a CASA folder with the data to quickly generate another
        #elif os.path.exists(path+"/saved_fits/simanalyze_"+alma_config+"_"+fname+".fits") or os.path.exists(path+"/saved_fits/simanalyze_"+alma_config+"_"+fname+"pbcor.fits"):
        #    export_fits(fname, alma_config, path, verbose, pbcor)
        #else: 
        #    run_casa(fname, npix, alma_config, path, verbose, threshold=threshold, niter=niter) # Run the CASA commands
        #    export_fits(fname, alma_config, path, verbose, pbcor)
        #
        ## Load in the fits file that CASA has created...
        #if pbcor:
        #    if verbose: print("Loading in the primary beam corrected image...")
        #    data, header = fits.getdata(path+"/saved_fits/simanalyze_"+alma_config+"_"+fname+"pbcor.fits", header = True)
        #else:
        #    if verbose: print("Loading in the non-primary beam corrected image...")
        #    data, header = fits.getdata(path+"/saved_fits/simanalyze_"+alma_config+"_"+fname+".fits", header = True)
        #
        #stats = get_stats(fname, alma_config, path, pbcor)

        # Let's create an image class so we can handle the moment map creation with the same script
        #casa_img = np.transpose(data[0,:,:,:], (1, 2, 0))
        #x_coord = (np.linspace(header["CRVAL1"] - (header["NAXIS1"]/2-1) * header["CDELT1"], (header["NAXIS1"]/2-1) * header["CDELT1"] + header["CRVAL1"], header["NAXIS1"]) - header["OBSRA"]) * np.pi/180 * 140 * unit.pc.to(unit.cm)
        #y_coord = (np.linspace(header["CRVAL2"] - (header["NAXIS2"]/2-1) * header["CDELT2"], (header["NAXIS2"]/2-1) * header["CDELT2"] + header["CRVAL2"], header["NAXIS2"]) - header["OBSDEC"]) * np.pi/180 * 140 * unit.pc.to(unit.cm)
        #pixsize_x = np.abs(header["CDELT1"] * np.pi/180 * 140 * unit.pc.to(unit.cm))
        #pixsize_y = np.abs(header["CDELT2"] * np.pi/180 * 140 * unit.pc.to(unit.cm))
        #image_freqs = np.linspace(start=header["CRVAL3"], stop=20*header["CDELT3"]+header["CRVAL3"], num=header["NAXIS3"])
        #casa_beam = (header["BMAJ"]/header["CDELT1"], header["BMIN"]/header["CDELT2"], header["BPA"]) # beam size in px
        #casa_rms = stats["rms"]
        #img = casaImageClass(casa_image=casa_img, x=x_coord[::-1], y=y_coord, nx=header["NAXIS1"], ny=header["NAXIS2"], sizepix_x=pixsize_x, sizepix_y=pixsize_y, nfreq=header["NAXIS3"], freqs=image_freqs, beam=casa_beam, rms=casa_rms)

    return img

def create_moment_map(isink, iout, npix, sizeau, setthreads, iline, widthkms, linenlam, imolspec=1, moment=0, view=None, inclination=None, rotangle=None, 
                      dpc=None, beam=None, verbose=1, nostar=True, casa=False, writefits=False, antennalist=None, totaltime=None, threshold="4mJy", niter=5000, return_img=False):
    '''
    Creates a moment map for a multi-wavelength image, based on the function in radmc3dPy, but changed to fit the data structure of this package.
    
    Parameters:
        isink: (int) The sink ID
        iout: (int) The snapshot ID
        npix: (int) Number of pixels on the rectangular images
        wav: (float) Wavelength of the image in micron - Ignore if doing line imaging
        sizeau: (float) Diameter of the image in au
        setthreads: (int) Number of threads for RADMC-3D to use for image generation
        iline: (int, optional) Line transition index
        widthkms: (float, optional) Width of the frequency axis of the channel maps
        linenlam: (int, optional) Number of wavelengths to calculate images at
        imolspec: (int, default=1) Molecule to use from 'lines.inp' file
        moment: (int, default=0) Which moment map to generate
        view: (string, optional) Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination: (float, optional) The inclination of the viewed image
        rotangle: (float, optional) The clockwise rotation around the z-axis of the system
        dpc: (float, optional) Distance to source in pc
        beam: (tuple, optional) FWHM beam in arcseconds (beam_x, beam_y)
        verbose: (bool, default=1) Report task activity
        nostar: (bool, default=True) If True the calculated images will not contain stellar emission
        casa: (bool, default=False) Whether to put the image through the CASA simobserve+simanalyze pipeline. Automatically sets `writefits` = True
        writefits: (bool, default=False) Convert the RADMC-3D image to a .fits file
        alma_config: (string, optional) ALMA configuration file to use with CASA pipeline.
        threshold: (string, default="1e-6Jy") The threshold to which the image should be cleaned
        niter: (int, default=500000) The number of iterations the cleaning algorithm will run before forcefully stopping.
        pbcor: (bool, default=True) Whether to export the primary beam corrected image.
        return_img: (bool, default=False) Whether to also return the radmc3dPy.image.radmc3dImage class

    Returns:
        (ndarray) Moment map of size-length `npix`. If `return_img` = True, it also returns the radmc3dPy.image.radmc3dImage class (ndarray, radmc3dPy.image.radmc3dImage class)
    '''
    
    # Check if all the folders, that need to exist, do exist
    path = goto_folder(isink, iout)
    check_folders(path)

    # Error handling
    if linenlam < 2:
        raise ValueError("Cannot create a moment map from a singular wavelength. 'linenlam' must be > 1.")
    
    img = molecular_lines_image(isink=isink, iout=iout, npix=npix, sizeau=sizeau, setthreads=setthreads, iline=iline, widthkms=widthkms, linenlam=linenlam, 
                                imolspec=imolspec, view=view, inclination=inclination, rotangle=rotangle, nostar=nostar, verbose=verbose, beam=beam, dpc=dpc, 
                                writefits=writefits, casa=casa, antennalist=antennalist, totaltime=totaltime, threshold=threshold, niter=niter)

    # Calculate rest wavelength of the image
    if img.nfreq % 2 == 0:
        nu0 = img.freq[img.nfreq//2-1]/2 + img.freq[img.nfreq//2]/2
    else:
        nu0 = img.freq[img.nfreq//2]

    # This part of the program is now appropriated from radmc3dPy !!
    # Calculate velocity field
    v_kms = cnst.c.value * (nu0 - img.freq) / nu0 / 1e3

    vmap = np.zeros([img.nx, img.ny, img.nfreq], dtype=np.float64)
    for ifreq in range(img.nfreq):
        vmap[:, :, ifreq] = v_kms[ifreq]

    # Now calculate the moment map
    y = img.image * (vmap**moment)

    dum = (vmap[:, :, 1:] - vmap[:, :, :-1]) * (y[:, :, 1:] + y[:, :, :-1]) * 0.5

    mmap = dum.sum(2)

    if moment > 0:
        y = img.image
        dum0 = (vmap[:, :, 1:] - vmap[:, :, :-1]) * (y[:, :, 1:] + y[:, :, :-1]) * 0.5
        
        mmap0 = dum0.sum(2)
        mmap = mmap / mmap0

    if return_img: 
        return mmap, img
    else: 
        return mmap

def create_sed(isink, iout, npix, sizeau, setthreads, dpc, view=None, inclination=None, rotangle=None, sed_points=20, subtract_isrf=True, verbose=1):
    '''
    Create the Spectral Energy Distribution (SED) for the given sink and output

    Parameters:
        isink: (int) The sink ID
        iout: (int) The snapshot ID
        npix: (int) Number of pixels on the rectangular images
        sizeau: (float) Diameter of the image in au
        setthreads: (int) Number of threads for RADMC-3D to use for image generation
        dpc: (float, optional) Distance to source in pc
        view: (string, optional) Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination: (float, optional) The inclination of the viewed image
        rotangle: (float, optional) The clockwise rotation around the z-axis of the system
        sed_points: (int, default=20) How many data points to make for the SED
        subtract_isrf: (bool, default=True) Whether to subtract the interstellar radiation field (background emission)
        verbose: (bool, default=1) Report task activity
        

    Returns:
        freqs, fluxes (ndarray, ndarray) The frequencies used and the flux calculated at the given frequencies
    
    '''
    # Check if all the folders, that need to exist, do exist
    path = goto_folder(isink, iout)
    check_folders(path)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)
        
    # Handling of different cases
    # if SED exists but not ISRF subtracted
    if os.path.exists(path+"/saved_values/sed-"+view_str+"-"+str(sizeau)+"au.dat") and subtract_isrf:
        if verbose: print("The SED exists but is not ISRF subtracted. Performing subtraction and re-saving...")
        # Load non-subtracted ISRF
        freqs, fluxes = np.loadtxt(path+"/saved_values/sed-"+view_str+"-"+str(sizeau)+"au.dat", unpack=True)
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
        np.savetxt(path+"/saved_values/sed-"+view_str+"-"+str(sizeau)+"au-isrf_subtracted.dat", np.transpose([freqs, fluxes]))

    # Check if the values for the SED already exist
    elif os.path.exists(path+"/saved_values/sed-"+view_str+"-"+str(sizeau)+"au.dat") or os.path.exists(path+"/saved_values/sed-"+view_str+"-"+str(sizeau)+"au-isrf_subtracted.dat"):
        if verbose: print("The SED for this configuration already exists. Loading it in...")
        # Load in the SED data from the folder
        if subtract_isrf:
            freqs, fluxes = np.loadtxt(path+"/saved_values/sed-"+view_str+"-"+str(sizeau)+"au-isrf_subtracted.dat", unpack=True)
        else:
            freqs, fluxes = np.loadtxt(path+"/saved_values/sed-"+view_str+"-"+str(sizeau)+"au.dat", unpack=True)
    else: # Create SED
        if verbose: print("The SED for this configuration doesn't exist. Creating...")
        # Create wavelength spectrum
        light_speed = 299792458 * 1e6 # mum/s
        freqs = np.logspace(np.log10(30), np.log10(30000), sed_points) * 1e9 # Hz
        wavs = light_speed/freqs
        # Create the flux array
        fluxes = np.zeros_like(freqs)

        for i in range(len(freqs)):
            create_image(isink, iout, npix=npix, wav=wavs[i], sizeau=sizeau, setthreads=setthreads,
                         inclination=inclination, rotangle=rotangle)
            
            img = image.readImage(fname=path+"/image.out")

            # If a distance is given, we'd like to convert the units to Jy/px
            # Conversion from erg/s/cm/cm/ster to Jy/pixel (from radmc3dPy)
            if verbose: print("Distance to source given. Converting flux unit to Jy/px...")
            conv = img.sizepix_x * img.sizepix_y / (dpc * cnst.pc.cgs.value)**2. * 1e23
            img.image *= conv            

            # Save the flux in the array
            fluxes[i] = np.sum(img.image.flatten())
        
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
            np.savetxt(path+"/saved_values/sed-"+view_str+"-"+str(sizeau)+"au-isrf_subtracted.dat", np.transpose([freqs, fluxes]))
        else:
            np.savetxt(path+"/saved_values/sed-"+view_str+"-"+str(sizeau)+"au.dat", np.transpose([freqs, fluxes]))

    # Now we can output the frequencies and the fluxes
    return freqs, fluxes

def calc_column_density(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, verbose=1):
    '''
    Calculate the column density in the simulated RAMSES data.

    Parameters:
        isink: (int) The sink ID
        iout: (int) The snapshot ID
        resolution: (int) The resolution of the generated image
        width: (float) The width in AU of the generated image
        dz: (float) The depth in AU of the generated image
        view: (string, optional) Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination: (float, optional) The inclination of the viewed image
        rotangle: (float, optional) The clockwise rotation around the z-axis of the system
        verbose: (bool, default=1) Report task activity
    Returns:
        (ndarray) Calculated column densities
    '''
    # Check if all the folders, that need to exist, do exist
    path = goto_folder(isink, iout)
    check_folders(path)

    # We get the values from the osyris2Dslab calculation
    osyris_data = run_osyris2dslab(isink, iout, resolution=resolution, width=width, dz=dz, view=view, inclination=inclination, rotangle=rotangle, verbose=verbose)

    # We return the density
    return osyris_data["d"]

    #view_str, inclination, rotangle = get_view(view=view, inclination=inclination, rotangle=rotangle, verbose=verbose)
#
    ## Let's create the 2D slab of the column density with RAMSES and Nexus :)
    ## We've made a slight modification so that Nexus takes input "face-on", "edge-on-A", "edge-on-B"
    #if not os.path.exists(path+"/saved_values/column-density-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)+".dat"):
    #    print("The data for this configuration doesn't exist already. Generating...")
    #    ramses = nexus.dataclass()
    #    datadir = '/lustre/astro/troels/IMF_512_cores/christian/sink_'+str("{:03}".format(isink+1))+'/data'
    #    ramses.load(snap = iout, io = 'RAMSES', path = datadir, sink_id=isink, verbose=verbose, dtype = 'float64')
#
    #    # Use the calculated vectors for viewing the disk
    #    east_vector, north_vector, normal_vector = calc_view_vectors(isink, iout, inclination=inclination, rotangle=rotangle)
    #    # Create dictionary
    #    viewpoint = {'new_x': east_vector, 
    #                 'new_y': north_vector, 
    #                 'view_vector': normal_vector}
#
    #    print("Creating the 2D slab with Osyris. This might take a while...")
    #    ramses.osyris2Dslab(variables = ['d'], 
    #                        viewpoint=viewpoint, 
    #                        resolution=resolution, 
    #                        view = width, 
    #                        dz = dz,
    #                        weights=[None])
    #    surf_density = ramses.osyris_ivs['data1']['d'] * ramses.d_cgs
    #    # Save the data
    #    #np.savetxt(path+"/saved_values/column-density-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)+".dat", surf_density)
    #else:
    #    print("The data for this configuration already exists. Loading...")
    #    surf_density = np.loadtxt(path+"/saved_values/column-density-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)+".dat")
#
    #return surf_density

def calc_temperature(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, verbose=1):
    '''
    Calculate the temperature in the simulated RADMC-3D data and project onto RAMSES data.

    Parameters:
        isink: (int) The sink ID
        iout: (int) The snapshot ID
        resolution: (int) The resolution of the generated image
        width: (float) The width in AU of the generated image
        dz: (float) The depth in AU of the generated image
        view: (string, optional) Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination: (float, optional) The inclination of the viewed image
        rotangle: (float, optional) The clockwise rotation around the z-axis of the system
        verbose: (bool, default=1) Report task activity
    Returns:
        (ndarray) Calculated line-of-sight dust temperature
    '''
    # Check if all the folders, that need to exist, do exist
    path = goto_folder(isink, iout)
    check_folders(path)

    # We get the values from the osyris2Dslab calculation
    osyris_data = run_osyris2dslab(isink, iout, resolution=resolution, width=width, dz=dz, view=view, inclination=inclination, rotangle=rotangle, verbose=verbose)

    # We return the temperature
    return osyris_data["Td"] / osyris_data["d"]

def calc_velocities(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, verbose=1):
    '''
    Calculate the velocities in the simulated RAMSES data.

    Parameters:
        isink: (int) The sink ID
        iout: (int) The snapshot ID
        resolution: (int) The resolution of the generated image
        width: (float) The width in AU of the generated image
        dz: (float) The depth in AU of the generated image
        view: (string, optional) Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination: (float, optional) The inclination of the viewed image
        rotangle: (float, optional) The clockwise rotation around the z-axis of the system
        verbose: (bool, default=1) Report task activity
    Returns:
        (ndarray) Calculated velocities
    '''
    # Check if all the folders, that need to exist, do exist
    path = goto_folder(isink, iout)
    check_folders(path)

    view_str, inclination, rotangle = get_view(view=view, inclination=inclination, rotangle=rotangle, verbose=verbose)

    if not os.path.exists(path+"/saved_values/velocities-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)+".dat"):
        print("The data for this configuration doesn't exist already. Generating...")
        ramses = nexus.dataclass()
        datadir = '/lustre/astro/troels/IMF_512_cores/christian/sink_'+str("{:03}".format(isink+1))+'/data'
        ramses.load(snap = iout, io = 'RAMSES', path = datadir, sink_id=isink, verbose=verbose, dtype = 'float64')
    
        # Orient the coordinate system to the angular momentum vector'
        # If we're too close to the object in zoom, we should realign the coordinate system
        if width < 150:
            ramses.recalc_L(r = width)
        else:
            ramses.recalc_L()
        ramses.calc_trans_xyz()
        ramses.vx, ramses.vy, ramses.vz = ramses.trans_vrel

        # Use the calculated vectors for viewing the disk
        east_vector, north_vector, normal_vector = calc_view_vectors(isink, iout, view=view, inclination=inclination, rotangle=rotangle)
        # Create dictionary
        viewpoint = {'new_x': east_vector, 
                     'new_y': north_vector, 
                     'view_vector': normal_vector}

        print("Creating the 2D slab with Osyris. This might take a while...")
        ramses.osyris2Dslab(variables = ['vx', 'vy', 'vz'], 
                            viewpoint=viewpoint, 
                            resolution=resolution, 
                            view = width, 
                            dz = dz,
                            weights=['mass', 'mass', 'mass'])
        
        # We only want to see gas moving towards/away from us
        # For this view it makes more sense to look at the x-component of the velocity, which in turn should be the x-component of v_phi
        velocities = ramses.osyris_ivs['data1']['vx'] * ramses.v_cgs * 1e-5 # km/s

        #if (np.isnan(velocities).mask).any() == True:
        #    print("Warning: NaN values detected in the array! Interpolating missing values...")
        #    # Get the coordinates of the valid (non-NaN) points
        #    valid_coords = np.array(np.where(~velocities.mask)).T
        #    valid_values = velocities[~velocities.mask]
#
        #    # Get the coordinates of all points
        #    all_coords = np.array(np.where(np.ones_like(velocities, dtype=bool))).T
#
        #    # Interpolate the NaN values
        #    interpolated_velocities = griddata(valid_coords, valid_values, all_coords, method='linear')
        #    velocities = interpolated_velocities.reshape(velocities.shape)
        np.savetxt(path+"/saved_values/velocities-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)+".dat", velocities)
    else:
        print("The data for this configuration already exists. Loading...")
        velocities = np.loadtxt(path+"/saved_values/velocities-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)+".dat")
    return velocities

def trace_tau(isink, iout, npix, wav=None, sizeau=1000, setthreads=4, view=None, inclination=None, rotangle=None,
              imolspec=1, iline=None, widthkms=None, linenlam=None, nostar=True, verbose=1):
    '''
    Generate an image of the optical depth.
    
    Parameters:
        isink: (int) The sink ID
        iout: (int) The snapshot ID
        npix: (int) Number of pixels on the rectangular images
        wav: (float) Wavelength of the image in micron - Ignore if doing line imaging
        sizeau: (float) Diameter of the image in au
        setthreads: (int) Number of threads for RADMC-3D to use for image generation
        view: (string, optional) Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination: (float, optional) The inclination of the viewed image
        rotangle: (float, optional) The clockwise rotation around the z-axis of the system
        imolspec: (int, default=1) Molecule to use from 'lines.inp' file
        iline: (int, optional) Line transition index
        widthkms: (float, optional) Width of the frequency axis of the channel maps
        linenlam: (int, optional) Number of wavelengths to calculate images at
        nostar: (bool, default=True) If True the calculated images will not contain stellar emission
        verbose: (bool, default=1) Report task activity
        
    Returns:
        radmc3dPy.image.radmc3dImage class

    '''
    # Check if all the folders, that need to exist, do exist
    path = goto_folder(isink, iout)
    check_folders(path)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)

    # Check if the inputs for molecular transitions are given
    if np.count_nonzero([iline, widthkms, linenlam]) == 3:
        # Let's read in which molecule is of interest here
        molecules = np.loadtxt(path+"/lines.inp", skiprows=2, dtype=str)[:,0]
        molecule_name = molecules[imolspec-1] # The imolspec is 1-indexed

        fname = "tauimage-"+molecule_name+"-"+view_str+"-npix"+str(npix)+"-"+str(sizeau)+"au-transition"+str(iline)+"-widthkms"+str(widthkms)+"-lines"+str(linenlam)
    elif np.count_nonzero([iline, widthkms, linenlam]) < 3 and np.count_nonzero([iline, widthkms, linenlam]) > 0:
        raise ValueError("To do molecular line images, 'iline', 'widthkms' and 'linenlam' must all be given.")
    else:
        fname = "tauimage-"+view_str+"-npix"+str(npix)+"-"+str(sizeau)+"au-"+str(int(wav))+"mu"

    if os.path.exists(path+"/saved_images/"+fname+".out"):
        print("The optical depth has already been traced. Loading image...")
    else:
        # Now that we've gone through these checks, we should have let through either the molecular or single wavelength image, which create_image will handle
        create_image(isink, iout, npix=npix, wav=wav, sizeau=sizeau, setthreads=setthreads, inclination=inclination, rotangle=rotangle, imolspec=imolspec, 
                        iline=iline, widthkms=widthkms, linenlam=linenlam, nostar=nostar, tracetau=True)
        shutil.move(path+"/image.out", path+"/saved_images/"+fname+".out")
    
    # Now we can load in the image
    img = image.readImage(fname=path+"/saved_images/"+fname+".out")

    # Fix the orientation of the image for plotting purposes
    img.image = np.rot90(img.image, k=1)
    
    return img

    #    if os.path.exists(path+"/saved_images/tauimage-"+molecule_name+"-"+view_str+"-"+str(sizeau)+"au-transition"+str(iline)+"-widthkms"+str(widthkms)+"-lines"+str(linenlam)+".out"):
    #        print("The optical depth has already been traced. Loading image...")
    #    else:
    #        print("The optical depth for this configuration hasn't been found. Generating...")
    #        create_image(isink, iout, npix=npix, sizeau=sizeau, setthreads=setthreads, 
    #                     imolspec=imolspec, iline=iline, widthkms=widthkms, linenlam=linenlam, 
    #                     nostar=nostar, tracetau=True, inclination=inclination, rotangle=rotangle)
    #        shutil.move(path+"/image.out", path+"/saved_images/tauimage-"+molecule_name+"-"+view_str+"-"+str(sizeau)+"au-transition"+str(iline)+"-widthkms"+str(widthkms)+"-lines"+str(linenlam)+".out")
    #    img = image.readImage(fname=path+"/saved_images/tauimage-"+molecule_name+"-"+view_str+"-"+str(sizeau)+"au-transition"+str(iline)+"-widthkms"+str(widthkms)+"-lines"+str(linenlam)+".out")
#
    #elif np.count_nonzero([iline, widthkms, linenlam]) < 3 and np.count_nonzero([iline, widthkms, linenlam]) > 0:
    #    raise ValueError("To do molecular line images, 'iline', 'widthkms' and 'linenlam' must all be given.")
    #
    #else:
    #    if os.path.exists(path+"/saved_images/tauimage-"+view_str+"-"+str(sizeau)+"au-"+str(int(wav))+"mu.out"):
    #        print("The optical depth has already been traced. Loading image...")
    #    else:
    #        print("The optical depth for this configuration hasn't been found. Generating...")
    #        create_image(isink, iout, npix=npix, wav=wav, sizeau=sizeau, setthreads=setthreads, inclination=inclination, rotangle=rotangle, nostar=nostar, tracetau=True)
    #        #print("Calling RADMC3D tausurf command...")
    #        #command = "/lustre/astro/troels/radmc3d/bin/radmc3d image lambda "+str(wav)+" npix "+str(npix)+" incl "+str(incl)+" phi "+str(phi)+" posang "+str(position_angle)+" sizeau "+str(sizeau) + " tracetau"
    #        #os.system(command)
    #        #print("Executing command:", command)
    #        shutil.move(path+"/image.out", path+"/saved_images/tauimage-"+view_str+"-"+str(sizeau)+"au-"+str(int(wav))+"mu.out")
    #    img = image.readImage(fname=path+"/saved_images/tauimage-"+view_str+"-"+str(sizeau)+"au-"+str(int(wav))+"mu.out")
    #return img

def find_Tbol(isink, iout, npix, sizeau, setthreads, view, inclination, rotangle, tbol_points=20, verbose=1):
    # FIXME this needs to be updated to the current version :)
    '''Calculate the bolometric temperatures in the image for the given sink and output'''
    # Check if all the folders, that need to exist, do exist
    path = goto_folder(isink, iout)
    check_folders(path)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)

    fname = "tbol-"+view_str+"-npix"+str(npix)+"-"+str(sizeau)+"au"

    # Check if the values for the SED already exist
    if not os.path.exists(path+"/saved_values/"+fname+".npz"):
        print("The bolometric temperatures for this configuration doesn't exist. Creating...")
        # Create wavelength spectrum
        wavs = np.linspace(1, 500, tbol_points)
        # Create the flux array
        images = []
        print("Calculating the images...")
        for i in range(len(wavs)):
            create_image(isink, iout, npix=npix, wav=wavs[i], sizeau=sizeau, setthreads=setthreads, inclination=inclination, rotangle=rotangle)
            img = image.readImage(fname=path+"/image.out")

            # Save the image in the array
            images.append(img.image[:,:,0])
        
        images = np.array(images)
        # Create an array to house the temperatures
        temps = np.zeros_like(img.image[:,:,0])
        # Now that we have an images.shape = (len(wavs),n,n) array, we can go through the pixels one by one
        # The image is defined as [y,x], meaning we can calculate the bolometric temperature in each image as
        
        print("Calculating the temperatures...")
        for y in range(images.shape[1]):
            for x in range(images.shape[2]):
                temps[y,x] = Tbol(images[:,y,x]*1e23, wav=wavs) # We convert to Jy by multiplying by 1e23
        
        print(f"Shape of the temperature array {temps.shape}")

        np.savez(path+"/saved_values/"+fname+".npz", wavelengths=wavs, temperatures=temps)
        return temps

    else:
        print("The bolometric temperatures for this configuration already exists. Loading it in...")
        # Load in the Tbol data from the folder
        data_to_print = np.load(path+"/saved_values/"+fname+".npz")
        return data_to_print["temperatures"]

# FIXME make it work like the SED function
def Tbol_map(isink, iout, npix, sizeau, iwav, setthreads, view=None, inclination=None, rotangle=None, dpc=None, verbose=1):
    '''
    This function calls the create image function for a single wavelength.

    Parameters:
        isink (int): The sink ID
        iout (int): The snapshot ID
        npix (int): Number of pixels on the rectangular images
        sizeau (float): Diameter of the image in au
        iwav (float): Number of wavelengths to calculate the datacube for
        setthreads (int): Number of threads for RADMC-3D to use for image generation
        view (string, optional): Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination (float, optional): The inclination of the viewed image
        rotangle (float, optional): The clockwise rotation around the z-axis of the system
        dpc (float): Distance to source in pc
        verbose (bool, default=1):  Report task activity

    Returns:
        img_stack (ndarray): 3D datacube (npix,npix,iwav) of calculated images
    '''
    path = goto_folder(isink, iout)
    view_str, inclination, rotangle = get_view(view=view, inclination=inclination, rotangle=rotangle, verbose=verbose)
    # We want to create a spectrum for each pixel, essentially
    freqs = np.logspace(np.log10(30), np.log10(30000), iwav) * 1e9 # Hz
    light_speed = 299792458 * 1e6 # mum/s
    wavs = light_speed/freqs

    fname = "3d-datacube-"+view_str+"-npix"+str(npix)+"-"+str(sizeau)+"au-"+str(iwav)+"pts"

    if os.path.exists(path+"/saved_values/"+fname+".npy"):
        if verbose: print("This data already exists. Loading...")
        img_stack = np.load(path+"/saved_values/"+fname+".npy")
    
    else:
        for i in range(len(wavs)):
            if verbose: print(f"Calculating for: {wavs[i]}")

            # Should return the img in Jy/px
            img = single_wavelength_image(isink, iout, npix=npix, wav=wavs[i], sizeau=sizeau, setthreads=setthreads, inclination=inclination, rotangle=rotangle, dpc=dpc)

            # We want to save the image grid in an array
            if i == 0:
                img_stack = np.copy(img.image)
            else:
                img_stack = np.dstack([img_stack, img.image])

        np.save(path+"/saved_values/"+fname+".npy", img_stack)
    
    return img_stack