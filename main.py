import os, sys
import numpy as np
from .helper_functions import goto_folder, check_folders, get_view
from .radmc_visualization import single_wavelength_image, molecular_lines_image, trace_tau
from .image_classes import casaImageClass, imageClass
from .synthetic_obs import run_simalma
sys.path.insert(0,'/lustre/hpc/software/astro/casa/casa-6.6.1-17-pipeline-2024.1.0.8/lib/py/lib/python3.8/site-packages/')
from casatools import synthesisutils
su = synthesisutils()

def continuum_image(isink, iout, npix, wav, sizeau, setthreads, dpc=140, view=None, inclination=None, rotangle=None, nostar=True, 
                    casa=False, antennalist=None, totaltime=None, threshold="4mJy", niter=5000, overwrite=False, verbose=1):
    # Check if all the folders, that need to exist, do exist
    path = goto_folder(isink, iout)
    check_folders(path)

    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)

    # Check if CASA is given the correct number of pixels for tclean
    if casa:
        optimal_npix = su.getOptimumSize(npix)
        if optimal_npix != npix: 
            print("For the cleaning algorithm, CASA demands 'npix' is factorisable by 2,3,5 only. Setting 'npix' = "+str(optimal_npix))
            npix = optimal_npix
            if verbose: print("Set new 'npix' = "+str(npix))

    # Construct the file name
    fname = "image-"+view_str+"-npix"+str(npix)+"-singlewav-"+str(sizeau)+"au-"+str(int(wav))+"mu"

    # We run the RADMC-3D command either way, since it accounts for the image existing or not
    single_wavelength_image(isink=isink, iout=iout, npix=npix, wav=wav, sizeau=sizeau, setthreads=setthreads, dpc=dpc, inclination=inclination, rotangle=rotangle, nostar=nostar)

    if casa:
        if verbose: print("Creating CASA image...")
        if len(antennalist) > 1:
            config_str = "combined"+"_".join(antennalist).replace("alma.cycle","").replace(".cfg","")
        else:
            config_str = antennalist[0]

        # If it exists we load it. Otherwise run the simalma command
        if os.path.exists(path+"/saved_fits/simalma_"+config_str+"_"+fname+".fits"): 
            print("The requested image has already been generated. Loading...")
        else: 
            run_simalma(fname, path, antennalist=antennalist, totaltime=totaltime, pwv=0.5, threshold=threshold, niter=niter, overwrite=overwrite, verbose=bool(verbose))
        
        # Load it in using our class
        img = casaImageClass(isink=isink, iout=iout, image_name=fname, dpc=dpc, antennalist=antennalist)
        
    else:
        img = imageClass(isink=isink, iout=iout, image_name=fname, dpc=dpc)

    return img

def line_image(isink, iout, npix, sizeau, setthreads, iline, widthkms, linenlam, imolspec=1, view=None, inclination=None, rotangle=None, dpc=None, nostar=True, 
               casa=False, antennalist=None, totaltime=None, threshold="4mJy", niter=5000, overwrite=False, verbose=1):
    # Check if all the folders, that need to exist, do exist
    path = goto_folder(isink, iout)
    check_folders(path)

    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)
    
    # Let's read in which molecule is of interest here
    molecules = np.loadtxt(path+"/lines.inp", skiprows=2, dtype=str)[:,0]
    molecule_name = molecules[imolspec-1] # The imolspec is 1-indexed
    print_name = np.loadtxt(path+"/molecule_"+molecule_name+".inp", dtype=str, max_rows=2)[1]

    # Now we need to find the transition (which energy levels we move through)
    # Read how many transitions there are
    ntrans = np.loadtxt(path+"/molecule_"+molecule_name+".inp", skiprows=49, max_rows=1)
    # Load in the transition values [1-indexed as well]
    transitions = np.loadtxt(path+"/molecule_"+molecule_name+".inp", skiprows=51, max_rows=int(ntrans), usecols=(0,1,2), dtype=int)
    printtrans = transitions[iline-1] - 1

    # Save fname for later use
    fname = "image-"+molecule_name+"-"+view_str+"-npix"+str(npix)+"-"+str(sizeau)+"au-transition"+str(iline)+"-widthkms"+str(widthkms)+"-lines"+str(linenlam)

    # Check if CASA is given the correct number of pixels for tclean
    if casa:
        optimal_npix = su.getOptimumSize(npix)
        if optimal_npix != npix: print("For the cleaning algorithm, CASA demands 'npix' is factorisable by 2,3,5 only. Setting 'npix' = "+str(optimal_npix))
        npix = optimal_npix
        if verbose: print("Set new 'npix' = "+str(npix))

    # Call RADMC-3D since it handles whether to create or simply load the image
    molecular_lines_image(isink=isink, iout=iout, npix=npix, sizeau=sizeau, setthreads=setthreads, iline=iline, widthkms=widthkms, linenlam=linenlam, 
                          imolspec=imolspec, inclination=inclination, rotangle=rotangle, dpc=dpc, nostar=nostar, verbose=verbose)

    if casa:
        if verbose: print("Creating CASA image...")
        if len(antennalist) > 1:
            config_str = "combined"+"_".join(antennalist).replace("alma.cycle","").replace(".cfg","")
        else:
            config_str = antennalist[0]

        # If it exists we load it. Otherwise run the simalma command
        if os.path.exists(path+"/saved_fits/simalma_"+config_str+"_"+fname+".fits"): 
            print("The requested image has already been generated. Loading...")
        else: 
            run_simalma(fname, path, antennalist=antennalist, totaltime=totaltime, pwv=0.5, threshold=threshold, niter=niter, overwrite=overwrite, verbose=bool(verbose))
        
        # Load it in using our class
        img = casaImageClass(isink=isink, iout=iout, image_name=fname, dpc=dpc, antennalist=antennalist, printname=print_name, printtrans=printtrans)
        
    else:
        img = imageClass(isink=isink, iout=iout, image_name=fname, dpc=dpc, printname=print_name, printtrans=printtrans)

    return img