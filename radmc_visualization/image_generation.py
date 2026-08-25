import os, sys
from ..helper_functions import goto_folder, check_in_folder, check_folders, calc_view_vectors, get_view, get_projang, get_posang
import shutil
import numpy as np
sys.path.insert(0,'/lustre/hpc/software/astro/casa/casa-6.6.1-17-pipeline-2024.1.0.8/lib/py/lib/python3.8/site-packages/')
from casatools import synthesisutils
su = synthesisutils()
sys.path.insert(0,"/groups/astro/andreask/python")
from radmc3dPy_SSJ import image

# Create RADMC image
def create_image(isink, iout, npix=800, wav=None, sizeau=1000, setthreads=4, view=None, 
                 inclination=None, rotangle=None, imolspec=1, iline=None, widthkms=None, linenlam=None, 
                 nostar=True, tracetau=False):
    '''
    This function calls the RADMC-3D create image function, for either a single wavelength, 
    a transition and for a given width around it, or traces the optical depth and outputs a 'image.out' file.

    Parameters:
        isink (int): The sink ID
        iout (int): The snapshot ID
        npix (int): Number of pixels on the rectangular images
        wav (float): Wavelength of the image in micron - Ignore if doing line imaging
        sizeau (float): Diameter of the image in au
        setthreads (int): Number of threads for RADMC-3D to use for image generation
        view (string, optional): Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination (float, optional): The inclination of the viewed image
        rotangle (float, optional): The clockwise rotation around the z-axis of the system
        imolspec (int, default=1): Molecule to use from 'lines.inp' file
        iline (int, optional): Line transition index
        widthkms (float, optional): Width of the frequency axis of the channel maps
        linenlam (int, optional): Number of wavelengths to calculate images at
        nostar (bool, default=True): If True the calculated images will not contain stellar emission
        tracetau (bool, default=False): If True returns the traced optical depth instead of emission
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
                            doppcatch=True, exe = '/lustre/astro/troels/radmc3d/bin/radmc3d_ifx')
        
        elif np.count_nonzero([iline, widthkms, linenlam]) < 3 and np.count_nonzero([iline, widthkms, linenlam]) > 0:
            raise ValueError("To do molecular line images, 'iline', 'widthkms' and 'linenlam' must all be given.")
        
        else: # Do single wavelength image
            image.makeImage(npix=npix, incl=incl, phi=phi, wav=wav, sizeau=sizeau,
                            setthreads=setthreads, posang=position_angle, nostar=nostar, tracetau=tracetau,
                            doppcatch=True, exe = '/lustre/astro/troels/radmc3d/bin/radmc3d_ifx')
            
        # Leave the folder
        os.chdir(org_path)
    except: # Go to original directory if program fails.
        print("Program failed?")
        os.chdir(org_path)
        

def single_wavelength_image(isink, iout, npix, wav, sizeau, setthreads, dpc=140, view=None, inclination=None, rotangle=None, nostar=True, verbose=1):
    '''
    This function calls the create image function for a single wavelength.

    Parameters:
        isink (int): The sink ID
        iout (int): The snapshot ID
        npix (int): Number of pixels on the rectangular images
        wav (float): Wavelength of the image in micron - Ignore if doing line imaging
        sizeau (float): Diameter of the image in au
        setthreads (int): Number of threads for RADMC-3D to use for image generation
        view (string, optional): Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination (float, optional): The inclination of the viewed image
        rotangle (float, optional): The clockwise rotation around the z-axis of the system
        dpc (float, default = 140): Distance to source in pc
        nostar (bool, default=True): If True the calculated images will not contain stellar emission
        tracetau (bool, default=False): If True returns the traced optical depth instead of emission
        overwrite (bool, default=False): Whether to overwrite the CASA project folder
    '''
    # Check if all the folders, that need to exist, do exist
    path = goto_folder(isink, iout)
    check_folders(path)

    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)

    # Construct the file name
    fname = "image-"+view_str+"-npix"+str(npix)+"-singlewav-"+str(sizeau)+"au-"+str(int(wav))+"mu"

    if not os.path.exists(path+"/saved_fits/"+fname+".fits"):
        if verbose: print("An image of this configuration doesn't exist. Creating it...")
        # Make the image
        create_image(isink, iout, npix=npix, wav=wav, sizeau=sizeau, setthreads=setthreads, nostar=nostar, 
                     inclination=inclination, rotangle=rotangle)
        # Assume "image.out" is created and not overwritten in the time between creation and reading in...
        img = image.readImage(os.path.join(path,"image.out"))
        img.writeFits(fname=path+"/saved_fits/"+fname+".fits", dpc=dpc, coord="04h04m43.08s 26d18m56.12s", casa=True, nu0=img.freq[0])

    else:
        if verbose: print("An image for this configuration already exists. Loading...")
        # At this point the main function should handle loading...

def molecular_lines_image(isink, iout, npix, sizeau, setthreads, iline, widthkms, linenlam, imolspec=1, view=None, inclination=None, 
                          rotangle=None, dpc=None, nostar=True, verbose=1):
    '''
    This function calls the create image function for an image with multiple wavelengths around a given molecular line transition.

    Parameters:
        isink (int): The sink ID
        iout (int): The snapshot ID
        npix (int): Number of pixels on the rectangular images
        wav (float): Wavelength of the image in micron - Ignore if doing line imaging
        sizeau (float): Diameter of the image in au
        setthreads (int): Number of threads for RADMC-3D to use for image generation
        iline (int, optional): Line transition index
        widthkms (float, optional): Width of the frequency axis of the channel maps
        linenlam (int, optional): Number of wavelengths to calculate images at
        imolspec (int, default=1): Molecule to use from 'lines.inp' file
        view (string, optional): Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination (float, optional): The inclination of the viewed image
        rotangle (float, optional): The clockwise rotation around the z-axis of the system
        dpc (float, optional): Distance to source in pc
        nostar (bool, default=True): If True the calculated images will not contain stellar emission
        verbose (bool, default=1): Report task activity
    '''
    # Check if all the folders, that need to exist, do exist
    path = goto_folder(isink, iout)
    check_folders(path)
    view_str, inclination, rotangle = get_view(view, inclination, rotangle, verbose)

    # Let's read in which molecule is of interest here
    molecules = np.loadtxt(path+"/lines.inp", skiprows=2, dtype=str)[:,0]
    molecule_name = molecules[imolspec-1] # The imolspec is 1-indexed

    # Save fname for later use
    fname = "image-"+molecule_name.replace("+","plus")+"-"+view_str+"-npix"+str(npix)+"-"+str(sizeau)+"au-transition"+str(iline)+"-widthkms"+str(widthkms)+"-lines"+str(linenlam)

    if not os.path.exists(path+"/saved_fits/"+fname+".fits"):
        if verbose: print("An image of this configuration doesn't exist. Creating it...")
        # Make the image with create_image
        create_image(isink, iout, npix=npix, sizeau=sizeau, setthreads=setthreads, inclination=inclination, rotangle=rotangle,
                     imolspec=imolspec, iline=iline, widthkms=widthkms, linenlam=linenlam, nostar=nostar)

        # Read it in and convert to FITS, so we don't have to save all image.out files (loads faster, takes less space)
        img = image.readImage(os.path.join(path,"image.out"))
        if verbose: print("Writing image to fits file...")
        if img.nfreq % 2 == 0:
            nu0 = img.freq[img.nfreq//2-1]/2 + img.freq[img.nfreq//2]/2
        else:
            nu0 = img.freq[img.nfreq//2]
        bandwidth = np.abs(img.freq[0] - img.freq[1]) * 1e-6 # Calculate the bandwidth (assumed the same between freqs)
        img.writeFits(path+"/saved_fits/"+fname+".fits", dpc=dpc, coord="04h04m43.07s 26d18m56.4s", bandwidthmhz=bandwidth, casa=True, nu0=nu0)

    else:
        # If it already exists we load it in...
        if verbose: print("An image for this configuration already exists. Loading...")

def trace_tau(isink, iout, npix, wav=None, sizeau=1000, setthreads=4, view=None, inclination=None, rotangle=None,
              imolspec=1, iline=None, widthkms=None, linenlam=None, nostar=True, verbose=1):
    '''
    Generate an image of the optical depth.
    
    Parameters:
        isink (int): The sink ID
        iout (int): The snapshot ID
        npix (int): Number of pixels on the rectangular images
        wav (float): Wavelength of the image in micron - Ignore if doing line imaging
        sizeau (float): Diameter of the image in au
        setthreads (int): Number of threads for RADMC-3D to use for image generation
        view (string, optional): Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination (float, optional): The inclination of the viewed image
        rotangle (float, optional): The clockwise rotation around the z-axis of the system
        imolspec (int, default=1): Molecule to use from 'lines.inp' file
        iline (int, optional): Line transition index
        widthkms (float, optional): Width of the frequency axis of the channel maps
        linenlam (int, optional): Number of wavelengths to calculate images at
        nostar (bool, default=True): If True the calculated images will not contain stellar emission
        verbose (bool, default=1): Report task activity
        
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