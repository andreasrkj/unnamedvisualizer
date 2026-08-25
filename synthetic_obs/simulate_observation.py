import os, sys
import astropy.io.fits as fits
from ..helper_functions import get_casa_project_name

def run_simalma(image_name, path, antennalist=['alma.cycle7.8.cfg', 'alma.cycle7.5.cfg'], totaltime=['15h', '3h'], pwv=1.5, threshold="4mJy", niter=5000, overwrite=False, verbose=True):
    project_name = get_casa_project_name(image_name)
    root = os.getcwd()
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    # ------- OpenMP settings for CASA -------
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OMP_PLACES"] = "cores"
    os.environ["OMP_PROC_BIND"] = "close"

    sys.path.insert(0,'/lustre/hpc/software/astro/casa/casa-6.6.1-17-pipeline-2024.1.0.8/lib/py/lib/python3.8/site-packages/')
    import casatasks
    # -------------- END SETUP ---------------

    data, header = fits.getdata(path+"/saved_fits/"+image_name+".fits", header=True)

    map_x = header["NAXIS1"] * header["CDELT1"] * 3600; map_y = header["NAXIS2"] * header["CDELT2"] * 3600

    if verbose: print("Running simALMA with the following parameters:")
    if verbose: print("Antenna list:", antennalist, "with times:", totaltime)
    if verbose: print("Precipitable water vapor (pwv):", pwv)
    if verbose: print("And cleaning threshold", threshold, "with niter", niter)

    try:
        os.chdir(path+"/casa_projects")
        casatasks.simalma(project=project_name, dryrun=False, skymodel="../saved_fits/"+image_name+".fits", setpointings=True, integration="4500s", mapsize=[str(map_x)+'arcsec',str(map_y)+'arcsec'],
                          antennalist=antennalist, hourangle='transit', totaltime=totaltime, pwv=pwv, image=True, imsize=[header["NAXIS1"], header["NAXIS2"]], niter=niter, 
                          threshold=threshold, graphics='file', verbose=True, overwrite=overwrite, parallel=False)
        os.chdir(root)
    except:
        os.chdir(root)
        raise OSError("simalma failed. See CASA logs for details.")

    # After running we should export to FITS file
    if verbose: print("Exporting CASA files to .fits")
    if len(antennalist) > 1:
        if verbose: print("Outputting combined antenna image as FITS...")
        config_str = "_".join(antennalist).replace("alma.cycle","").replace(".cfg","")
        casatasks.exportfits(imagename=path+"/casa_projects/"+project_name+"/"+project_name+".concat.image.pbcor", 
                             fitsimage=path+"/saved_fits/simalma_combined"+config_str+"_"+image_name+".fits")
    else:
        if verbose: print(f"Outputting image for configuration {antennalist[0]}")
        casatasks.exportfits(imagename=path+"/casa_projects/"+project_name+"/"+project_name+"."+antennalist[0].replace("cfg","noisy")+".image.pbcor", 
                             fitsimage=path+"/saved_fits/simalma_"+antennalist[0].replace("cfg","noisy")+"_"+image_name+".fits")