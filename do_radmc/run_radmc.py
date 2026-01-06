#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'/groups/astro/troels/python')
sys.path.insert(0,'/groups/astro/troels/python/sfrimann')
sys.path.insert(0,'/groups/astro/troels/python/sigurd')
sys.path.insert(0,'/groups/astro/andreask/python')
from ..sink_config import sink_dirs
import numpy as np
import pyradmc3d as pyrad
import pyramses as pyram
#import pms_interpolate as pms
from astropy import constants as con
import os, shutil

from glob import glob
from pyradmc3d.model.model_pyramses import PyRamsesMulti
from astropy.io import fits as pyfits

from multiprocessing import Pool, current_process

#from bezier_interpolation import *

Lsun = con.L_sun.cgs.value
cc = con.c.cgs.value
hh = con.h.cgs.value
kb = con.k_B.cgs.value
spy = 365.25*24*3600

def createCell(directory, nsink, nout, datadir = "./data", full_oct=False, sizeCutout=40000, scales_file="./scales_IMF512.fits"): 
    '''
    Create cell and return it (no save).
    '''
    try:
        sinks = pyram.rsink(nout, datadir=datadir)
    except IOError:
        print("rsink could not load")
        return

    # ------------------------------
    info_file = os.path.join(datadir, 'output_{0:05d}/info_{0:05d}.txt'.format(nout))
    info = pyram.rd_info(info_file)
    lvlmax = info['levelmax']

    datadir = os.path.join(directory, datadir)
    scales_fits = os.path.join(directory, scales_file)

    # ------------------------------
    # create sink
    sink = pyram.Sink(nout, datadir=datadir, scales_file=scales_fits)
    sink.rsink()
    reference_sink = sink.reference_sink(sink_dirs[str(nsink)][1])  # Calculate reference
    sink.reference = reference_sink

    # ------------------------------
    # create cell
    cell = pyram.Cell(nout, datadir=datadir, scales_file=scales_fits,
                        reference=reference_sink, lvlmax=lvlmax)
    cell.amr2cell(dr='{:d} AU'.format(sizeCutout), silent=True)

    return cell

# -----------------------------------------------------------------------
def loadSink(directory, nsink, nout):

    modeldir = os.path.join(directory, './sink{:03d}/nout{:04d}'.format(nsink, nout))
    if not os.path.exists(os.path.join(modeldir, 'sink.hdf5')):
        raise OSError('Could not find sink.hdf5')

    sink = pyram.Sink()
    sink.load(directory=modeldir)
    reference_sink = sink.reference_sink(sink_dirs[str(nsink)][1])  # Calculate reference

    return sink, reference_sink

# -----------------------------------------------------------------------
def loadCell(directory, nsink, nout):

    modeldir = os.path.join(directory, './sink{:03d}/nout{:04d}'.format(nsink, nout))
    if not os.path.exists(os.path.join(modeldir, 'cell.hdf5')):
        raise OSError('Could not find cell.hdf5')

    cell = pyram.Cell()
    try:
        cell.load(directory=modeldir)
    except KeyError:
        # In the case of KeyError we try to delete existing cell.hdf5 and create a new one.
        os.system("rm -rf %s" %os.path.join(modeldir, 'cell.hdf5'))
        cell = createCell(directory, nsink, nout)
    return cell

# -----------------------------------------------------------------------
def load_hdf(directory, nsink, nout):
    '''
        Load sink.hdf and cell.hdf
    '''

    sink, reference_sink = loadSink(directory, nsink, nout)
    cell = loadCell(directory, nsink, nout)

    return sink, cell

# -----------------------------------------------------------------------
def calc_time(isink, iout, datadir):
    s = pyram.rsink(iout, datadir=datadir)
    sink_age = ((s["snapshot_time"] - s["tcreate"][0]) - (s["tcreate"][sink_dirs[str(isink)][1]] - s["tcreate"][0])) * 21728716.037350874 #code2yr
    return sink_age

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def get_luminosities(isink, iout, datadir, mesadatadir):
    # We need to grab Lstar, Lacc, T_eff and R_star
    mesa_data = np.loadtxt(mesadatadir, skiprows=1, delimiter=" ")
    age   = mesa_data[:,0]; Teff  = mesa_data[:,3]; 
    Lstar = mesa_data[:,4]; Lacc  = mesa_data[:,5]

    # Find time to average over
    t_prev = calc_time(isink, iout-1, datadir)
    t_curr = calc_time(isink, iout, datadir)
    t_next = calc_time(isink, iout+1, datadir)

    t_lb = 0.5 * (t_prev + t_curr)
    t_ub = 0.5 * (t_next + t_curr)

    # Find nearest values
    idx_lower = find_nearest(age, t_lb); idx_upper = find_nearest(age, t_ub)

    # Calculate mean values
    Ltot_mean = np.mean(Lstar[idx_lower:idx_upper+1]) + np.mean(Lacc[idx_lower:idx_upper+1])
    Teff_mean = np.mean(Teff[idx_lower:idx_upper+1]**4)**(1/4)
    
    return Ltot_mean, Teff_mean

#def get_luminosities(dmdt, mass, age, returnSeparateLum=False):
#    '''
#        Modified S. Frimann routine to calculate the luminosity and effective temp. of
#        PMS stars by interpolating between PMS models.
#    '''
#    racc = 2.5  # accretion radius in units of Rsun
#    Lphot, Te, _ = pms.ageMassInterpolate(np.abs(mass), age, addtime=-100000.0, grid='DM97')
#    Lacc = 7.820124319632798*np.abs(mass)/0.25*dmdt/2.5e-6*2.5/racc # almost 0.5 * G M Mdot/ R
#
#    if np.isnan(Lphot).any():
#        for i, L in enumerate(Lphot):
#            if np.isnan(L):
#                Lphot[i] = 0.0
#        #Lphot = np.zeros_like(Lphot)
#
#    if returnSeparateLum:
#        return (Lphot+Lacc), Te, Lphot, Lacc
#    else:
#        return (Lphot+Lacc), Te

# -----------------------------------------------------------------------
def doRADMC(directory, nsink, nout, datadir='./data', mesadir='', scales_file='./scales_IMF512.fits', 
            nsink2=None, threads=1, nphot=50000000, molecule='', mumolecule=0., abundance=0.):
    '''
    Calculate RADMC3D models in Lrange determined from accretion rates. Uses PyRamsesMulti RADMC3D model which includes neighbouring protostars.

    n_points specifies the amount of points between Lmin and Lmax 
    on which to calculate the radiative transfer models for later interpolation.
    '''

    # -------------------
    # Read sink data for nout:
    noutdirectory = directory + '/sink%.3i/nout%.4i' % (nsink, nout)

    # -------------------
    # Load sink and cell objects used for RADMC3D structure model
    datadir = os.path.join(directory, datadir)
    info_file = os.path.join(
        datadir, 'output_{0:05d}/info_{0:05d}.txt'.format(nout))
    info = pyram.rd_info(info_file)
    lvlmax = info['levelmax']

    scales_fits = os.path.join(directory, scales_file)

    #print "loading sink"
    sink, cell = load_hdf(directory=directory, nsink=nsink, nout=nout)
    sink_numbers = [sink_dirs[str(nsink)][1]]
    if nsink2 != None:
        sink_numbers.append(nsink2)

    # -------------------
    ### read sink data ###
    #dmdt = sink.logdmdt_manual.to('Msun/yr')[nsink].value # manual accretion rate calculations
    #print("dmdt      : ", dmdt)
    #print("sink mass : ", sink.mass.to('Msun')[nsink].value)
    #print("sink age  : ", sink.age.to('yr')[nsink].value)
    #lum, Te = get_luminosities(dmdt, sink.mass.to('Msun')[nsink].value, age=sink.age.to('yr')[nsink].value) # TODO add own values!
    lum, Te = get_luminosities(nsink, nout, datadir, mesadir)
    lum = [lum]
    Te = [Te]
    if nsink2 != None:
        dmdt2 = sink.logdmdt_manual.to('Msun/yr')[nsink2].value # manual accretion rate calculations
        lum2, Te2 = get_luminosities(dmdt2, sink.mass.to('Msun')[nsink2].value, age=sink.age.to('yr')[nsink2].value)
        lum.append(lum2)
        Te.append(Te2)

    sink_numbers = np.array(sink_numbers)
    lum = np.array(lum)
    Te = np.array(Te)
    
    # -------------------
    # Get Lacc, Lphot and Teff based on range of dMdt and mass(es)
    radmc_model = 'model'

    modelname = os.path.join(
        noutdirectory, 'dust_temperature_%s.bdat' % (radmc_model))                                                       #! T format changed

    print('Doing RADMC-3D models')
    # -------------------
    # Load dust
    #dust = pyrad.semenov(metallicity='normal',
    #                     name='homogeneous spheres', temperature=100)
    dust = pyrad.ossenkopfHenning('thin6',wl=None,extrapolate=False) 

    # -------------------
    # Load gas
    molecule = ['co', 'ph2co', 'c18o', '13co']
    mumolecule = np.array([28, 30, 30, 29]) # If code fixed - should be unimportant
    abundance = np.array([1e-4, 1e-8, 1e-4/560, 1e-4/68]) # Now number density
    # Freeze-out below a certain temperature abundance = 1e-6 @ T = 25/30K
    # H2CO - Temperature tracer (difference between lines) (transition 3, 10) abundance=1e-8, 1e-10 (frozen out)
    # https://home.strw.leidenuniv.nl/~moldata/datafiles/ph2co-h2.dat
    #vmic = 1e4
    
    # -------------------
    # Run RADMC3D:

    ## Clean directory:
    inp = glob('%s/*.inp' % noutdirectory)
    fits = glob('%s/*.fits' % noutdirectory)
    binp = glob('%s/*.binp' % noutdirectory)
    # clean-up from previous RADMC models which may conflict the run:
    trash = np.concatenate([inp, fits, binp])

    if os.path.exists(modelname):
        #print "Deleting file: ", modelname
        os.system("rm %s" % modelname)
    for tpath in trash:
        # print "Deleting: ", tpath
        os.system("rm %s" % tpath)
    # ------- Do checks
    if np.isnan(lum).any() or (lum <= 0.0).any():
        print("Luminosity is NaN or zero. Changing to 1e-5")
        print(lum)
        lum[np.isnan(lum)] = 1e-5
        lum[lum <= 0.0] = 1e-5
    if np.isnan(Te).any() or (Te <= 0.0).any():
        print("Te is NaN or zero. Changing to 2000")
        print(Te)
        Te[np.isnan(Te)] = 2000.0
        Te[Te <= 0.0] = 2000.0

    # ------- Make RADMC model
    print(sink_numbers.shape, lum.shape, Te.shape)
    if molecule != '':
        model = PyRamsesMulti(cell, sink, dust=dust, modeldir=noutdirectory,
                            sink_numbers=sink_numbers, luminosities=lum, Teffs=Te, velocity=True,
                            molecule=molecule, mumolecule=mumolecule, abundance=abundance)
        print('Velocity shape:', model.gasVelocity.shape)
    else:
        model = PyRamsesMulti(cell, sink, dust=dust, modeldir=noutdirectory,
                            sink_numbers=sink_numbers, luminosities=lum, Teffs=Te)

    # -------------------
    # Add mcmono_wavelength_micron.inp to noutfolder:
    mcmono_wl = np.linspace(0.091, 0.12, 15)
    with open(os.path.join(noutdirectory, 'mcmono_wavelength_micron.inp'), 'w') as f: 
        f.write('15\n')
        for wl in mcmono_wl:
            f.write('%e\n' % wl)
        f.close()

    # -------------------
    # Add external field (ISRF) to noutfolder:
    shutil.copyfile('/lustre/astro/troels/radmc3d/external_source_av2.inp', os.path.join(noutdirectory, 'external_source.inp'))


    # Define a function to print stars to a file
    def print_stars_to_file(print_function, filename):
        try:
            # Check if the file exists
            if not os.path.exists(filename):
                # If the file doesn't exist, create it
                with open(filename, 'w') as f:
                    pass  # This just creates an empty file

            # Open the file in append mode
            with open(filename, 'a') as f:
                # Redirect standard output to the file
                sys.stdout = f
                # Call the original print_stars function
                print_function()
                # Restore standard output
                sys.stdout = sys.__stdout__
        except Exception as e:
            print("An error occurred:", e)
    


    output_file_path = "/groups/astro/andreask/production_run/sink{:03d}/meta_data.txt".format(nsink)
    # Call print_stars function and redirect output to a file
    print_stars_to_file(model.star.print_stars, output_file_path)



    model.star.print_stars()
    model.write(use_existing=False)
    mode = 'new'
    pyrad.writeRadmc3dInp(
        modeldir=noutdirectory, nphot=nphot, nphot_mono=0, modified_random_walk=True, tgas_eq_tdust = 1, mode=mode) # Default nphot=50000000, mono=25000000
        #modeldir=noutdirectory, nphot=5000, nphot_mono=2500, modified_random_walk=True, tgas_eq_tdust = 1, mode=mode)
    print('Finished writing RADMC-3D model')

    lexe = '/lustre/astro/troels/radmc3d/bin/radmc3d_ifx'

    # ------- run RADMC-3D
    print('Running mctherm...')
    pyrad.mctherm(overwrite=True, modeldir=noutdirectory,
                   silent=False, lexe=lexe, threads=threads)
    print('Finished mctherm')

    # Copy dust_temperature data
    #shutil.copyfile(src=os.path.join(noutdirectory, 'dust_temperature.bdat'),                                                                           #! T format changed!
    #                dst=modelname)

    # Dictionary with information on the filenames of the different dust temp. models
    #dict_out = 'dust_temperature_%s.bdat' % radmc_model                                                                                                  #! T format changed!
    dict_out = None

    return dict_out, lum, cell

# -----------------------------------------------------------------------
# def read_external_source(dirpath, fname='external_source.inp'):
#     f = open(os.path.join(dirpath, fname), 'r')
#     iformat = int(f.readline())
#     nlines = int(f.readline())
#     wl = [float(f.readline()) for i in range(nlines)]
#     I = [float(f.readline()) for i in range(nlines)]
#     return np.array(wl), np.array(I)

def process_my_task(task):
    # get current affinity mask
    affinity_org = os.sched_getaffinity(0)
    ncores = len(affinity_org) // 2 # /2 because we have hyperthreading
    ncores = 256

    # FIXME hardcoded, has to be the same as below
    npool = 1

    # set affinity mask according to how many worker processes we have
    current = current_process()
    id_worker = current._identity[0]
    cpu_start = (id_worker-1) * ncores // npool
    cpu_end   =  id_worker    * ncores // npool
    threads = cpu_end - cpu_start
    print("id of worker ", id_worker, " threads ", threads)

    affinity_mask = {i for i in range(cpu_start,cpu_end)}
    os.sched_setaffinity(0, affinity_mask)

    # set up OpenMP environment for RadMC child processes
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OMP_PLACES"] = "cores"
    os.environ["OMP_PROC_BIND"] = "close"

    directory = './'
    nsink  = 13
    nsink2 = None
    nout = task
    datadir = "/lustre/astro/troels/IMF_512_cores/sink_13/data/"
    #datadir='/groups/astro/andreask/zoomin_data/sink_025/data' # where to find data for run
    scales_file='./scales_IMF512.fits'

    dict_out, lum, cell = doRADMC(directory, nsink, nout, datadir=datadir, scales_file=scales_file, nsink2=nsink2, threads=threads)

    # reset affinity mask for this process, so that it is ready for next iteration
    os.sched_setaffinity(0,affinity_org)

# FIXME probably not necessary, can simply call doRADMC with threads. We rarely process more than one at a time...
#if __name__ == '__main__':
#    # FIXME hardcoded, has to be the same as above
#    npool = 1
#
#    nstart = int(os.environ["NSTART"])
#    nend   = int(os.environ["NEND"])
#    step = 1
#
#    pool = Pool(npool)                                # Create a multiprocessing Pool
#    pool.map(process_my_task, range(nstart,nend+1, step))   # process data_inputs iterable with pool
#
#    pool.close()
#    pool.join()