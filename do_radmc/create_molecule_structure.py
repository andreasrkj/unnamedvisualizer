# Code for adding molecules after-the-fact without disrupting the things done by doradmc3dCO.py
# Writes numberdens_MOLECULE.binp and lines.inp files, which are independent of dust_temperature.bdat calculations made with doradmc3dCO.py
# Can be run if one wants to add molecules after the costly MC run

import os, sys
sys.path.insert(0,'/groups/astro/troels/python')
sys.path.insert(0,'/groups/astro/troels/python/sfrimann')
sys.path.insert(0,'/groups/astro/troels/python/sigurd')
sys.path.insert(0,'/groups/astro/andreask/python')
import pyradmc3d as pyrad
import pyramses as pyram
from radmc3dPy_SSJ import analyze
import numpy as np

from ..helper_functions import goto_folder

def loadCell(isink, iout): # Taken from doradmc3dCO.py

    path = goto_folder(isink, iout)
    if not os.path.exists(os.path.join(path, 'cell.hdf5')):
        raise OSError('Could not find cell.hdf5')

    cell = pyram.Cell()
    cell.load(directory=path)
    return cell

def getGasTemp(isink, iout):
    root = os.getcwd()
    path = goto_folder(isink, iout)
    try:
        os.chdir(path)
        grid = analyze.readGrid()
        data = analyze.readData(gdens=True, dtemp=True, ispec='co', grid=grid)
        os.chdir(root)
    except:
        os.chdir(root)
    return data.dusttemp.flatten()

def writeNumberDensity(isink, iout, molecules, abundances, freeze_temp, ice_factor, overwrite=False):
    # Write number densities after the fact
    print("Loading calculated dust (and gas) temperature of the cells...")
    path = goto_folder(isink, iout)
    cell = loadCell(isink, iout)

    gasDensity = cell.density.to('g/cm3').value
    gasTemperature = getGasTemp(isink, iout)
    unit_mass = 1.66053906660e-24

    numDens = gasDensity / (2 * unit_mass) # number density of H2 molecules
    dum = numDens * abundances.reshape(-1, 1) # We take abundance relative to H2

    for i in range(len(molecules)):
        if not os.path.exists(path+"/numberdens_"+molecules[i]+".binp") or overwrite:
            print(molecules[i])
            moleculeDens = np.where(gasTemperature > freeze_temp[i], dum[i], dum[i] * ice_factor[i])
            pyrad.readwriteinp.writeMoleculeDensity(name=molecules[i],numberdens=moleculeDens,modeldir=path,iformat=1,binary=True,overwrite=overwrite)
        else:
            print("Number density already calculated for this output. Set 'overwrite' = True to overwrite.")

def writeLinesList(isink, iout, molecules, overwrite=False):
    '''
    Make or rewrite the lines.inp file for the lines
    '''
    path = goto_folder(isink, iout)
    
    if os.path.exists(path+"/lines.inp"):
        # Load in list:
        dicts = pyrad.readwriteinp.readLines(modeldir=path)
        saved_mols = []
        for i in range(len(dicts)):
            saved_mols.append(dicts[i]["name"])
        for i in range(len(molecules)):
            if molecules[i] not in saved_mols:
                new_dict = {'name': molecules[i], 'inpstyle': 'leiden', 'iduma': 0, 'idumb': 0, 'ncol': 0}
                dicts.append(new_dict)
            else: continue
    else: # If it doesn't exist
        dicts = []
        for i in range(len(molecules)):
            new_dict = {'name': molecules[i], 'inpstyle': 'leiden', 'iduma': 0, 'idumb': 0, 'ncol': 0}
            dicts.append(new_dict)

    pyrad.readwriteinp.writeLines(dicts, modeldir=path, overwrite=overwrite)

def create_molecule_files(isink, iout, molecules, abundances, freeze_temp, ice_factor, overwrite=False):
    '''
    Create the files needed to create molecular line images in RADMC-3D
    '''
    writeNumberDensity(isink, iout, molecules, abundances, freeze_temp, ice_factor, overwrite=overwrite)
    writeLinesList(isink, iout, molecules, overwrite=overwrite)
    print(f"Created molecular files for sink {isink}, snapshot {iout} for molecules: {molecules}")