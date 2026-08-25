import os, sys
import numpy as np
from ..helper_functions import goto_folder, get_view, calc_view_vectors
from ..sink_config import sink_dirs
import astropy.units as unit
import astropy.constants as cnst
import pickle
from astropy.convolution import convolve as convolve_func

sys.path.insert(0,"/groups/astro/andreask/python")
import RaDvisPython as radvis
from radmc3dPy_SSJ import analyze

def get_sinkdir(isink):
    if isinstance(isink, float):
            sink_id, level = str(isink).split(".")
            sink_id = int(sink_id); level = int(level)
            print(f"You have specified isink as a float, interpreted as sink ID {sink_id} with max level {level}")
            error_msg = f"The data directory for sink {sink_id} with max level {level} hasn't been configured. Please specify the data directory in 'sink_config.py'." # Only necessary if not configured!
    else:
        sink_id = sink_dirs[str(isink)][1]
        # Default level for max resolution - hardcoded FIXME
        level = 20
        error_msg = f"The data directory for sink {sink_id} hasn't been configured. Please specify the data directory in 'sink_config.py'." # Only necessary if not configured!
    try:
        datadir = sink_dirs[str(isink)][0]
    except:
        raise ValueError(error_msg)
    return sink_id, level, datadir

def radvis_temperature(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, convolve=False, beam_kernel=None, verbose=1):
    '''
    Visualize the RADMC-3D temperature in the simulated RAMSES data.

    Parameters:
        isink (int): The sink ID
        iout (int): The snapshot ID
        resolution (int): The resolution of the generated image
        width (float): The width in AU of the generated image
        dz (float): The depth in AU of the generated image
        view (string, optional): Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination (float, optional): The inclination of the viewed image
        rotangle (float, optional): The clockwise rotation around the z-axis of the system
        verbose (bool, default=1): Report task activity
    Returns:
        (ndarray) Density-weighted temperature along line of sight
    '''
    view_str, inclination, rotangle = get_view(view=view, inclination=inclination, rotangle=rotangle, verbose=verbose)
    path = goto_folder(isink, iout)
    fname = "temperature-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)+".dat"

    if not os.path.exists(path+"/saved_values/"+fname):
        sink_id, level, datadir = get_sinkdir(isink)
        ramses = radvis.dataclass()

        org_path = os.getcwd()
        try:
            os.chdir(path)

            grid = analyze.readGrid()
            data = analyze.readData(gdens=True, dtemp=True, ispec='co', grid=grid)
            temp = data.dusttemp.flatten()

            os.chdir(org_path)
        except:
            os.chdir(org_path)
            raise OSError("Something went wrong...")
        
        ramses.load(snap = iout, io = 'RAMSES', path = datadir, sink_id=sink_id, verbose=verbose, dtype = 'float64')
        # Recalculate the coordinates
        ramses.calc_trans_xyz()

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
        ramses.osyris2Dslab(variables = ['d', 'Td'], 
                        viewpoint=custom_viewpoint, 
                        resolution=resolution, 
                        view = width, 
                        dz = dz,
                        weights=[None, None])

        # Convert to CGS units (except "T" which is already in K)
        ramses.osyris_ivs["data1"]["d"]  *= ramses.d_cgs # g/cm^3
        ramses.osyris_ivs["data1"]["Td"] *= ramses.d_cgs # g/cm^3 * K

        # Save the arrays, in this case we can also save the column density (since we have it)
        coldens = ramses.osyris_ivs["data1"]["d"]
        temperature = ramses.osyris_ivs["data1"]["Td"] / coldens

        np.savetxt(path+"/saved_values/"+fname, temperature)
        # If it isn't already generated, let's save it
        if not os.path.exists(path+"/saved_values/coldens-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)+".dat"):
            np.savetxt(path+"/saved_values/coldens-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)+".dat", coldens)
    # Check if we want the convolved version and it doesn't exist, make it
    if convolve and not os.path.exists(path+"/saved_values/"+"convolved_"+fname):
        # Load it in and convolve it with the beam
        temperature = np.loadtxt(path+"/saved_values/"+fname)
        
        # Convolve with the given beam_kernel
        if beam_kernel is not None:
            if verbose: print("Convolving temperature map with given beam.")
            convolved_temp = convolve_func(temperature, beam_kernel)
        else:
            raise ValueError("'beam_kernel' must be supplied, e.g. Gaussian2DKernel.")

        # Save the convolved_temp
        np.savetxt(path+"/saved_values/"+"convolved_"+fname, convolved_temp)
        temperature = convolved_temp
    else: # Load in the file
        if convolve:
            temperature = np.loadtxt(path+"/saved_values/"+"convolved_"+fname)
        else:
            temperature = np.loadtxt(path+"/saved_values/"+fname)

    return temperature



def radvis_column_density(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, convolve=False, beam_kernel=None, verbose=1):
    '''
    Calculate the column density in the simulated RAMSES data.

    Parameters:
        isink (int): The sink ID
        iout (int): The snapshot ID
        resolution (int): The resolution of the generated image
        width (float): The width in AU of the generated image
        dz (float): The depth in AU of the generated image
        view (string, optional): Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination (float, optional): The inclination of the viewed image
        rotangle (float, optional): The clockwise rotation around the z-axis of the system
        verbose (bool, default=1): Report task activity
    Returns:
        (ndarray) Calculated column densities along line of sight
    '''
    view_str, inclination, rotangle = get_view(view=view, inclination=inclination, rotangle=rotangle, verbose=verbose)
    path = goto_folder(isink, iout)
    fname = "coldens-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)+".dat"

    if not os.path.exists(path+"/saved_values/"+fname):
        sink_id, _, datadir = get_sinkdir(isink)
        ramses = radvis.dataclass()        
        ramses.load(snap = iout, io = 'RAMSES', path = datadir, sink_id=sink_id, verbose=verbose, dtype = 'float64')
        # Recalculate the coordinates
        ramses.calc_trans_xyz()

        # Run Osyris2Dslab
        east_vector, north_vector, normal_vector = calc_view_vectors(isink, iout, inclination=inclination, rotangle=rotangle)

        custom_viewpoint = {'new_x': east_vector, 
                            'new_y': north_vector, 
                            'view_vector': normal_vector}

        if verbose: print("Creating the 2D slab with Osyris. This might take a while...")
        ramses.osyris2Dslab(variables = ['d'], 
                        viewpoint=custom_viewpoint, 
                        resolution=resolution, 
                        view = width, 
                        dz = dz,
                        weights=[None])

        # Convert to CGS units
        ramses.osyris_ivs["data1"]["d"]  *= ramses.d_cgs # g/cm^3

        # Save the array
        coldens = ramses.osyris_ivs["data1"]["d"]
        np.savetxt(path+"/saved_values/"+fname, coldens)

    # Check if we want the convolved version and it doesn't exist, make it
    if convolve and not os.path.exists(path+"/saved_values/"+"convolved_"+fname):
        # Load it in and convolve it with the beam
        coldens = np.loadtxt(path+"/saved_values/"+fname)
        
        # Convolve with the given beam_kernel
        if beam_kernel is not None:
            if verbose: print("Convolving column density with given beam.")
            convolved_coldens = convolve_func(coldens, beam_kernel)
        else:
            raise ValueError("'beam_kernel' must be supplied, e.g. Gaussian2DKernel.")

        # Save the convolved_temp
        np.savetxt(path+"/saved_values/"+"convolved_"+fname, convolved_coldens)
        coldens = convolved_coldens
    else: # Load in the file
        if convolve:
            coldens = np.loadtxt(path+"/saved_values/"+"convolved_"+fname)
        else:
            coldens = np.loadtxt(path+"/saved_values/"+fname)

    return coldens

def radvis_spherical_vels(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, verbose=1):
    '''
    Calculate the spherical velocities mass-averaged along the line of sight in the simulated RAMSES data.

    Parameters:
        isink (int): The sink ID
        iout (int): The snapshot ID
        resolution (int): The resolution of the generated image
        width (float): The width in AU of the generated image
        dz (float): The depth in AU of the generated image
        view (string, optional): Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination (float, optional): The inclination of the viewed image
        rotangle (float, optional): The clockwise rotation around the z-axis of the system
        verbose (bool, default=1): Report task activity
    Returns:
        (3, ndarray) velocities in km/s for the 3 spherical coordinates (radial, inclined, azimuthal)
    '''

    view_str, inclination, rotangle = get_view(view=view, inclination=inclination, rotangle=rotangle, verbose=verbose)
    path = goto_folder(isink, iout)
    fname = "sph_velocities-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)+".p"

    if not os.path.exists(path+"/saved_values/"+fname):
        sink_id, _, datadir = get_sinkdir(isink)
        ramses = radvis.dataclass()        
        ramses.load(snap = iout, io = 'RAMSES', path = datadir, sink_id=sink_id, verbose=verbose, dtype = 'float64')
        # Recalculate the coordinates and find spherical basis
        ramses.calc_trans_xyz()
        ramses.spherical_basisvectors()

        ramses.vr = np.sum(ramses.trans_vrel * ramses.er_sphere, axis=0)
        ramses.vθ = np.sum(ramses.trans_vrel * ramses.eθ_sphere, axis=0) 
        ramses.vφ = np.sum(ramses.trans_vrel * ramses.eφ_sphere, axis=0)

        # Run Osyris2Dslab
        east_vector, north_vector, normal_vector = calc_view_vectors(isink, iout, inclination=inclination, rotangle=rotangle)

        custom_viewpoint = {'new_x': east_vector, 
                            'new_y': north_vector, 
                            'view_vector': normal_vector}

        if verbose: print("Creating the 2D slab with Osyris. This might take a while...")
        ramses.osyris2Dslab(variables = ['vr','vθ','vφ'], 
                        viewpoint=custom_viewpoint, 
                        resolution=resolution, 
                        view = width, 
                        dz = dz,
                        weights=['mass','mass','mass'])

        # Make dictionary with values for now
        vel_dict = {}
        vel_dict["radial"] = ramses.osyris_ivs["data1"]["vr"] * ramses.v_cgs * 1e-5 # Should all be in km/s
        vel_dict["inclination"] = ramses.osyris_ivs["data1"]["vθ"] * ramses.v_cgs * 1e-5
        vel_dict["azimuth"] = ramses.osyris_ivs["data1"]["vφ"] * ramses.v_cgs * 1e-5

        with open(path+"/saved_values/"+fname, "wb") as f:
            pickle.dump(vel_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    else:
        with open(path+"/saved_values/"+fname, "rb") as f:
            vel_dict = pickle.load(f)
    
    return (vel_dict["radial"], vel_dict["inclination"], vel_dict["azimuth"])

def radvis_peak_spherical_vels(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, hist_bin_size=None, hist_bin_edges=None, verbose=1):
    '''
    MODIFIED VERSION OF OSYRIS2DSLAB THAT RETURNS THE DENSITY-WEIGHTED PEAK VELOCITY RATHER THAN THE AVERAGED. NOT THOROUGHLY TESTED, USE AT OWN DISCRETION.
    REQUIRES MODIFIED VERSION OF OSYRIS FOUND IN /groups/astro/andreask/python/osyris

    Calculate the peak spherical velocities along the line of sight in the simulated RAMSES data.

    Parameters:
        isink (int): The sink ID
        iout (int): The snapshot ID
        resolution (int): The resolution of the generated image
        width (float): The width in AU of the generated image
        dz (float): The depth in AU of the generated image
        view (string, optional): Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
        inclination (float, optional): The inclination of the viewed image
        rotangle (float, optional): The clockwise rotation around the z-axis of the system
        verbose (bool, default=1): Report task activity
    Returns:
        (3, ndarray) velocities in km/s for the 3 spherical coordinates (radial, inclined, azimuthal)
    '''

    view_str, inclination, rotangle = get_view(view=view, inclination=inclination, rotangle=rotangle, verbose=verbose)
    path = goto_folder(isink, iout)
    fname = "peak_sph_velocities-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)+".p"

    if not os.path.exists(path+"/saved_values/"+fname):
        sink_id, _, datadir = get_sinkdir(isink)
        ramses = radvis.dataclass()        
        ramses.load(snap = iout, io = 'RAMSES', path = datadir, sink_id=sink_id, verbose=verbose, dtype = 'float64')
        # Recalculate the coordinates and find spherical basis
        ramses.calc_trans_xyz()
        ramses.spherical_basisvectors()

        ramses.vr = np.sum(ramses.trans_vrel * ramses.er_sphere, axis=0)
        ramses.vθ = np.sum(ramses.trans_vrel * ramses.eθ_sphere, axis=0) 
        ramses.vφ = np.sum(ramses.trans_vrel * ramses.eφ_sphere, axis=0)

        # Run Osyris2Dslab
        east_vector, north_vector, normal_vector = calc_view_vectors(isink, iout, inclination=inclination, rotangle=rotangle)

        custom_viewpoint = {'new_x': east_vector, 
                            'new_y': north_vector, 
                            'view_vector': normal_vector}

        # Get the bin count from the edges + bin size
        hist_range_kms = hist_bin_edges[1] - hist_bin_edges[0]
        hist_bins = np.ceil(hist_range_kms / hist_bin_size)  # km/s to cm/s -> code units
        hist_range = (hist_bin_edges[0] * 1e5 / ramses.v_cgs, hist_bin_edges[1] * 1e5 / ramses.v_cgs)

        if verbose:
            for component in [ramses.vr, ramses.vθ, ramses.vφ]:
                percentile_1 = np.percentile(component, 1)
                percentile_99 = np.percentile(component, 99)

                if np.any(percentile_1 < hist_range[0]):
                    print("The bottom histogram limit is higher than the 1st percentile of the velocity data.")
                
                if np.any(percentile_99 > hist_range[1]):
                    print("The upper histogram limit is lower than the 99th percentile of the velocity data.")

        if verbose: print("Creating the 2D slab with Osyris. This might take a while...")
        ramses.osyris2Dslab_mod(variables = ["vr", "vθ", "vφ"], 
                            viewpoint=custom_viewpoint, 
                            resolution=resolution, 
                            view = width, 
                            dz = dz,
                            operation = "hist_peak_std",
                            hist_bins=hist_bins, 
                            hist_range=hist_range)

        # Make dictionary with values for now
        vel_dict = {}
        variables = ["vr", "vθ", "vφ"]
        components = ["radial", "inclination", "azimuth"]
        for i, component in enumerate(components):
            vel_dict[component] = {"peak": ramses.osyris_ivs["data1"][variables[i]][0] * ramses.v_cgs * 1e-5,
                                   "stdv": ramses.osyris_ivs["data1"][variables[i]][1] * ramses.v_cgs * 1e-5}

        with open(path+"/saved_values/"+fname, "wb") as f:
            pickle.dump(vel_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    else:
        with open(path+"/saved_values/"+fname, "rb") as f:
            vel_dict = pickle.load(f)
    
    return (vel_dict["radial"]["peak"], vel_dict["inclination"]["peak"], vel_dict["azimuth"]["peak"])

#def radvis_velocities(isink, iout, resolution, width, dz, view=None, inclination=None, rotangle=None, verbose=1):
#    '''
#    Calculate the line-of-sight velocity in the simulated RAMSES data.
#
#    Parameters:
#        isink (int): The sink ID
#        iout (int): The snapshot ID
#        resolution (int): The resolution of the generated image
#        width (float): The width in AU of the generated image
#        dz (float): The depth in AU of the generated image
#        view (string, optional): Built-in viewpoint ('face-on', 'edge-on-a' or 'edge-on-b')
#        inclination (float, optional): The inclination of the viewed image
#        rotangle (float, optional): The clockwise rotation around the z-axis of the system
#        verbose (bool, default=1): Report task activity
#    Returns:
#        (ndarray) Velocity along the line of sight
#    '''
#    view_str, inclination, rotangle = get_view(view=view, inclination=inclination, rotangle=rotangle, verbose=verbose)
#    path = goto_folder(isink, iout)
#    fname = "velocities-"+view_str+"-res"+str(resolution)+"-width"+str(width)+"-dz"+str(dz)+".dat"
#
#    if not os.path.exists(path+"/saved_values/"+fname):
#        sink_id, _, datadir = get_sinkdir(isink)
#        ramses = radvis.dataclass()        
#        ramses.load(snap = iout, io = 'RAMSES', path = datadir, sink_id=sink_id, verbose=verbose, dtype = 'float64')
#        # Recalculate the coordinates
#        ramses.calc_trans_xyz()
#        ramses.vx, ramses.vy, ramses.vz = ramses.trans_vrel
#
#        # Run Osyris2Dslab
#        east_vector, north_vector, normal_vector = calc_view_vectors(isink, iout, inclination=inclination, rotangle=rotangle)
#
#        custom_viewpoint = {'new_x': east_vector, 
#                            'new_y': north_vector, 
#                            'view_vector': normal_vector}
#
#        if verbose: print("Creating the 2D slab with Osyris. This might take a while...")
#        ramses.osyris2Dslab(variables = ['vx','vy','vz'], 
#                        viewpoint=custom_viewpoint, 
#                        resolution=resolution, 
#                        view = width, 
#                        dz = dz,
#                        weights=['mass','mass','mass'])
#        
#        xyz_velocities = np.dstack([ramses.osyris_ivs["data1"]["vx"],ramses.osyris_ivs["data1"]["vy"],ramses.osyris_ivs["data1"]["vz"]])
#        proj_vels = np.empty_like(ramses.osyris_ivs["data1"]["vx"], dtype=float)
#
#        if verbose: print("Projecting velocity components along line-of-sight")
#        for i in range(xyz_velocities.shape[0]):
#            for j in range(xyz_velocities.shape[1]): # Converts to km/s while we're at it
#                proj_vels[i,j] = (np.dot(xyz_velocities[i,j,:], -normal_vector) / np.dot(normal_vector, normal_vector) * ramses.v_cgs * 1e-5)
#
#        np.savetxt(path+"/saved_values/"+fname, proj_vels)
#    else:
#        # Load in the file
#        proj_vels = np.loadtxt(path+"/saved_values/"+fname)
#    
#    return proj_vels