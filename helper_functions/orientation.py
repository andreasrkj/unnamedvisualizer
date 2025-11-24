import os, sys
from ..sink_config import sink_dirs
from .data_structure import goto_folder, check_folders, get_view
import numpy as np
import RaDvisPython as radvis

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

    # First we check if the data file already exists
    if os.path.exists(path+"/saved_values/coordinate_basis.dat"):
        print("Coordinate basis already generated, loading from file...")
        spin_vector, plane_vector1, plane_vector2 = np.loadtxt(path+"/saved_values/coordinate_basis.dat", unpack=True)
    else:
        check_folders(path) # Check if folder exists
        print("Coordinate basis doesn't exist for this file, generating...")
        # Load the data from the "master" folder
        ramses = radvis.dataclass()
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
    # load spin vector and coordinate basis for the disk from snapshot
    spin_vector, x_vector, y_vector = calc_coord_basis(isink, iout) # # z, x, y

    # Use face-on as the zero point

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