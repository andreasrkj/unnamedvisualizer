import os
from ..sink_config import radmc_datadir

# Function to check folders
def check_folders(path):
    '''Check if the folder for saved values exists'''
    for folder in ["saved_values","saved_plots","saved_images","saved_fits","casa_projects"]:
        if not os.path.exists(path+"/"+folder):
            os.makedirs(path+"/"+folder)
            print("Sucessfully created folder " + folder)
    # Create the subfolders for organizing plots
    for folder in ["TauMap","ColumnDensity","Velocities","MomentMaps","SingleWav","ChannelMaps","TemperatureMap"]:
        if not os.path.exists(path+"/saved_plots/"+folder):
            os.makedirs(path+"/saved_plots/"+folder)
            print("Succesfully created subfolder " + folder)

# This function generates the casa project name (creates long files - must be shortened)
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

# This function finds the given data folder, assuming it's in the same folder as the sink folder
def goto_folder(isink, iout, directory=radmc_datadir):
    '''Returns the path to the data folder of the given sink and output folder calculated by RADMC-3D.
    Takes input: 
    isink (integer) - sink number, 
    iout (integer) - output folder'''
    if isinstance(isink, float):
        sink_id, level = str(isink).split(".")
        sink = "{:03}".format(int(sink_id))+"_"+level
    else:
        sink = "{:03}".format(isink)
    
    if isinstance(iout, str): # If specific string just use that instead of the int
        path = os.path.join(directory, 'sink'+str(sink)+'/'+iout)
    else:
        out  = '{:04}'.format(iout)
        path = os.path.join(directory, 'sink'+str(sink)+'/nout'+out)
    return path

# This function checks whether we're in the data folder, otherwise it moves us there. Good for separating plotting and data generation functions.
def check_in_folder(path):
    '''Checks whether the program is in the correct folder
    Takes input:
    path (string) - path to the data directory'''
    if path not in os.getcwd():
        os.chdir(path) # Function assumes you're in the parent directory (might be a problem?)

def get_view(view=None, inclination=None, rotangle=None, verbose=1):
    '''
    Get the view string to generate or load an image made with this package.

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