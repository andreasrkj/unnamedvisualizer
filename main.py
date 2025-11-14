import os, sys, re
import subprocess as sp
import numpy as np
from .sink_config import radmc_datadir

# Function to check folders
def check_folders(path):
    '''Check if the folder for saved values exists'''
    if not os.path.exists(path+"/saved_values"): # If we haven't previously made a directory to store values
        os.makedirs(path+"/saved_values") # We'll create one to store values like the north vector
        print("Succesfully created folder 'saved_values'")
    if not os.path.exists(path+"/saved_plots"):
        os.makedirs(path+"/saved_plots")
        print("Succesfully created folder 'saved_plots'")
    # Create the subfolders for organizing plots
    for folder in ["TauMap","TauContour","ColumnDensity","Velocities","MomentMaps","SingleWav","MolComparison",
                   "ViewComparison","TauComparison","ChannelMaps"]:
        if not os.path.exists(path+"/saved_plots/"+folder):
            os.makedirs(path+"/saved_plots/"+folder)
            print("Succesfully created subfolder " + folder)
    if not os.path.exists(path+"/saved_images"):
        os.makedirs(path+"/saved_images")
        print("Succesfully created folder 'saved_images'")
    if not os.path.exists(path+"/saved_fits"):
        os.makedirs(path+"/saved_fits")
        print("Succesfully created folder 'saved_fits'")
    if not os.path.exists(path+"/casa_projects"):
        os.makedirs(path+"/casa_projects")
        print("Succesfully created folder 'casa_projects'")

def run_radmc(isink, iout, override_input=False):
    # Run the RADMC-3D setup in "doradmc3dCO.py" on the given file
    if not override_input: 
        a = input("Enter y/n whether you want RADMC-3D to run (Highly recommended not to do in a notebook)")
        if a.lower() == "y":
            run = True
        elif a.lower() == "n":
            print(f"RADMC-3D will not run for sink {isink}, output {iout}")
            run = False
        else:
            print("Enter either y/n")
            run = False
    if override_input or run:
        if override_input: print("Input option overriden.")
        # Create environment dictionary to pass on
        d = dict(os.environ)
        d["NSTART"] = str(iout)
        d["NEND"]   = str(iout)

        # Let's check first if the sink.hdf5 and cell.hdf5
        try:
            completed = sp.run([sys.executable, "create_sink_structure.py"],
                                env=d, capture_output=True, text=True, check=True)
            print("RADMC-3D finished successfully.")
            if completed.stdout:
                print(completed.stdout)
            if completed.stderr:
                print(completed.stderr)
        except sp.CalledProcessError as e:
            print("Sink & cell file structure failed with returncode", e.returncode)
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)

        # Then run RADMC-3D
        print(f"RADMC-3D will now run for sink {isink}, output {iout}")
        try:
            completed = sp.run([sys.executable, "doradmc3dCO.py"],
                                env=d, capture_output=True, text=True, check=True)
            print("RADMC-3D finished successfully.")
            if completed.stdout:
                print(completed.stdout)
            if completed.stderr:
                print(completed.stderr)
        except sp.CalledProcessError as e:
            print("RADMC-3D invocation failed with returncode", e.returncode)
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)

# This function checks if the files have been run through RADMC previously
def check_radmc(isink, iout, override_input=False):
    path = goto_folder(isink, iout)

    if not os.path.exists(path+"/radmc3d.inp") and not os.path.exists(path+"/radmc3d.out"):
        print("This folder doesn't have the RADMC-3D data necessary to make images.")
        run_radmc(isink, iout, override_input)
    else:
        print("RADMC-3D has already run for this file and image generation can proceed...")

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

def list_files(isink, iout):
    '''This function lists the already generated files in each directory.'''
    path = goto_folder(isink, iout)
    os.chdir(path)
    check_folders() # Check if the folders exist :)
    files = os.listdir("saved_values")
    print(f"LISTING OUTPUT FILES FOR NSINK: {isink} | OUTPUT FILE {iout}")

    # General saved values
    general_values = []
    print("================================= GENERAL VALUES =================================")
    print("        FILE         SAVED?")
    if 'coordinate_basis.dat' in files:
        general_values.append(["COORD BASIS", "✓"])
    else:
        general_values.append(["COORD BASIS", "X"])
    for row in general_values:
            print('    {:>}        {:>}'.format(general_values[0][0], general_values[0][1]))

    # Printing column densities
    print("============================= SAVED COLUMN DENSITIES =============================")
    identifiers = '  DISK VIEW   RESOLUTION   WIDTH   DZ (DEPTH)'
    column_densities = []
    for i in range(len(files)):
        if "column-density" in files[i]:
            if "edge-on" in files[i]:
                characteristics = ["Edge on"]
            elif 'face-on' in files[i]:
                characteristics = ["Face on"]
            characteristics.append([int(s) for s in re.findall(r'\d+', files[i])])
            column_densities.append(characteristics)
    if not column_densities:
        print("No column densities calculated...")
    else:
        print(identifiers)
        for row in column_densities:
            print('    {:>}         {:>4}   {:>5}         {:>4}'.format(row[0], row[1][0], row[1][1], row[1][2]))

    # Print velocities
    print("================================ SAVED VELOCITIES ================================")
    velocities = []
    for i in range(len(files)):
        if "velocities" in files[i]:
            if "edge-on" in files[i]:
                characteristics = ["Edge on"]
            elif 'face-on' in files[i]:
                characteristics = ["Face on"]
            characteristics.append([int(s) for s in re.findall(r'\d+', files[i])])
            column_densities.append(characteristics)
    if not column_densities:
        print("No velocities calculated...")
    else:
        print(identifiers)
        for row in column_densities:
            print('    {:>}         {:>4}   {:>5}         {:>4}'.format(row[0], row[1][0], row[1][1], row[1][2]))

    # Print energy distribution
    print("========================== SPECTRAL ENERGY DISTRIBUTION ==========================")
    seds = []
    for i in range(len(files)):
        if "sed" in files[i]:
            if "edge-on-A" in files[i]:
                characteristics = ["Edge on A"]
            elif "edge-on-B" in files[i]:
                characteristics = ["Edge on B"]
            elif 'face-on' in files[i]:
                characteristics = ["Face on"]
            seds.append(characteristics)
    if not column_densities:
        print("No SEDs calculated...")
    else:
        for sed in seds:
            print("SED created for {:>}".format(sed[0]))

    # Print RADMC3D images
    files = os.listdir("saved_images")
    print("================================= RADMC3D IMAGES =================================")
    identifiers = "   IMAGE TYPE   DISK VIEW   AU SIZE   λ (MICRON)   TRANSITION   WIDTHKMS   # OF λ"
    # First we find single wavelength images
    single_wavs = []
    for i in range(len(files)):
        if "singlewav" in files[i]:
            characteristics = ["Single λ"]
            if "edge-on-A" in files[i]:
                characteristics.append("Edge on A")
            elif "edge-on-B" in files[i]:
                characteristics.append("Edge on B")
            elif 'face-on' in files[i]:
                characteristics.append("Face on")
            characteristics.append([int(s) for s in re.findall(r'\d+', files[i])])
            single_wavs.append(characteristics)
    if single_wavs:
        print(identifiers)
        for row in single_wavs:
            print('     {:>}     {:>}     {:>5}         {:>4}'.format(row[0], row[1], row[2][0], row[2][1]))
    CO_imgs = []
    for i in range(len(files)):
        if "image-co" in files[i]:
            characteristics = ["CO Image"]
            if "edge-on-A" in files[i]:
                characteristics.append("Edge on A")
            elif "edge-on-B" in files[i]:
                characteristics.append("Edge on B")
            elif 'face-on' in files[i]:
                characteristics.append("Face on")
            characteristics.append([int(s) for s in re.findall(r'\d+', files[i])])
            CO_imgs.append(characteristics)
    if CO_imgs:
        for row in CO_imgs:
            print('     {:>}   {:>}     {:>5}                         {:>1}         {:>2}       {:>2}'.format(row[0], 
                                                                                                                row[1], 
                                                                                                                row[2][0], 
                                                                                                                row[2][1], 
                                                                                                                row[2][2], 
                                                                                                                row[2][3]))
    if not CO_imgs and not single_wavs:
        print("No RADMC-3D images generated...")
    os.chdir("../../")
    