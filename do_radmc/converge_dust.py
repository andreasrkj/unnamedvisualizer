import os, sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from .run_radmc import doRADMC
from .create_sinkcell import create_hdfs
from ..sink_config import radmc_datadir, sink_dirs, mesa_dirs
from ..helper_functions import goto_folder
sys.path.insert(0,'/groups/astro/andreask/python')
from radmc3dPy_SSJ import analyze

def calc_dusttemp(isink, iout, sizeCutout, setthreads, overwrite=False, output_analysis=True):
    path = goto_folder(isink, iout)

    # Check whether to proceed with the run ??
    # If we don't want to overwrite and it already exists, we end the function
    if not overwrite and os.path.exists(os.path.join(path, "dust_temperature.bdat")):
        print(f"The dust temperature has already been calculated for sink {isink} snapshot {iout}")
        print("If you wish to overwrite this file, please call the function with 'overwrite' = True.")
        return None

    # Possible nphot levels
    nphot_levels = np.array([1e6, 1e7, 2.5e7, 5e7, 1e8, 1.5e8, 2e8])

    # Hopefully run through SLURM
    if os.environ["SLURM_CPUS_PER_TASK"]:
        threads = int(os.environ["SLURM_CPUS_PER_TASK"])
    elif os.environ["OMP_NUM_THREADS"]:
        threads = int(os.environ["OMP_NUM_THREADS"])
    else:
        threads = setthreads
    print("Running with ", threads, "threads")

    scales_file = os.path.join(radmc_datadir, 'scales_IMF512.fits')

    # Make cell.hdf5 and sink.hdf5
    if os.path.exists(os.path.join(path, "cell.hdf5")) and os.path.exists(os.path.join(path, "sink.hdf5")):
        print("cell.hdf5 and sink.hdf5 already exists. Skipping this step...")
    else:
        print(f"Creating sink structure for sink {isink}, snapshot {iout}")
        create_hdfs(directory=radmc_datadir, nsink=isink, nout=iout, datadir=sink_dirs[str(isink)][0], sizeCutout=sizeCutout)

    # The 1 million nphot run to compare the 10 million run to!
    print("Running initial nphot = 1 million")
    doRADMC(directory=radmc_datadir, nsink=isink, nout=iout, datadir=sink_dirs[str(isink)][0], mesadir=mesa_dirs[str(isink)], scales_file=scales_file, nsink2=None, threads=threads, nphot=1000000)

    # Analyze the data for <5 K cells (undervisited cells)
    origin = os.getcwd()
    # We have to move into the folder to use analyze
    try:
        os.chdir(path)
        grid = analyze.readGrid()
        data = analyze.readData(gdens=True, dtemp=True, ispec='co', grid=grid)
        prev_temp = data.dusttemp.flatten()
        os.chdir(origin)
    except:
        os.chdir(origin)
        raise OSError("Something went wrong with the analyze function")

    # Check the number of temperature values under 5 K (~ expected lower limit of background temperature)
    zero_cells = len(prev_temp[prev_temp <= 5])

    # Initialize the while loop
    relative_error = 1; lvl=0

    if output_analysis:
        err_array = [relative_error]
        zero_array = [zero_cells]

        fig = plt.figure(figsize=(20,12))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(2,2,4, sharex = ax2)
        ax = [ax1,ax2,ax3]

        ax[0].hist(prev_temp, bins=int(np.sqrt(len(prev_temp))), label=f"nphot = {nphot_levels[lvl]*1e-6}$\\times {10}^{6}$", alpha=0.5, histtype="step", density=True)

    # While we have >20 cells with < 5K temp or (for large number of cells) more than 0.001% of cells are < 5K
    while (zero_cells > 20 or zero_cells / len(prev_temp) > 1e-5) or relative_error > 0.05:
        # Check whether to continue loop
        if lvl == len(nphot_levels)-1:
            print("nphot limit reached. Terminating loop...")
            break
        else:
            lvl += 1
            print("RADMC-3D didn't converge. Re-running with new level nphot = ", nphot_levels[lvl])

        # After creating the sink and cell structure, we'll run RADMC-3D (initially for 10M photons)
        doRADMC(directory=radmc_datadir, nsink=isink, nout=iout, datadir=sink_dirs[str(isink)][0], mesadir=mesa_dirs[str(isink)], scales_file=scales_file, nsink2=None, threads=threads, nphot=int(nphot_levels[lvl]))

        # Analyze the data for convergence
        origin = os.getcwd()
        try:
            os.chdir(path)
            grid = analyze.readGrid()
            data = analyze.readData(gdens=True, dtemp=True, ispec='co', grid=grid)
            temp = data.dusttemp.flatten()
            os.chdir(origin)
        except:
            os.chdir(origin)
            raise OSError("Something went wrong with the analyze function")

        # Check the number of temperature values under 5 K (~ expected lower limit of background temperature)
        zero_cells = len(temp[temp <= 5])

        # Check the relative error (cannot divide by zero, so these are left out in the calculation (saved by zero cell criteria))
        relative_error = np.sqrt(np.mean(((temp[temp > 0] - prev_temp[temp > 0]) / temp[temp > 0])**2))
        print(f"REPORT nphot = {nphot_levels[lvl]} | Undervisited Cells: {zero_cells} or {np.round(zero_cells/len(temp)*100,5)}%, Relative Error: {relative_error}")
        if output_analysis:
            ax[0].hist(temp, bins=int(np.sqrt(len(temp))), label=f"nphot = {nphot_levels[lvl]*1e-6}$\\times {10}^{6}$", alpha=0.5, histtype="step", density=True)
            err_array.append(relative_error)
            zero_array.append(zero_cells)
        prev_temp = np.copy(temp)

    final_nphot = nphot_levels[lvl]
    print("Finished running RADMC-3D with final nphot = ", final_nphot)

    if output_analysis:
        ax[1].plot(nphot_levels[:lvl+1]*1e-6, err_array, 'o--')
        ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax[1].fill_between([-2,202], -0.01, 0.05, color="lightgreen")
        ax[1].set_xlim(-2,nphot_levels[:lvl+1][-1]*1e-6+2); ax[1].set_ylim(-0.01, 0.2)
        ax[2].plot(nphot_levels[:lvl+1]*1e-6, zero_array, 'o--')
        # The inset only makes sense if we don't immediately converge at 10 million
        if len(zero_array) >= 3:
            if zero_array[-2] == 0 and zero_array[-1] == 0:
                insax = ax[2].inset_axes([0.4, 0.5, 0.55, 0.45], xlim=(nphot_levels[lvl-1]*1e-6-2, nphot_levels[lvl]*1e-6+2), ylim=(-2, 2), transform=ax[2].transAxes)
            else:
                insax = ax[2].inset_axes([0.4, 0.5, 0.55, 0.45], xlim=(nphot_levels[lvl-1]*1e-6-2, nphot_levels[lvl]*1e-6+2), ylim=(-2, zero_array[-2]+2), transform=ax[2].transAxes)
            insax.plot(nphot_levels[:lvl+1]*1e-6, zero_array, 'o--')
            insax.tick_params(labelsize=14)
            ax[2].indicate_inset_zoom(insax, edgecolor="black")

        ax[0].set_xlabel("Dust Temperature [K]", fontsize=18); ax[0].set_ylabel("Percentage of total cell count", fontsize=18)

        ax[1].set_ylabel("Relative Error", fontsize=18); ax[2].set_ylabel("Number of cells with $T < 5$ K", fontsize=18)
        ax[2].set_xlabel("Number of million photon packages", fontsize=18)
        for iax in ax:
            iax.tick_params(labelsize=14)
            iax.grid()
        ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].legend(fontsize=16)
        fig.suptitle(f"Sink {isink}, Snapshot {iout}", fontsize=22, y=0.92)
        plt.savefig(path+"/radmc3d-convergence-analysis.png", bbox_inches="tight")

    return final_nphot