import sys
sys.path.insert(0,'/groups/astro/troels/python')
sys.path.insert(0,'/groups/astro/troels/python/sfrimann')
import numpy as np
import pyramses as pyram
from astropy import constants as con
import os
from astropy.io import fits as pyfits
from ..sink_config import sink_dirs

from ramses_accretion import calculate_dm
from multiprocessing import Pool

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

Lsun = con.L_sun.cgs.value
cc = con.c.cgs.value
hh = con.h.cgs.value
kb = con.k_B.cgs.value
spy = 365.25*24*3600
#
# -----------------------------------------------------------------------

def create_hdfs(directory, nsink, nout, datadir="../data", returnContainers=False, full_oct=True, sizeCutout=10000,
                scales_file="./scales_IMF512.fits"):
    '''
        Create sink, cell, and particle file if needed.
    '''
    try:
        sinks = pyram.rsink(nout, datadir=datadir)
    except IOError:
        print("rsink could not load")
        return

    try:
        _ = sinks['x'][sink_dirs[str(nsink)][1]]
    except IndexError:
        print('sink not formed yet')
        return

    # ------------------------------
    print("Creating sink, cell, and particle file for nsink %i, nout %i" % (nsink, nout))
    datadir = os.path.join(directory, datadir)
    info_file = os.path.join(
        datadir, 'output_{0:05d}/info_{0:05d}.txt'.format(nout))
    info = pyram.rd_info(info_file)
    lvlmax = info['levelmax']

    if not os.path.exists('sink{:03d}/nout{:04d}'.format(nsink, nout)):
        os.makedirs('sink{:03d}/nout{:04d}'.format(nsink, nout))

    noutdirectory = os.path.join(
        directory, './sink{:03d}/nout{:04d}'.format(nsink, nout))

    noutmin = nout - 1
    noutmax = nout + 1
    scales_fits = os.path.join(directory, scales_file)
    header = pyfits.getheader(scales_fits, 0)
    starsdat = [pyram.rsink(n, datadir=datadir)
                for n in range(noutmin, noutmax+1)]

    # ------------------------------
    # create sink
    sink = pyram.Sink(nout, datadir=datadir, scales_file=scales_fits)
    sink.rsink()
    dm_manual, accretion_window = calculate_dm(sinks, starsdat)
    sink.dm_manual = dm_manual  # add manual accretion rate calculations TODO add own values?
    sink.accretion_window = accretion_window
    reference_sink = sink.reference_sink(sink_dirs[str(nsink)][1])  # Calculate reference
    sink.reference = reference_sink
    sink.save(directory=noutdirectory)

    # ------------------------------
    # create cell
    cell = pyram.Cell(nout, datadir=datadir, scales_file=scales_fits,
                      reference=reference_sink, lvlmax=lvlmax)
    cell.amr2cell(dr='{:d} AU'.format(sizeCutout), silent=True)
    cellid = cell.amrsort(full_oct=full_oct)

    cell.save(directory=noutdirectory)

    if returnContainers:
        return sink, cell

def process_data(i):
    #nsink = 24   # primary sink for run sink_025
    #datadir='/groups/astro/andreask/zoomin_data/sink_025/data' # where to find data for run
    
    # Try with sink 13
    nsink = 13
    datadir="/lustre/astro/troels/IMF_512_cores/sink_13/data/"
    create_hdfs('.', nsink, i, datadir=datadir, sizeCutout=20000) # sizecutout changed to 20000 AU for ability to make images over a greater region.

# -----------------------------------------------------------------------
if __name__ == '__main__':
    nstart = int(os.environ["NSTART"])
    nend   = int(os.environ["NEND"])

    pool = Pool(12)                         # Create a multiprocessing Pool
    pool.map(process_data, range(nstart,nend+1))  # process data_inputs iterable with pool

    pool.close()
    pool.join()
