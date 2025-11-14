# Import functions necessary to run
import os, sys
import pickle
from .sink_config import sink_dirs
import astropy.constants as cnst
import astropy.units as unit
import astropy.io.fits as fits
sys.path.insert(0,'/groups/astro/troels/python')
sys.path.insert(0,'/groups/astro/troels/python/sfrimann')
sys.path.insert(0,'/groups/astro/troels/python/sigurd')
sys.path.insert(0,"/groups/astro/andreask/python")
import pyradmc3d as pyrad
from radmc3dPy_SSJ import image
from radmc3dPy_SSJ import analyze


import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Import functions from main
from .main import *
from .data_generation import *

# Class for loading in image data from FITS file when saved
#class ImageClass:
#    def __init__(self):
#        return None

#class casaImageClass:
#    def __init__(self, isink, iout, image_name, dpc, antennalist=None):
#        self.path = goto_folder(isink, iout)
#        # Load in the data and assign keywords
#        if antennalist is not None:
#            # Transform given image name to simalma image name
#            if len(antennalist) > 1:
#                config_str = "combined"+"_".join(['alma.cycle7.8.cfg', 'alma.cycle7.5.cfg']).replace("alma.cycle","").replace(".cfg","")
#            else:
#                config_str = antennalist[0]
#            data, header = fits.getdata(self.path+"/saved_fits/simalma_"+config_str+"_"+image_name+".fits", header=True)
#            # Load in the beam and RMS value if CASA image
#            stats = get_stats(self.path, image_name, antennalist)
#            self.beam = (header["BMAJ"]/header["CDELT1"], header["BMIN"]/header["CDELT2"], header["BPA"]) # beam size in px
#            self.rms = stats["rms"]
#        else:
#            data, header = fits.getdata(self.path+"/saved_fits/"+image_name+".fits", header=True)
#        if header["NAXIS3"] > 1: # If multi-wavelength
#            self.image = data[0,:,::-1,:].transpose((1,2,0)) # Aligned with the axis that radmc3dPy loads in with
#        else:
#            self.image = data[0,0,:,:].transpose(1,0)
#        # Assign header keywords
#        self.x = (np.arange(1,header["NAXIS1"]+1) - header["CRPIX1"]) * np.abs(header["CDELT1"]) * np.pi/180 * dpc * unit.pc.to(unit.cm)
#        self.y = (np.arange(1,header["NAXIS2"]+1) - header["CRPIX2"]) * np.abs(header["CDELT2"]) * np.pi/180 * dpc * unit.pc.to(unit.cm)
#        self.nx = len(self.x)
#        self.ny = len(self.x)
#        self.sizepix_x = np.abs(header["CDELT1"] * np.pi/180 * dpc * unit.pc.to(unit.cm))
#        self.sizepix_y = np.abs(header["CDELT2"] * np.pi/180 * dpc * unit.pc.to(unit.cm))
#        self.freq = np.linspace(start=header["CRVAL3"], stop=(header["NAXIS3"]-1)*header["CDELT3"]+header["CRVAL3"], num=header["NAXIS3"])
#        self.nfreq = len(self.freq)
#        self.wav = (cnst.c / (self.freq * unit.Hz)).to(unit.micron).value
#        self.nwav = len(self.wav)
    
#    def plot_image(self, log=False, ifreq=None, ax=None):
#        # Error handling
#        if self.nfreq > 1 and ifreq is None:
#            raise ValueError("This is a multi-wavelength image. Specify 'ifreq' keyword if a wavelength is wanted.")
#        
#        if not ax: # Create a figure if not supplied
#            fig, ax = plt.subplots(1,1, figsize=(8,10))
#
#        if log:
#            cb_label = "$\\log(I_\\nu / \\mathrm{max}(I_\\nu))$"
#            plot_img = np.log10(self.image / self.image.max())
#        elif antennalist is not None:
#            cb_label = "[mJy/beam]"
#            plot_img = img.image*1e3
#            else:
#                cb_label = "[$\\mathrm{erg/s/cm^2/Hz/ster}$]"
#                plot_img = img.image
#
#        if not vmax:
#            vmax = plot_img.max()
#            extend="neither"
#        if vmax < plot_img.max():
#            extend="max"
#        else:
#            extend="neither"
#
#        im = ax.imshow(plot_img, cmap="magma", vmax=vmax, extent=(-sizeau/2,sizeau/2,-sizeau/2,sizeau/2))
#        plt.colorbar(im, ax=ax, label=cb_label, pad=0, orientation="horizontal", location="top", extend=extend)
#
#        # Remove axes
#        ax.xaxis.label.set_visible(False); ax.yaxis.label.set_visible(False)
#        ax.set_yticklabels([]); ax.set_yticks([])
#        ax.set_xticklabels([]); ax.set_xticks([])
#        
#        # Create scale bar
#        # We should normalize the distances to the edges
#        bar_length = 25 + 25 * (sizeau // 250)
#        end_point = 4/5 * sizeau//2
#
#        ax.hlines(-375/500 * sizeau//2, end_point - bar_length, end_point, color="white", linestyles="solid", linewidths=3)
#        ax.text(end_point - bar_length/2, -375/500 * sizeau//2 -sizeau//2*20/500, str(bar_length)+" AU", ha="center", va='top', color="white", fontsize=20, weight="heavy")
#        ax.text(0, 49/50 * sizeau//2, "$\\lambda = "+str(np.round(wav,2))+"$µm", ha="center", va="top", color="white", fontsize=18)
#
#        if not ax: 
#            if verbose: print("Outputting image plot as .png")
#            plt.savefig(path+"/saved_plots/SingleWav/image-"+view_str+"-singlewav-"+str(sizeau)+"au-"+str(int(wav))+"mu.png", bbox_inches="tight")
#        return None
#
#    def plot_moment_map(self):
#        return None
#    
#    def plot_channel_map(self):
#        return None