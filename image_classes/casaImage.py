from ..helper_functions import goto_folder, get_casa_project_name
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.offsetbox import AuxTransformBox, AnchoredOffsetbox
import astropy.constants as cnst
import astropy.units as unit
import astropy.io.fits as fits

plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Image class for handling all interactions with a CASA image
class casaImageClass:
    '''
    A class for loading in RADMC-3D + simalma images
    '''
    def __init__(self, isink, iout, image_name, dpc, antennalist, printname=None, printtrans=None):
        # Initial loadings
        self.path = goto_folder(isink, iout)
        self.fname = image_name
        self.dpc = dpc # Distance to source in pc
        if printname is not None:
            self.mol_name = printname
            self.transition = printtrans
        
        #project_name = get_casa_project_name(self.fname)

        # Check whether to load in combined or single config observations
        if len(antennalist) > 1:
            config_str = "combined"+"_".join(antennalist).replace("alma.cycle","").replace(".cfg","")
            #stats = imstat(imagename=self.path+"/casa_projects/"+project_name+"/"+project_name+".concat.image.pbcor")
        else:
            config_str = antennalist[0].replace("cfg","noisy")
            #stats = imstat(imagename=self.path+"/casa_projects/"+project_name+"/"+project_name+"."+str(antennalist[0]).replace(".cfg","")+".noisy.image.pbcor")

        # Load in the fits file
        data, header = fits.getdata(self.path+"/saved_fits/simalma_"+config_str+"_"+self.fname+".fits", header=True)
        # Load in the beam and RMS value if CASA image
        self.beam_px     = (header["BMAJ"]/header["CDELT1"], header["BMIN"]/header["CDELT2"], header["BPA"]) # beam size in px
        self.beam_arcsec = (header["BMAJ"]*3600, header["BMIN"]*3600, header["BPA"])
        #self.rms = stats["rms"]

        #if header["NAXIS3"] > 1: # If multi-wavelength
        #    self.image = data[0,:,::-1,:].transpose((1,2,0)) # Aligned with the axis that radmc3dPy loads in with
        #else:
        #    self.image = data[0,0,:,:].transpose(1,0)
        self.image = data[0,:,:,:].transpose((1,2,0))

        # Let's calculate the image background by taking the first two images, subtracting them and taking the mean std (should be true noise?)
        self.rms = np.std(self.image[:,:,2]-self.image[:,:,1])

        # Assign header keywords
        self.x = (np.arange(1,header["NAXIS1"]+1) - header["CRPIX1"]) * np.abs(header["CDELT1"]) * np.pi/180 * dpc * unit.pc.to(unit.cm)
        self.y = (np.arange(1,header["NAXIS2"]+1) - header["CRPIX2"]) * np.abs(header["CDELT2"]) * np.pi/180 * dpc * unit.pc.to(unit.cm)
        self.nx = len(self.x)
        self.ny = len(self.x)
        self.sizepix_x = np.abs(header["CDELT1"] * np.pi/180 * dpc * unit.pc.to(unit.cm))
        self.sizepix_y = np.abs(header["CDELT2"] * np.pi/180 * dpc * unit.pc.to(unit.cm))
        self.sizeau = (self.sizepix_x * header["NAXIS1"] * unit.cm).to(unit.AU).value
        self.freq = np.linspace(start=header["CRVAL3"], stop=(header["NAXIS3"]-1)*header["CDELT3"]+header["CRVAL3"], num=header["NAXIS3"])
        self.nfreq = len(self.freq)
        self.wav = (cnst.c / (self.freq * unit.Hz)).to(unit.micron).value
        self.nwav = len(self.wav)
        if self.nfreq % 2 == 0:
            self.nu0 = self.freq[self.nfreq//2-1]/2 + self.freq[self.nfreq//2]/2
        else:
            self.nu0 = self.freq[self.nfreq//2]

    def _stylize_plot(self, ax, plot_text=None, color="white", text_size=18):
        # Remove axes
        ax.xaxis.label.set_visible(False); ax.yaxis.label.set_visible(False)
        ax.set_yticklabels([]); ax.set_yticks([])
        ax.set_xticklabels([]); ax.set_xticks([])
        
        # Create scale bar
        # We should normalize the distances to the edges
        plot_size = np.abs(ax.get_xlim()[1] - ax.get_xlim()[0])
        bar_length = int(25 + 25 * (plot_size // 250))
        end_point = 4/5 * plot_size//2

        ax.hlines(-375/500 * plot_size//2, end_point - bar_length, end_point, color=color, linestyles="solid", linewidths=3)
        ax.text(end_point - bar_length/2, -375/500 * plot_size//2 -plot_size//2*20/500, str(bar_length)+" AU", ha="center", va='top', color=color, fontsize=text_size)
        if plot_text is not None: ax.text(0, 49/50 * plot_size//2, plot_text, ha="center", va="top", color=color, fontsize=text_size)

        # Add beam
        aux_tr_box = AuxTransformBox(ax.transData)
        aux_tr_box.add_artist(Ellipse((0,0), self.beam_px[0] * self.sizepix_x/unit.AU.to(unit.cm), self.beam_px[1] * self.sizepix_y/unit.AU.to(unit.cm), self.beam_px[2], color="black"))
        box = AnchoredOffsetbox(child=aux_tr_box, loc="lower left", frameon=True)
        ax.add_artist(box)

    def plot_singlewav(self, log=False, ifreq=None, mask=False, vmin=None, vmax=None, ax=None, xlim=None, ylim=None, save=False):
        if not ax: # Create a figure if not supplied
            fig, ax = plt.subplots(1,1, figsize=(8,10))
        else:
            if save: print("Note, you've supplied a matplotlib axis while setting 'save' = True, this may create a weird-looking plot in the .png")

        # Error handling
        if self.nfreq > 1 and ifreq is None:
            raise ValueError("This is a multi-wavelength image. Specify 'ifreq' keyword if a wavelength is wanted.")
        elif self.nfreq > 1 and ifreq is not None:
            # Calculate distance from line center
            v_kms = cnst.c.value * (self.nu0 - self.freq[ifreq]) / self.nu0 / 1e3
            plot_text = self.mol_name+" J="+str(self.transition[0])+"-"+str(self.transition[1])+" transition @ "+str(np.round(v_kms,2))+" km/s"
            plot_img = self.image[:,:,ifreq]*1e3
            cmap = "Spectral_r"
            save_name = self.fname + "ifreq"+str(ifreq)
            cb_label = "[mJy/beam]"

        else:
            plot_text = "$\\lambda = "+str(np.round(self.wav[0],2))+"$µm"
            cmap = "magma"
            save_name = self.fname
            if log:
                # If log-scale and CASA, we'll probably have negative values...
                cb_label = "$\\log(I_\\nu / \\mathrm{max}(I_\\nu))$"
                plot_img = np.log10(self.image / self.image.max())
                extend="neither"
            else:
                cb_label = "[mJy/beam]"
                plot_img = self.image*1e3

        if mask:
            plot_img = np.ma.masked_less_equal(plot_img, 3*self.rms * 1e3) # Mask image

        if vmin is None: vmin = plot_img.min()
        if vmax is None: vmax = plot_img.max()

        if vmin > plot_img.min():
            extend = "min"
        elif vmax < plot_img.max():
            extend="max"
        elif (vmin > plot_img.min()) and (vmin < plot_img.max()):
            extend="both"
        else:
            extend="neither"

        im = ax.imshow(plot_img, cmap=cmap, vmin=vmin, vmax=vmax, extent=(-self.sizeau/2,self.sizeau/2,-self.sizeau/2,self.sizeau/2), origin="lower")
        cbar = plt.colorbar(im, ax=ax, pad=0, orientation="horizontal", location="top", extend=extend)
        cbar.set_label(cb_label, size=20)
        cbar.ax.tick_params(labelsize=14)

        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

        self._stylize_plot(ax, plot_text)

        if save: 
            print("Outputting image plot as .png")
            plt.savefig(self.path+"/saved_plots/SingleWav/"+save_name+".png", bbox_inches="tight")

    def calc_moment(self, moment=0):
        if self.nfreq < 2:
            raise ValueError("Cannot create moment map for a single wavelength image")

        # This part of the program is now appropriated from radmc3dPy !!
        # Calculate velocity field
        v_kms = cnst.c.value * (self.nu0 - self.freq) / self.nu0 / 1e3

        if moment in [0,1,2]:
            vmap = np.zeros([self.nx, self.ny, self.nfreq], dtype=np.float64)
            for ifreq in range(self.nfreq):
                vmap[:, :, ifreq] = v_kms[ifreq]

            # Now calculate the moment map
            y = self.image * (vmap**moment)

            dum = (vmap[:, :, 1:] - vmap[:, :, :-1]) * (y[:, :, 1:] + y[:, :, :-1]) * 0.5

            mmap = dum.sum(2)

            if moment > 0:
                y = self.image
                dum0 = (vmap[:, :, 1:] - vmap[:, :, :-1]) * (y[:, :, 1:] + y[:, :, :-1]) * 0.5
                
                mmap0 = dum0.sum(2)
                mmap = mmap / mmap0

        elif moment == 8:
            mmap = self.image.max(axis=2)

        elif moment == 9:
            mmap = v_kms[np.argmax(self.image, axis=2)]
        
        else:
            raise ValueError("Cannot create moment maps other than 0, 1, 2, 8, 9")

        return mmap

    def plot_moment(self, moment=0, mask=True, vmin=None, vmax=None, ax=None, xlim=None, ylim=None, save=False):
        if ax is None: # Create a figure if not supplied
            fig, ax = plt.subplots(1,1, figsize=(8,10))
        else:
            if save: print("Note, you've supplied a matplotlib axis while setting 'save' = True, this may create a weird-looking plot in the .png")

        mmap = self.calc_moment(moment)

        # Mask values
        if mask:
            if moment in [0,8]:
                mmap = np.ma.masked_less_equal(mmap, 3*self.rms)
            elif moment in [1,2]:
                mmap0 = np.ma.masked_less_equal(self.calc_moment(moment=0), 3*self.rms)
                mmap = np.ma.masked_where(mmap0 <= 3*self.rms, mmap)
            elif moment == 9:
                mmap8 = np.ma.masked_less_equal(self.calc_moment(moment=8), 3*self.rms)
                mmap = np.ma.masked_where(mmap8 <= 3*self.rms, mmap)

        # Set plot labels and colorbar
        if moment == 0:
            cmap = "Spectral_r"
            mmap *= 1e3 # Turn to mJy :)
            cb_label = "[mJy/beam $\\times$ km/s]"
        elif moment == 1:
            cmap = "RdBu_r"
            cb_label = 'Velocity [km/s]'
        elif moment == 2:
            powex = str(moment)
            cmap = "Spectral_r"
            cb_label = r'v$^' + powex + '$ [(km/s)$^' + powex + '$]'
        elif moment == 8:
            cmap = "Spectral_r"
            mmap *= 1e3 # Turn to mJy
            cb_label = "Peak Intensity [mJy/beam]"
        elif moment == 9:
            cmap = "RdBu_r"
            cb_label = "Peak Velocity [km/s]"
        
        if vmin is None: vmin = mmap.min()
        if vmax is None: vmax = mmap.max()

        if vmin > mmap.min():
            extend = "min"
        elif vmax < mmap.max():
            extend="max"
        elif (vmin > mmap.min()) and (vmin < mmap.max()):
            extend="both"
        else:
            extend = "neither"
            # ADDITION: Let's make the min and max symmetric if km/s
            if moment in [1,9]:
                if np.abs(mmap.max()) > 0 and np.abs(mmap.max()) > np.abs(mmap.min()):
                    vmin = -mmap.max(); vmax = mmap.max()
                elif np.abs(mmap.max()) > 0 and np.abs(mmap.max()) < np.abs(mmap.min()):
                    vmin = mmap.min(); vmax = np.abs(mmap.min())

        im = ax.imshow(mmap, extent=(-self.sizeau/2,self.sizeau/2,-self.sizeau/2,self.sizeau/2), cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        cbar = plt.colorbar(im, ax=ax, pad=0, orientation="horizontal", location="top", extend=extend)
        cbar.set_label(cb_label, size=20)
        cbar.ax.tick_params(labelsize=14)

        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

        self._stylize_plot(ax, self.mol_name+" J="+str(self.transition[0])+"-"+str(self.transition[1])+" transition", color="black")

        if save: 
            print("Outputting image plot as .png")
            plt.savefig(self.path+"/saved_plots/MomentMaps/moment-"+str(moment)+"-map-"+self.fname.replace("image-","")+".png", bbox_inches="tight")

    def plot_channel_map(self, mask=True, xlim=None, ylim=None, vmin=None, vmax=None, save=False):
        # Depending on the resolution we define the number of maps made
        if self.nfreq <= 9: n = 3
        elif self.nfreq <= 16: n = 4
        elif self.nfreq <= 25: n = 5
        elif self.nfreq <= 36: n = 6
        elif self.nfreq <= 49: n = 7
        else:
            raise ValueError("linenlam exceeds the recommended number of maps.")
        
        fig, ax = plt.subplots(n,n, figsize=(16,16))
        ax = ax.flatten()

        v_kms = cnst.c.value * (self.nu0 - self.freq) / self.nu0 / 1e3
        max_Tb = 0
        min_Tb = 0

        if mask:
            plot_img = np.ma.masked_less_equal(self.image, 3*self.rms) * 1e3 # mJy/beam
        else:
            plot_img = self.image * 1e3 # mJy/beam

        for i in range(len(ax)):
            if i < self.nfreq:
                # Calculate the brightness temperature of the image
                Tb = 1.222e3 * plot_img[:,:,i] / ((self.freq[i]*1e-9)**2 * self.beam_arcsec[0] * self.beam_arcsec[1]) # https://science.nrao.edu/facilities/vla/proposing/TBconv
                if Tb.max() > max_Tb:
                    max_Tb = Tb.max()
                if Tb.min() < min_Tb:
                    min_Tb = Tb.min()

                plot = ax[i].imshow(Tb, cmap="Spectral_r", origin="lower", vmin=vmin, vmax=vmax, extent=(-self.sizeau/2,self.sizeau/2,-self.sizeau/2,self.sizeau/2))

                if xlim is not None:
                    ax[i].set_xlim(xlim[0], xlim[1])
                if ylim is not None:
                    ax[i].set_ylim(ylim[0], ylim[1])

                plot_size = np.abs(ax[i].get_xlim()[1] - ax[i].get_xlim()[0])
                ax[i].text(-475/500 * plot_size//2, 490/500 * plot_size//2 ,str(np.round(v_kms[i],2)) + " km/s", va="top", ha="left", color="black", size=18)
                self._stylize_plot(ax[i], color="black", text_size=10)
            else:
                fig.delaxes(ax[i])
        if vmin is None: vmin = min_Tb
        if vmax is None: vmax = max_Tb

        if vmin > min_Tb and vmax < max_Tb:
            cb_extend = "both"
        elif vmin > min_Tb:
            cb_extend = "min"
        elif vmax < max_Tb:
            cb_extend = "max"
        else:
            cb_extend = "neither"
        

        plt.subplots_adjust(wspace=0.01, hspace=0.01, top=0.88)
        # compute combined horizontal span of the top row and place colorbar centered above it
        left = min(a.get_position().x0 for a in ax[:n])
        right = max(a.get_position().x1 for a in ax[:n])
        top = max(a.get_position().y1 for a in ax[:n]) + 0.01
        cbar_ax = fig.add_axes([left, top, right - left, 0.015])
        cbar = fig.colorbar(plot, cax=cbar_ax, orientation="horizontal", extend=cb_extend)
        cbar.set_label("Brightness Temperature [K]", size=20)
        cbar.ax.tick_params(labelsize=14)
        cbar_ax.xaxis.set_label_position('top')
        cbar_ax.xaxis.set_ticks_position('top')

        fig.suptitle("Channel Map "+self.mol_name+" J="+str(self.transition[0])+"-"+str(self.transition[1])+" transition", size=30)
        plt.savefig(self.path+"/saved_plots/ChannelMaps/channel-map-"+self.fname.replace("image-","")+".png", bbox_inches="tight", dpi=300)