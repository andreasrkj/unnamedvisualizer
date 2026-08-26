# NOTE: Update SynthObserver release version
# Now uses spectralcube to handle moment mapping for CASA images

from ..helper_functions import goto_folder
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.offsetbox import AuxTransformBox, AnchoredOffsetbox
import astropy.constants as cnst
import astropy.units as unit
from spectral_cube import SpectralCube, Projection
from regions import Regions

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

        # Check whether to load in combined or single config observations
        if len(antennalist) > 1:
            config_str = "combined"+"_".join(antennalist).replace("alma.cycle","").replace(".cfg","")
        else:
            config_str = antennalist[0].replace("cfg","noisy")

        # Load in the fits file
        cube = SpectralCube.read(self.path+"/saved_fits/simalma_"+config_str+"_"+self.fname+".fits")
        # Save the header as a future reference, if needed
        self.header = cube.header

        # Load in the beam from the cube and calculate pixel and arcsec values
        self.beam_px = (np.abs(cube.beam.major.to_value(unit.deg)/cube.header["CDELT1"]), cube.beam.minor.to_value(unit.deg)/cube.header["CDELT1"], cube.beam.pa)
        self.beam_arcsec = (cube.beam.major.to_value(unit.arcsec), cube.beam.minor.to_value(unit.arcsec), cube.beam.pa)

        # Assign header keywords
        self.x = (np.arange(1,self.header["NAXIS1"]+1) - self.header["CRPIX1"]+0.5) * np.abs(self.header["CDELT1"]) * np.pi/180 * dpc * unit.pc.to(unit.cm)
        self.y = (np.arange(1,self.header["NAXIS2"]+1) - self.header["CRPIX2"]+0.5) * np.abs(self.header["CDELT2"]) * np.pi/180 * dpc * unit.pc.to(unit.cm)
        self.nx = len(self.x)
        self.ny = len(self.x)
        self.sizepix_x = np.abs(self.header["CDELT1"] * np.pi/180 * dpc * unit.pc.to(unit.cm))
        self.sizepix_y = np.abs(self.header["CDELT2"] * np.pi/180 * dpc * unit.pc.to(unit.cm))
        self.sizeau = (self.sizepix_x * self.header["NAXIS1"] * unit.cm).to(unit.AU).value
        self.freq = np.linspace(start=self.header["CRVAL3"], stop=(self.header["NAXIS3"]-1)*self.header["CDELT3"]+self.header["CRVAL3"], num=self.header["NAXIS3"])
        self.nfreq = len(self.freq)
        self.wav = (cnst.c / (self.freq * unit.Hz)).to(unit.micron).value
        self.nwav = len(self.wav)
        if self.nfreq % 2 == 0:
            self.nu0 = self.freq[self.nfreq//2-1]/2 + self.freq[self.nfreq//2]/2
        else:
            self.nu0 = self.freq[self.nfreq//2]
        self.hpbw = (1.13 * (cnst.c / (self.nu0 * unit.Hz)) / (12 * unit.m) * unit.rad).to(unit.arcsec).value * self.dpc # in AU
        self.hp0_2w = 1.517 * self.hpbw # also in AU

        # Mask the image within 20% primary beam intensity
        xx, yy = np.meshgrid((self.x * unit.cm).to(unit.AU).value, (self.y * unit.cm).to(unit.AU).value)
        
        # Mask out the cube
        self.image = cube.with_mask(xx**2 + yy**2 < (self.hp0_2w/2)**2).with_spectral_unit(unit.km/unit.s, velocity_convention="radio", rest_value=self.nu0 * unit.Hz)

        # Calculate RMS map assuming the first 20 channels are free of (extended) emission
        self.rms = self.image[0:20,:,:].mad_std(axis=0)

    def _stylize_plot(self, ax, plot_text=None, color="white", text_size=18):
        # Remove axes
        ax.xaxis.label.set_visible(False); ax.yaxis.label.set_visible(False)
        ax.set_yticklabels([]); ax.set_yticks([])
        ax.set_xticklabels([]); ax.set_xticks([])
        
        # Create scale bar
        # We should normalize the distances to the edges
        plot_size = np.abs(ax.get_xlim()[1] - ax.get_xlim()[0])
        bar_length = int(25 * (plot_size // 250))
        bar_length_normalized = bar_length / (np.abs(ax.get_xlim()[1] - ax.get_xlim()[0]))

        ax.hlines(0.07, 0.93-bar_length_normalized/2, 0.93+bar_length_normalized/2, color="black", linestyles="solid", linewidths=3, transform=ax.transAxes)
        ax.text(0.93, 0.05, str(bar_length)+" AU", ha="center", va='top', color="black", transform=ax.transAxes, fontsize=text_size)

        if plot_text is not None: ax.text(0.01, 0.99, plot_text, ha="left", va="top", color=color, fontsize=text_size, transform=ax.transAxes)

        # Add beam
        aux_tr_box = AuxTransformBox(ax.transData)
        aux_tr_box.add_artist(Ellipse((0,0), self.beam_px[0] * self.sizepix_x/unit.AU.to(unit.cm), self.beam_px[1] * self.sizepix_y/unit.AU.to(unit.cm), self.beam_px[2].to_value(unit.deg), color="black"))
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

        if vmin is None: vmin = np.nanmin(plot_img)
        if vmax is None: vmax = np.nanmax(plot_img)

        if vmin > np.nanmin(plot_img):
            extend = "min"
        elif vmax < np.nanmax(plot_img):
            extend="max"
        elif (vmin > np.nanmin(plot_img)) and (vmin < np.nanmax(plot_img)):
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
            plt.savefig(self.path+"/saved_plots/SingleWav/simalma_"+save_name+".png", bbox_inches="tight")

    def calc_moment(self, moment=0, int_lims=(-2,2)):        
        try: mmap = getattr(self, "moment"+str(moment)) # If we have already calculated it, no need to do it again
        except:
            if self.nfreq < 2:
                raise ValueError("Cannot create moment map for a single wavelength image")

            if moment in [0,1,2]:
                try: self.snr_map
                except: self.calc_snr_map()

                spectral_slab = self.image.with_mask(self.snr_map > 1).spectral_slab(int_lims[0] * unit.km/unit.s, int_lims[1] * unit.km/unit.s).to(unit.K)
                mmap = spectral_slab.moment(order=moment)

            elif moment == 8:
                mmap = self.image.max(axis=0)

            elif moment == 9:
                mmap = self.image.argmax_world(axis=0)
            
            else:
                raise ValueError("Cannot create moment maps other than 0, 1, 2, 8, 9")

            setattr(self, "moment"+str(moment), mmap)

        return mmap

    def calc_snr_map(self):
        '''Calculate Signal-to-Noise Ratio in each pixel for contours and masking '''
        try: self.moment8
        except: self.calc_moment(moment=8)

        self.snr_map = self.moment8 / self.rms

        return self.snr_map

    def load_mask(self, mask_path):
        '''Load a region file into the class'''
        region = Regions.read(mask_path, format="crtf")
        # Get mask from region
        streamer_mask = region[0].to_pixel(self.image.wcs).to_mask(mode="center").to_image(self.image.shape).astype(int)

        self.streamer_mask = streamer_mask 

    def plot_moment(self, moment=0, mask=25, vmin=None, vmax=None, ax=None, xlim=None, ylim=None, save=False):
        if ax is None: # Create a figure if not supplied
            fig, ax = plt.subplots(1,1, figsize=(8,10))
        else:
            if save: print("Note, you've supplied a matplotlib axis while setting 'save' = True, this may create a weird-looking plot in the .png")

        try: mmap = getattr(self, "moment"+str(moment))
        except: mmap = self.calc_moment(moment=moment)

        # Mask values
        if mask and moment in [1,2,9]:
            mmap8 = self.calc_moment(moment=8)
            mmap = np.where(mmap8 > mask*self.rms, mmap, np.nan) # Mask out values below SNR threshold

        # Set plot labels and colorbar
        if moment == 0:
            cmap = "Spectral_r"
            cb_label = "[K $\\times$ km/s]"
            mmap = mmap.to_value(unit.K * unit.km/unit.s)
        elif moment == 1:
            cmap = "RdYlBu_r"
            cb_label = 'Velocity [km/s]'
            mmap = mmap.to_value(unit.km/unit.s)
        elif moment == 2:
            cmap = "Spectral_r"
            cb_label = "Velocity Dispersion $\\sigma$ [km/s]"
            mmap = np.sqrt(mmap).to_value(unit.km/unit.s)
        elif moment == 8:
            cmap = "Spectral_r"
            mmap = mmap.to_value(unit.mJy/unit.beam) # Turn to mJy
            cb_label = "Peak Intensity [mJy/beam]"
        elif moment == 9:
            cmap = "RdYlBu_r"
            cb_label = "Peak Velocity [km/s]"
            mmap = mmap.to_value(unit.km/unit.s)
        
        if vmin is None: vmin = np.nanmin(mmap)
        if vmax is None: vmax = np.nanmax(mmap)

        if (vmin > np.nanmin(mmap)) and (vmin < np.nanmax(mmap)):
            extend="both"
        elif vmin > np.nanmin(mmap):
            extend = "min"
        elif vmax < np.nanmax(mmap):
            extend="max"
        else:
            extend = "neither"
        #    # ADDITION: Let's make the min and max symmetric if km/s
        #     if moment in [1,9]:
        #         if np.abs(mmap.max()) > 0 and np.abs(mmap.max()) > np.abs(mmap.min()):
        #             vmin = -mmap.max(); vmax = mmap.max()
        #         elif np.abs(mmap.max()) > 0 and np.abs(mmap.max()) < np.abs(mmap.min()):
        #             vmin = mmap.min(); vmax = np.abs(mmap.min())

        im = ax.imshow(mmap, extent=(-self.sizeau/2,self.sizeau/2,-self.sizeau/2,self.sizeau/2), cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        cbar = plt.colorbar(im, ax=ax, pad=0, orientation="horizontal", location="top", extend=extend)
        cbar.set_label(cb_label, size=20)
        cbar.ax.tick_params(labelsize=14)

        # Add contour lines if velocity maps
        if moment in [1,2,9]:
            mmap8 = self.calc_moment(moment=8)
            snr_map = (mmap8 / self.rms).to_value(unit.dimensionless_unscaled)
            ax.contour(snr_map, levels=[50, 100], extent=(-self.sizeau/2,self.sizeau/2,-self.sizeau/2,self.sizeau/2), origin="lower", colors="black")


        # Create the FWHM primary beam circle
        hpbw_50_circle = plt.Circle((0,0), self.hpbw / 2, color="black", linestyle="--", fill=False)
        ax.add_patch(hpbw_50_circle)

        # Mark the star in the center
        ax.scatter(0,0, marker="*", color="white", s=100, edgecolors="black")
        
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

        #self._stylize_plot(ax, self.mol_name+" J="+str(self.transition[0])+"-"+str(self.transition[1])+" transition", color="black")
        self._stylize_plot(ax, f"{np.round(self.nu0 * 1e-9, 3)} GHz\n{self.mol_name}", color="black")

        if save: 
            print("Outputting image plot as .png")
            plt.savefig(self.path+"/saved_plots/MomentMaps/simalma_moment-"+str(moment)+"-map-"+self.fname.replace("image-","")+".png", bbox_inches="tight")

    def plot_channel_map(self, mask=True, xlim=None, ylim=None, vmin=None, vmax=None, save=False): # TODO - Update to use SpectralCube
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
        plt.savefig(self.path+"/saved_plots/ChannelMaps/simalma_channel-map-"+self.fname.replace("image-","")+".png", bbox_inches="tight", dpi=300)