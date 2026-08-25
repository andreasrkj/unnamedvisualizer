# synthObserver
![Alt text](synthObserver logo)<img src="synthObserver logo.svg">

**synthObserver** converts [RAMSES](https://ramses.cnrs.fr/) simulations into synthetic observations using [RADMC-3D](https://github.com/dullemond/radmc3d-2.0) and [CASA](https://casadocs.readthedocs.io/en/stable/index.html).

### Current features:
- Set up and create necessary files for running RADMC-3D, as well as running the dust temperature calculation
- Simple chemistry: Freeze out molecules and change their abundance at specific temperatures
- Single wavelength (pseudo-dust continuum) images
- Molecular line images (moment maps, channel maps, peak intensity/velocity maps)
- Create and plot SEDs along with the bolometric temperature
- Create and plot bolometric temperature maps of the system by generating an SED for each pixel along the line of sight
- Temperature (RADMC-3D), column density and velocities using [RaDvisPython](https://github.com/CGHolm/RaDvisPython)

### Future features:
- Improve CASA interactability (speed up cleaning, logfile handling)
- Add DISPATCH functionality
- Improve user access to settings

### Known issues:
- Central pixel is VERY bright on all scales, this is possibly due to a problem with the call to RADMC-3D. It causes a very high emission (roughly the size of the beam) in the center of the image, which causes oversubtraction due to the side lobes of the beam when running simALMA and thus mildly to very negative emission. This can cause problems for velocity maps.
- More testing needed, but it is currently recommended to run RADMC-3D and simALMA separately (so first call with casa=False and high number of threads, then with casa=True and low number of threads), as the function seems to slow down with high number of threads. (?)