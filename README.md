# Unnamed Visualizer
*(Yes, I haven't figured out a name yet)*

**Unnamed Visualizer** converts [RAMSES](https://ramses.cnrs.fr/) simulations into synthetic observations using [RADMC-3D](https://github.com/dullemond/radmc3d-2.0) and [CASA](https://casadocs.readthedocs.io/en/stable/index.html).

### Current features:
- Set up and create necessary files for running RADMC-3D, as well as running the dust temperature calculation
- Single wavelength (pseudo-dust continuum) images
- Molecular line images (moment maps, channel maps, peak intensity/velocity maps)
- Create and plot SEDs along with the bolometric temperature
- Create and plot bolometric temperature maps of the system by generating an SED for each pixel along the line of sight
- Temperature (RADMC-3D), column density and velocities using [RaDvisPython](https://github.com/CGHolm/RaDvisPython)

### Future features:
- Improve CASA interactability
- Enable CASA parallelization