# Ice-ages-in-AlphaCen

This is a repository containing processed data and python scripts for the article "Milankovitch Cycles for a Circumstellar Earth-analog within Alpha Centauri-like Binaries".  The paper investigates the potential for Milankovitch cycles for a circumstellar Earth-analog in the Alpha Centauri AB system. The planetary obliquity evolution uses methods from [Quarles, Li, & Lissauer (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...886...56Q/abstract), which is incorporated into a 1D energy balance model (EBM) with ice sheets using [VPLanet](https://github.com/VirtualPlanetaryLaboratory/vplanet) from [Barnes et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020PASP..132b4502B/abstract). Through the EBM,  we evaluate how planetary obliquity (axial tilt) variations due to the stellar companion affect the potential ice distribution states: ice free, ice caps, ice belt, or snowball.

In the python-scripts folder, there are a number of python files that reproduce the figures in our work.  The python scripts assume a directory tree produced from "git clone \*.git" to find/use the "data" subfolder and produce the associated figure in the "Figs" subfolder.  See the python-scripts folder for more details, but here are summaries of each script and its usage.  

-  plot_FigXX.py
   -  These scripts do not require any line arguments and are called via "python plot_FigXX.py" from an anaconda terminal, where the XX is replace with 1, 4, 5, 6, 10, 11, 12, or 16.
-  plot_Cycles_aCen.py
   -  This script requires some line arguments from the user to produce Figs. 2, 3, 7, 8, or 9.  The line arguments are the initial obliquity (23 deg.), spin precession (10, 46, or 85 ''/yr), mutual inclination (2, 10, 30 deg.), and the host star (A or B).  For example, 'python plot_Cycles_aCen_py 23 10 85 A' reproduces Figure 2.
-  plot_Cycles_GenBin.py 
   -   This script is similar to the plot_Cycles_aCen.py script, but applies to our study of more general binaries in Figs. 13, 14, and 15.  It also uses line arguments, which are the initial binary semimajor axis (20, 25, or 30) and eccentricity (0.2).  For example, 'python plot_Cycles_GenBin.py 20 0.2' reproduces Figure 13.
-  aCen_climate.py
   -   This script demonstrates how we generate our initial conditions for our simulations.

The python scripts use packages from numpy, scipy (>= 1.4), & matplotlib. 

Attribution
--------
A more detailed description of these simulations, methods, and the context for the future observations are available in the following paper.  Please use the following citation, if you find these data/tools useful in your research. 

```
@ARTICLE{Quarles2021,
       author = {{Quarles}, Billy and {Li}, Gongjie and {Lissauer}, Jack J.},
        title = "{Milankovitch Cycles for a Circumstellar Earth-analog within $\alpha$ Centauri-like Binaries}",
      journal = {\mnras},
     keywords = {binaries: general, stars: individual: 𝛼 Centauri,planets and satellites: atmospheres, planets and satellites: dynamical evolution and stability},
         year = 2021,
        month = Oct,
        pages = {in Press},
archivePrefix = {arXiv},
       eprint = {2108.12650},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210812650Q},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
