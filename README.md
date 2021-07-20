# Ice-ages-in-AlphaCen

This is a repository containing processed data and python scripts for the article "Milankovitch Cycles for a Circumstellar Earth-analog within Alpha Centauri-like Binaries".  The paper investigates the potential for Milankovitch cycles for a circumstellar Earth-analog in the Alpha Centauri AB system. The planetary obliquity evolution uses methods from [Quarles, Li, & Lissauer (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...886...56Q/abstract), which is incorporated into a 1D energy balance model (EBM) with ice sheets using VPLanet [Barnes et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020PASP..132b4502B/abstract).  Through the EBM,  we evaluate how planetary obliquity (axial tilt) variations due to the stellar companion affect the potential ice distribution states: ice free, ice caps, ice belt, or snowball.

The python scripts use packages from numpy, scipy (>= 1.4), & matplotlib.  See the python-scripts folder for the scripts used to produce figures in the paper and example codes to reproduce our results.  The python scripts assume a directory tree produced from "git clone \*.git" to find/use the "data" and "Figs" subfolders.

Attribution
--------
A more detailed description of these simulations, methods, and the context for the future observations will be available in the following paper.  Please use the following citation, if you find these data/tools useful in your research. 

```
@article{Quarles2021,
 author = {{Quarles}, Billy and {Eggl}, Siegfried and {Rosario-Franco}, Marialis and {Li}, Gongjie},
        title = "{Milankovitch Cycles for a Circumstellar Earth-analog within {\ensuremath{\alpha}} Centauri-like Binaries}",
      journal = {\mnras},
         year = 2021,
        month = jul,
        pages = {"submitted"}
}
```
