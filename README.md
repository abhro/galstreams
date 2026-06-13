# **galstreams**

![see plot here](notebooks/fig_all_streams_lib.png?raw=true "galstreams 06-2024")

### DESCRIPTION:

This is the *galstreams* Library of Stellar Streams in the Milky Way. Since v1.0 stellar streams are supported as track SkyCoord objects (Track6D), rather than the Footprint objects provided in the previous version (v0.1). The main features of the library since v1.0 are:

-  Celestial, distance, proper motion and radial velocity tracks for each stream (pm/vrad when available) stored as astropy SkyCoord objects
-  Stream's (heliocentric) coordinate frame realised as astropy reference frame
-  Stream's end-points and mid-point
- Polygon Footprints
-  Pole (at mid point) and pole tracks in the heliocentric and Galactocentric (GSR) frames
-  Angular momentum track in a heliocentric reference frame at rest with respect to the Galactic centre
-  Summary object for the full library: Uniformly reported stream length, end points and mid-point, heliocentric and Galactocentric mid-pole, track and discovery references and information flag denoting which of the 6D attributes (sky, distance, proper motions and radial velocity) are available in the track object.

The latest version (1.2) of the library includes >200 stream tracks corresponding to 141 distinct stellar streams. The library (v1.0) is described in detail in [Mateu (2023)](https://ui.adsabs.harvard.edu/link_gateway/2023MNRAS.520.5225M/doi:10.1093/mnras/stad321).

### LATEST CHANGES

- 2026/06 Galstreams added to PyPi, pip install now supported. V1.2 tagged.
- 2024/06 major track update. Over 80 new tracks included (40 new streams, over 40 tracks updated), Vrad tracks included for S5, plus several other individual stream tracks added.
- 2023/05 Widths in phi2, pmphi1_cosphi2/phi2 are now implemented: median widths available in summary table.
- 2023/04 The quick plot method is back! Check out MWStreams.plot_stream_compilation() in the Quick plots section of the examples notebook
- 2023/04 Pandas (v1.3) unit handling error in summary attribute fixed (this fixes errors that came up in the newest pandas versions when accesing mws.summary using .loc)
- 2023/02 get_track_names_in_sky_window method added for convenience. It provides a list of the stream tracks present in a sky window with limits provided by the user in an arbitrary coordinate frame
- 2023/02 TOPCAT-friendly csv files containing all stream tracks, end-points, mid-points and the library summary table can now printed when the library is instantiated. They are stored by default in galstreams/tracks, but can be saved at any user-defined location if using the print_topcat_friendly_compilation method

### REQUIREMENTS

- Python modules required are NUMPY, SCIPY, MATPLOTLIB, ASTROPY and GALA. 
- Installing Basemap is highly recommended (but not needed) for more involved plots. If installing with conda, make sure to update the proj4 library before installing Basemap to avoid errors.

----------

### INSTALLATION

To install the latest development version of galstreams, use `pip` by running the
following command in a terminal:

    pip install git+https://github.com/cmateu/galstreams

Or, download or clone the source code, change directory into the galstreams repository
folder and run:

    pip install .

If importing galstreams fails due to a broken or missing gala installation (for example with a GSL or dlopen error), install gala using conda-forge:

    conda install -c conda-forge gala

----------
# Quick Guide

For a detailed walk through the library please see the example Python notebooks provided [here](notebooks/).
