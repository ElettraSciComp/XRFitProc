# XRFitProc
<div align="justify">  
Source code of web app developed for manuscript: "XRFitProc: a novel web-based XRF fitting system" M. Ippoliti, F. Guzzi, A. Gianoncelli, F. Billè, G. Kourousias (submitted to <a href="https://onlinelibrary.wiley.com/journal/10974539"><em>X-Ray Spectrometry</em></a>))

### Version 09.01.23

XRFitProc, is a novel web-based application developed in [Python](https://www.python.org/), [Numba](https://numba.pydata.org/), and [Dash](https://plotly.com/dash/). The aim of the algorithm is to enable users to fit X-Ray Fluorescence (XRF) spectral data, within a straight-fortward and easy to use GUI. In particular, the presented application has been designed on XRF spectroscopy data acquired on the LEXRF system present at the [TwinMic](https://www.elettra.trieste.it/elettra-beamlines/twinmic.html) beamline in Elettra Sincrotrone Trieste (Trieste, Italy), but can be used on XRF sepctroscopic data acquired on any setup, provided a specific structure is provided in the input data. 

## Installation
After cloning the repository locally, you will need to install and setup an [Anaconda](https://www.anaconda.com/products/distribution) environment with all needed dependencies. This can be achieved by opening a terminal in the "modules" folder and by copying and pasting the following lines of code:

```
conda create -n XRF python=3.8
conda activate XRF
conda install numpy scipy h5py matplotlib flask numba pandas
conda install -c conda-forge xraydb tifffile
conda install -c numba icc_rt
pip install dash==2.7.0 dash-bootstrap-components dash-html-components dash-core-components plotly
```

## Running XRFitProc
### Initialization

![alt text](https://github.com/m-ippoliti/XRFitProc/blob/main/videos/initialization.mp4)

Once the setup is complete, run the application by entering the "modules" folder and executing the following commands in a shell:
```
conda activate XRF
```
and
```
chmod +x run_xrfitproc.sh
./run_xrfitproc.sh --port 8080 --dash_debug False
```
where the port number and debugging flag can be tuned manually. Once the application is running the user will be able to access it in their web-browser of choice, by navigating to the Url http://127.0.0.1:8080/. The command line should look as in the figure below.
<img src=https://github.com/m-ippoliti/XRFitProc/blob/main/images/run_xrfitproc.png alt="run_xrfitproc"/>


### Load Data
#### Example Data
We have provided an example dataset [here](https://dx.doi.org/10.34965/i10645), originally acquired in a study by [Marmorato et. al.](https://doi.org/10.1016/j.toxlet.2011.08.026). In order for the application to work, the "xrf_flat_scan_aligned_2fit.h5" file present at the given link, must be saved to the folder "/XRFitProc/modules/data/input/".


#### Create your own HDF input data
Alternatively, users can also attempt to construct their own [HDF5](https://www.hdfgroup.org/solutions/hdf5/) input file. As for the above case, the HDF5 file must be named "xrf_flat_scan_aligned_2fit.h5" and saved to the folder "/XRFitProc/modules/data/input/". The HDF5 file must also contain a Group named "dante" encapsulating the following data:
 * **beam_en**: float scalar, incident beam's energy in eV 
 * **channel_SUM**: int 2D array, spectroscopic data with shape = (number of pixels, number of channels in single pixel spectrum). This can represent a single detector or the sum of all available detectors 
 * **im_shape**: int 2D array or int tuple, the 2D image shape of the scanned object (x pixels, y pixels) 
 * **offset**: float scalar, used to calibrate channel axis to actual energy axis in eV 
 * **slope**: float scalar, used to calibrate channel axis to actual energy axis in eV

<div align="center"><img src=https://github.com/m-ippoliti/XRFitProc/blob/main/images/data_structure.png alt="data_structure"/></div>

The slope and offset parameters, are necessary in order to calibrate the channels axis belonging to the spectral data, into an energy axis expressed in eV, according to the linear relationship 

$$E(x) = slope*x + offset$$

where x represents the channel axis. In the above example the spectroscopic data was acquired at a beam energy of 1500 eV. The data is made of 14561 pixels, each with 4096 channels, and will produce images of shape 91x161.

**NOTE: it is possible to use the function "create_custom_HDF5_input" present in the "FitTools" module, in order to create a viable HDF5 file by simply providing the above listed data. The HDF5 file will automatically be saved in "/XRFitProc/modules/data/input" as "xrf_flat_scan_aligned_2fit.h5"**


### Fit Setup

<img src=https://github.com/m-ippoliti/XRFitProc/blob/main/images/GUI.png alt="data_structure"/>

In the "Fit Parameters" tab, after the data has been correclty loaded, a single spectra resulting from summing together all the available pixels is displayed in the "Sum Spectra" section. The parameters in the "Fit Setup" section need now to be adjusted:
* **Beam Energy**: is loaded automatically from the HDF5 file, but can be changed to tune to the scattering peak. This value is also used to establish the energetically viable XRF emission lines for all elements
* **Fit Boundaries**: window expressed in channels, in which to perform the fitting
* **Snip Width**: window width of SNIP algorithm, used to estimate a background or continuum 
* **Snip Iter**: number of iterations of SNIP algorithm, used to estimate a background or continuum
 
Lastly, the user has to select which XRF lines he/she would like to investigate, by selecting and adding the element to which they belong to in the dropdown menu present in the "XRF Lines" section. Thanks to the "Beam Energy parameter" all energetically viable XRF lines associated to a given element will be added automatically, by simply selecting the element of interest. The lines are added in bulk, but the user is free to remove any single one independently.

### Batch Fitting
In the “Batch Fitting” tab of XRFitProc, it is possible to run a pixel-wise fitting using the setup adopted in the "Fit Setup" tab. Once the fitting is over, the application displays the elemental presence maps in 2D on the left, which are obtained by summing together all the 2D maps of every XRF line belonging to a specific element. This is done for qualitative identification of areas where the element of interest is present. By hovering over any pixel in any of the presented elemental maps, the raw data and fits of all the XRF components are automatically displayed in a graph on the right side of the page. In this manner the user can immediately inspect and evaluate the results at the single pixel level. Lastly, the 2D maps of each single XRF line can be saved to disk in HDF5 format. The elemental presence maps can also be saved in HDF5 and also in .tiff 

<img src=https://github.com/m-ippoliti/XRFitProc/blob/main/images/BF_res.png alt="BF_res"/>

</div>
