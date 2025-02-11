import bokeh.plotting as bpl
import cv2
import datetime
import glob
import holoviews as hv
from IPython import get_ipython
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import psutil
from pathlib import Path

try:
    cv2.setNumThreads(0)
except():
    pass

try:
    if __IPYTHON__:
        get_ipython().run_line_magic('load_ext', 'autoreload')
        get_ipython().run_line_magic('autoreload', '2')
except NameError:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf, params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
from caiman.utils.visualization import view_quilt

bpl.output_notebook()
hv.notebook_extension('bokeh')

# Continuing with our basic setup, we will set up a logger, and also set some environment variables in case that wasn't done already in your shell. If you want to learn more about Caiman's logger functionality, or tweak your logger (e.g., to log to file instead of your console), see [the relevant documentation](https://caiman.readthedocs.io/en/latest/Getting_Started.html#logging). 

# set up logging
logging.basicConfig(format="{asctime} - {levelname} - [{filename} {funcName}() {lineno}] - pid {process} - {message}",
                    filename=None, 
                    level=logging.WARNING, style="{") #logging level can be DEBUG, INFO, WARNING, ERROR, CRITICAL

# set env variables 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


movie_path = download_demo('Sue_2x_3000_40_-46.tif')
print(f"Original movie for demo is in {movie_path}")

# press q to close
movie_orig = cm.load(movie_path) 
downsampling_ratio = 0.2  # subsample 5x
movie_orig.resize(fz=downsampling_ratio).play(gain=1.3,
                                              q_max=99.5, 
                                              fr=30,
                                              plot_text=True,
                                              magnification=2,
                                              do_loop=False,
                                              backend='opencv')


# ## Set initial parameters
# In general in Caiman, estimators are first initialized with a set of parameters, and then they are fit against actual data in a separate step. In this section, we'll define a `parameters` object that will subsequently be used to initialize our different estimators. 
# 
# The parameters are divided into different categories. We will not discuss them in detail in this section, but will go over them when needed (and note that in this notebook we will mostly focus on CNMF and later steps):


# general dataset-dependent parameters
fr = 30                     # imaging rate in frames per second
decay_time = 0.4            # length of a typical transient in seconds
dxy = (2., 2.)              # spatial resolution in x and y in (um per pixel)

# motion correction parameters
strides = (48, 48)          # start a new patch for pw-rigid motion correction every x pixels
overlaps = (24, 24)         # overlap between patches (width of patch = strides+overlaps)
max_shifts = (6,6)          # maximum allowed rigid shifts (in pixels)
max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
pw_rigid = True             # flag for performing non-rigid motion correction

# CNMF parameters for source extraction and deconvolution
p = 1                       # order of the autoregressive system (set p=2 if there is visible rise time in data)
gnb = 2                     # number of global background components (set to 1 or 2)
merge_thr = 0.85            # merging threshold, max correlation allowed
bas_nonneg = True           # enforce nonnegativity constraint on calcium traces (technically on baseline)
rf = 15                     # half-size of the patches in pixels (patch width is rf*2 + 1)
stride_cnmf = 10             # amount of overlap between the patches in pixels (overlap is stride_cnmf+1) 
K = 4                       # number of components per patch
gSig = np.array([4, 4])     # expected half-width of neurons in pixels (Gaussian kernel standard deviation)
gSiz = 2*gSig + 1           # Gaussian kernel width and hight
method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data see demo_dendritic.ipynb)
ssub = 1                    # spatial subsampling during initialization 
tsub = 1                    # temporal subsampling during intialization

# parameters for component evaluation
min_SNR = 2.0               # signal to noise ratio for accepting a component
rval_thr = 0.85             # space correlation threshold for accepting a component
cnn_thr = 0.99              # threshold for CNN based classifier
cnn_lowest = 0.1            # neurons with cnn probability lower than this value are rejected


# We place the above parameter values in a dictionary that is passed to the `CNMFParams` class (those parameters *not* explicitly defined will assume default values):

parameter_dict = {'fnames': movie_path,
                  'fr': fr,
                  'dxy': dxy,
                  'decay_time': decay_time,
                  'strides': strides,
                  'overlaps': overlaps,
                  'max_shifts': max_shifts,
                  'max_deviation_rigid': max_deviation_rigid,
                  'pw_rigid': pw_rigid,
                  'p': p,
                  'nb': gnb,
                  'rf': rf,
                  'K': K, 
                  'gSig': gSig,
                  'gSiz': gSiz,
                  'stride': stride_cnmf,
                  'method_init': method_init,
                  'rolling_sum': True,
                  'only_init': True,
                  'ssub': ssub,
                  'tsub': tsub,
                  'merge_thr': merge_thr, 
                  'bas_nonneg': bas_nonneg,
                  'min_SNR': min_SNR,
                  'rval_thr': rval_thr,
                  'use_cnn': True,
                  'min_cnn_thr': cnn_thr,
                  'cnn_lowest': cnn_lowest}

parameters = params.CNMFParams(params_dict=parameter_dict) # CNMFParams is the parameters class


print(f"You have {psutil.cpu_count()} CPUs available in your current environment")
num_processors_to_use = None

# Initialize a cluster of processors. If one has already been set up (the `cluster` variable is already in your namespace), then that cluster will be closed and a new one created.

if 'cluster' in locals():  # 'locals' contains list of current local variables
    print('Closing previous cluster')
    cm.stop_server(dview=cluster)
print("Setting up new cluster")
_, cluster, n_processes = cm.cluster.setup_cluster(backend='multiprocessing', 
                                                   n_processes=num_processors_to_use, 
                                                   ignore_preexisting=False)
print(f"Successfully initilialized multicore processing with a pool of {n_processes} CPU cores")

# Restart the cluster to clean up memory in preparation for CNMF run.

cm.stop_server(dview=cluster)
_, cluster, n_processes = cm.cluster.setup_cluster(backend='multiprocessing', 
                                                   n_processes=num_processors_to_use, 
                                                   single_thread=False)


# # Run CNMF on patches in parallel

cnmf_model = cnmf.CNMF(n_processes, 
                       params=parameters, 
                       dview=cluster)

cnmf_fit = cnmf_model.fit(images)

cnmf_fit.estimates.plot_contours_nb(img=correlation_image);

cnmf_refit = cnmf_fit.refit(images, dview=cluster)

cnmf_refit.estimates.plot_contours_nb(img=correlation_image);

# see shape of A and C
cnmf_refit.estimates.A.shape, cnmf_refit.estimates.C.shape

print("Thresholds to be used for evaluate_components()")
print(f"min_SNR = {cnmf_refit.params.quality['min_SNR']}")
print(f"rval_thr = {cnmf_refit.params.quality['rval_thr']}")
print(f"min_cnn_thr = {cnmf_refit.params.quality['min_cnn_thr']}")


# Run `estimates.evaluate_components()`. This can take a few minutes if you have a very large number of components:
cnmf_refit.estimates.evaluate_components(images, cnmf_refit.params, dview=cluster);

# This method filled in two arrays in the `estimates` class: `idx_components` (accepted) and `idx_components_bad` (rejected). 
print(f"Num accepted/rejected: {len(cnmf_refit.estimates.idx_components)}, {len(cnmf_refit.estimates.idx_components_bad)}")


# rejected components
if len(cnmf_refit.estimates.idx_components_bad) > 0:
    cnmf_refit.estimates.nb_view_components(img=correlation_image, 
                                            idx=cnmf_refit.estimates.idx_components_bad, 
                                            cmap='gray',
                                            denoised_color='red')
else:
    print("No components were rejected.")


idx_accepted = cnmf_refit.estimates.idx_components
all_contour_coords = [cnmf_refit.estimates.coordinates[idx]['coordinates'] for idx in idx_accepted]


# Each footprint in `A` is stored as a compressed sparse column array. We can convert it to a dense array with `toarray()`, and plot the contour and footprint together:


idx_to_plot = 30
component_number = idx_accepted[idx_to_plot]
component_contour = all_contour_coords[idx_to_plot]
component_footprint = np.reshape(cnmf_refit.estimates.A[:, component_number].toarray(), dims, order='F')


plt.figure(); 
plt.imshow(component_footprint, cmap='gray');
plt.plot(component_contour[:, 0], 
         component_contour[:, 1], 
         color='pink', 
         linewidth=2)
plt.title(f'Footprint/Contour {component_number}');


#  ## How to save and load results (optional)
# There is a built-in `save()` method for the `cnmf` object. It can save as `hdf5` or `nwb`.
# 
# > Note: when you save, you are only saving what is contained in the `cnmf` object. If you want to save other things in your workspace at the same time, you can attach them to your `estimates` object. We'll show how to do this with the correlation image which is useful for plotting.


save_results = True
if save_results:
    save_path =  r'demo_pipeline_results.hdf5'  # or add full/path/to/file.hdf5
    cnmf_refit.estimates.Cn = correlation_image # squirrel away correlation image with cnmf object
    cnmf_refit.save(save_path)


# ### Loading saved results
# You can use the `load_CNMF()` method to load your saved results. 


load_results = True
if load_results:
    save_path =  r'demo_pipeline_results.hdf5'  # or add full/path/to/file.hdf5
    cnmf_refit = cnmf.load_CNMF(save_path, 
                                n_processes=num_processors_to_use, 
                                dview=cluster)
    correlation_image = cnmf_refit.estimates.Cn
    print(f"Successfully loaded data.")


# # A few final things
# We have extracted the calcium traces `C`, spatial footprints `A`, and estimated spike counts `S`, which is the main goal with CNMF. But there are a few important things remaining.


# ## Extract $\Delta F/F$ values
# So far our calcium traces are in arbitrary units. It is more common to report the calcium fluorescence relative to some baseline value $F_0$:
# 
# $$
# \Delta F/F = (F(t)-F_0)/F_0
# $$
# 
# Traditionally, the baseline value was calculated during some initial period of queiscience (e.g., in the visual system, when the animal was sitting in the dark). However, it is problematic to make any assumptions about a "quiet" baseline period, so researchers have started to use a moving average to calculate $F_0$. This is also what Caiman does.
# 
# More specifically, in Caiman the baseline is a running percentile calculated over a `frames_window` moving window. You can calculate $\Delta F/F$ using raw traces or the denoised traces in C (this is toggled using the `use_residuals` argument).


if cnmf_refit.estimates.F_dff is None:
    print('Calculating estimates.F_dff')
    cnmf_refit.estimates.detrend_df_f(quantileMin=8, 
                                      frames_window=250,
                                      flag_auto=False,
                                      use_residuals=False);  
else:
    print("estimates.F_dff already defined")


# The `estimates` object will now have a `F_dff` field, which makes it easier to compare traces across neurons/sessions because the data is normalized. Note that F_dff values can be positive or negative because it is relative to a baseline value that is the "zero" value.
# 
# We can plot a temporal trace of a dff with time. In the following, we'll generate a `frame_time` variable for plotting. This is an  idealized time representation using `linspace()`, which assumes that images were acquired at a constant frequency throughout the session. In practice your hardware probably returns the time at which each frame was acquired, so you can use that information in your own analysis if you have it. 


frame_rate = cnmf_refit.params.data['fr']
frame_pd = 1/frame_rate
frame_times = np.linspace(0, num_frames*frame_pd, num_frames);


# plot F_dff
idx_to_plot = 30
idx_accepted = cnmf_refit.estimates.idx_components
component_number = idx_accepted[idx_to_plot]
f, ax = plt.subplots(figsize=(7,2))
ax.plot(frame_times, 
        cnmf_refit.estimates.F_dff[component_number, :], 
        linewidth=0.5,
        color='k');
ax.set_xlabel('Time (s)')
ax.set_ylabel('$\Delta F/F$')
ax.set_title(f"$\Delta F/F$ for unit {component_number}");
plt.tight_layout()


# <div class="alert alert-info" markdown="1">
#     <h2 style="margin-top: 0;">More on $\Delta F/F$</h2>  
# The above discussion of dff leaves out some details about it is actually calculated in Caiman. In practice, Caiman locally projects the model of the background activity to the spatial footprint of the neuron, and adds a moving percentile of this projected trace to the denominator as an additional normalizing term (see the discussion on the background model below). This acts as an empirical fudge factor that can keep $\Delta F/F$ from getting too large for very small values of $F_0$ (especially when using denoised traces). Users may prefer to use the more traditional equation directly. Thanks to Peter Rupprecht for a helpful discussion of this topic. 
# </div>


# ## Select only accepted components (optional)
# If you want to discard rejected components (`estimates.idx_components_bad`) from the `estimates` field, you can run  `select_components()`. This can be useful if you are sure you only want to focus on the accepted components for downstream analysis (e.g., to share final results with colleagues, for instance).
# 
# <div class="alert alert-warning" markdown="1">
#     <h4 style="margin-top: 0;">Warning: select_components() is a destructive operation</h4>  
# If you run this command, the rejected components will be removed from your <em>estimates</em> field. If you think you might want them later, you can set the <em>save_discarded_components</em> parameter to <em>True</em>. This will let you retrieve them later with the <em>restore_discarded_components()</em> method. 
# </div>


cnmf_refit.estimates.select_components(use_object=True);


# The `use_object` parameter specifies that we want to select the accepted components in `estimates.idx_components` (and remove the components in `idx_components_bad`). 


# ## Display final results
# View the final refined set of remaining components.


cnmf_refit.estimates.nb_view_components(img=correlation_image, 
                                        denoised_color='red',
                                        cmap='gray');


# ## Save results to csv
# When we showed how to save/load the cnmf model above, that was for working with the entire estimator object (for instance if you want to save your work and start again later). Often users just want to save certain results such as the calcium traces in `C` to csv and do downstream analysis later. In Python it is easiest to do this using the Pandas package.
# 
# The following shows how to save your calcium traces to a file called `C_traces.csv`, to the `caiman_data` folder. You can adapt this code to save other data fields such as dff. We will also save the frame times as a convenient reference.


data_to_save = np.vstack((frame_times, cnmf_refit.estimates.C)).T  # Transpose so time series are in columns
save_df = pd.DataFrame(data_to_save)
save_df.rename(columns={0:'time'}, inplace=True)
# check out the dataframe
save_df.head()


# Set up file path and save using the Pandas `to_csv()` method:


c_save_path = cm.paths.get_tempdir() + '\\C_traces.csv'
save_df.to_csv(c_save_path, index=False)
print(f"Saved estimates.C to {c_save_path}")


# ## View different result movies
# ### An aside on the underlying model
# To understand the next two visualizations, we need to discuss the model used by Caiman a little bit. The CNMF algorithm models the original movie as a sum of *neural activity* and *background activity*: the *noise* (or *residual*) is everything left out:
# 
#     original_movie = neural_activity + background + residual
#     
# In this model, `neural_activity` is the product of the matrices `AC`, the spatial and temporal components we have been exploring (`estimates.A` and `estimates.C`). It is our model of the neural bits that we care about.
# 
# `background` is the model's representation of all the background activity in our movie that we wish wasn't there -- this includes stray fluorescence from the neuropil and out-of-plane neural components. This background model is also broken up into spatial and temporal components, which are in `estimates.b` (background) and `estimates.f` (fluctuations), respectively. 
# 
# The "noise", or residual term, is by definition, everything else not captured by the model: 
# 
#     residual = original_movie - neural_activity - background


# ### View denoised movie
# We can rearrange the equations above to yield a "denoised" movie, which is just the original movie with the residual removed:
# 
#     denoised_movie = original_movie - residual = neural_activity + background
#     
# Plugging in appropriate terms from the models of neural activity and background activity (`AC` and `bf`) yields:


# reconstruct denoised movie
neural_activity = cnmf_refit.estimates.A @ cnmf_refit.estimates.C  # AC
background = cnmf_refit.estimates.b @ cnmf_refit.estimates.f  # bf
denoised_movie = neural_activity + background  # AC + bf

# turn into a movie object
denoised_movie = cm.movie(denoised_movie).reshape(dims + (-1,), order='F').transpose([2, 0, 1])


# View the denoised movie (with background included):


# press q to quit
downsampling_ratio = 0.2
denoised_movie.resize(fz=downsampling_ratio).play(gain=0.8,
                                                  q_min=30,
                                                  q_max=99, 
                                                  fr=30,
                                                  plot_text=True,
                                                  magnification=3,
                                                  backend='opencv')


# ### Visualize data, predicted activity, and residual
# For our final visualization, we will use a built-in method `play_movie()` that shows the original movie, the predicted movie (either `AC` or `AC + bf`), and the residual. 
# 
# Viewing the residuals (what the model doesn't explain) can be extremely useful: if you end up seeing lots of neural activity in the residual movie, that means your model is leaving something important out, and you might need to go tweak some parameters. In fact, in Caiman's online algorithms, this is how we decide whether to add new neurons after initialization! If neural activity is discovered in the residual buffer, then it's time to add a new neuron to `AC`! In the offline algorithms, it just means you might need to go check your parameters. No model is perfect: there is always some residual. You have to use your judgment about whether it is worth chasing with additional fitting.
# 
# > The `play_movie()` method has an option to include the background from the model (`bf`) or not (this is the `include_bck` Boolean parameter). If you set it to `False`, it will subtract `bf` from the original movie and will only include `AC` in the middle panel. 


# in case you are working from loaded data, recover the raw movie
Yr, dims, num_frames = cm.load_memmap(cnmf_refit.mmap_file)
images = np.reshape(Yr.T, [num_frames] + list(dims), order='F')


cnmf_refit.estimates.play_movie?


# press q to quit (can take a while to start running)
cnmf_refit.estimates.play_movie(images, 
                                q_max=99.9, 
                                gain_res=0.5,
                                magnification=2,
                                include_bck=True,
                                use_color=True,
                                frame_range=slice(None, None, 5), # sample every 5th frame for speedup
                                thr=0); # set thr to 0.1 to see contours


# Note that if you ran `select_components()` to remove rejected components, the residual movie will contain activity from those rejected bits. This doesn't mean the model is bad: the movie is simply showing false positives you already removed.


# # What next?
# The goal of Caiman is to help you extract high-quality signals from your data, and we have achieved that goal with the above steps. The next step, of doing actual *analysis* of these results, is where the most interesting neuroscience happens: what are the statistical features of these signals? How do they relate to other environmental, molecular, behavioral, and neuronal features that you are studying? What kinds of visualization and analysis tools can you build around this infrastructure? If you find/publish things that help, please share them with us, as we would love to hear about them! 


# # Clean up open resources
# We have a few resources we have left open we should take care of.
# 
# ## Shut down cluster 
# To free up processing resources, let's shut down the cluster in case it is still open. 


cm.stop_server(dview=cluster)


# ## Shut down logger, optionally remove log files
# If you set up your logger to log to files, and you don't want to preserve them, you can delete them with the following. If you have custom log filenames, you may have to change the `log_files` pattern for the following to work. 


# Shut down logger (otherwise will not be able to delete it)
logging.shutdown()


delete_logs = True
logging_dir = cm.paths.get_tempdir() 
if delete_logs:
    log_files = glob.glob(logging_dir + '\\demo_pipeline' + '*' + '.log')
    for log_file in log_files:
        print(f"Deleting {log_file}")
        os.remove(log_file)
else:
    print(f"If you want to inspect your logs they are in {logging_dir}")


