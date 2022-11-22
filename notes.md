# Notes
## General
- The parameters from the priors are maed 1/4 away from the priors borders to avoid errors

## Plan
- Implement nested fitting
- Speed up the general code in speed, remove all passes of data
- Use Py-Spy to check what takes the code long to run

## Things to check
* Write a fuckton more tests for the important calculation functions -> Thorough tests needed
* Does temperate gradient get negatively affected by np.inf values?
* Check impact of the FastFourierTransform reforming of the the Cphases? Against wrapping? -> Maybe ask Jacob here?
* Think about what is interpolated? Is Anthony's interpolation correct?? Maybe meshgrid?
* How does the triangle come together (uv coords for cphases), look at what Jozsef does
- Negative or positive closure phases?
* Check what rebinning factor to use (What is sufficient?)
* Check if sublimation temperature is calculated properly
* Is FFT completely correct?
- The Phases?
- The interpolation?
- The closure phases calculation? Calulation operation after the triangle conversion?
- The scaling of the axes? Conversion into meters from frequency scale?
- The zero division could be a problem?

## Problems
* Improve calculation times of the modelling!
* Memory leak in the model component initialisation!
* Code too slow! Make it faster by far? Faster array calculation? Other approaches?

## Ideas
* Switch to a faster array calculation?
* Recode all of this in Rust?

## Solutions
- Try to use the FFT standalone and test if this works, if not then check the rest of the code again

## Working-on-ATM
[] Add fit options
[] Fit visibilities for simple models
[] Include visibilities calculations for data-handler
[] Remove all superfluous imports (tests and such in file -> Make tests outside of file) from the individual scripts
[] Rename branches and move astropy to master branch rename it as well to master

### Slow Calculations
[] Rework of reformat theta to components function
[] Remove passing of the data class
[] Remove as many ifs as possible
- [] Instead of if, multiply result with 0 instead of return or break
[] Run a profiler over the code to check what takes longest
[] Don't pass the data-handler class as often -> Check

### Plotting
[] Add the wavelengths to the dynesty plots
[] Implement more colors for plotting
[] Make better uv coordinate plotting colors for different epochs
[] Fix plotting of the mas (also add offset) for both axes in the model image
- [] Fix FOV plotting (reduce the FOV or check the scaling of the fourier axis?)

### Data output
[] Add fitting method used
[] Add data about the time the model took to run and when it started
[] Add the name of the object that is saved
[] Change the folder names accordingly
[] Save the best fit data (theta, best_total_fluxes, best_correlated_fluxes, etc.) as data-files as well

### Fitting
[] Implement dynesty fitting again

### Model coding
[] Write tests for all that has been done
[] Check what FOV is needed, automatically calculate it for highest wl?
[] Check what pixel scaling is needed as well?

## To-Do
[] Reproduce models from Jozsef (HD163296), use the data from his paper
[] SIMDI Instructions for faster code? GPU coding?
[] Ignore errors at some point, or warnings that is
[] Make tests that compare fluxes to real values (e.g., Jozsef's code see flux values)
[] Implement tests for comparisons between analytical and numerical models -> FFT and all
[] Finish the _set_uv_grid method
[] Remove redundancies to improve code speed (for later)
[] Remove pixel scaling from DataHandler and wavelengths from CombinedModel
[] Implement and complete the other components (except delta and ring)
[] Make function that gives stuff like 'eval_model' automatically docstrings

## Done
[x] Ignore the error outputs for zero division
[x] Fix scaling of correlated flux plotting
[x] Check the plotting for the different epochs
[x] Add the component info to the write_out
[x] Add total fluxes to plotting in same plot
[x] Write all in a ini file
[x] Implement rebinning from high resolution to low -> Check that rebinning works as planned
[x] Rework the plotting functionality for the fitted models
[x] Reworking model.py, methods to implement:
[x] Reworking model_components
[x] Implemented utils and IterNamespace class
[x] Think about moving the functions from the class that calculate the flux or to a higher class (e.g., combined model)
[x] Implement azimuthal_modulation for every model by way of named tuples
[x] Implement azimuthal modulation via the params (modulation=true or just handing over params directly)
[x] Move a lot of the functionality from the Model class to the combined model class, except azimuthal and so
[x] Implement tests for (utils.py, model.py, all component classes)
[x] Write test for all the new functionality
- model.py with the fixed_params from utils
[x] Implement the Flux method for the combined model class
[x] Rewrite the model.py for more modular approach, not all functions in this class
[x] Create composite_model class (maybe its own file) and implement test
[x] Make the model_components (formerly independent models) work with the new model standard and implement test
[x] Make tests for utils.py
[x] Make all theta into named tuples to easily access their values
[x] Rework fourier.py and implement tests
[x] Make DataHandler get the priors from the components
[x] Check readout of uvcoords from fits file
[x] Make ring component match priors together, so they are not bigger than outer radius and such
[x] Make FastFourierTransform have the right output for fitting
[x] Make the composite model class output in a way that can be taken by the DataHandler class
[x] Make fits adapt to the new scheme
[x] Make the disc params and the general params setting more modular -> Should be possible

## Things-already-checked
* Field of view checked
- Should be ok as it is radius, just in the plotting at the end there is a plotting mistake
* When to modulate the parameters?
- Should not matter
* Check if the FFT zero-padding moves the true centre
- Should be ok tho?
