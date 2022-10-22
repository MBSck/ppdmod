# Notes
## General
- The parameters from the priors are maed 1/4 away from the priors borders to avoid errors
## Plan
- CombinedModel also does the FFT
- Rework DataHandler to take all the data required for the model
- Make CombinedModel completely modular with DataHandler and fitting
- Pass all parameters to DataHandler
- Make CombinedModel do also polychromatic modelling
## Things to check
* Check how the field of view works? Double 30, extends in both directions, half it?
* Write a fuckton more tests for the important calculation functions -> Thorough tests needed
* Does temperate gradient get negatively affected by np.inf values?
* When to modulate the parameters?
* Check if the FFT zero-padding moves the true centre -> Should be ok tho?
* Maybe combine the flux and the correlated fluxed for plotting?
* Check impact of the FastFourierTransform reforming of the the Cphases? Against wrapping? -> Maybe ask Jacob here?
* Think about what is interpolated? Is Anthony's interpolation correct?? Maybe meshgrid?
* Think about what the interpolation does??
* How does the triangle come together (uv coords for cphases), look at what Jozsef does
* Check the calulation operation after the triangle conversion
* Check conversion into meters from frequency scale
* Check what rebinning factor to use (What is sufficient?)
* Check if sublimation temperature is calculated properly
## Problems
* Improve calculation times of the modelling!
* Memory leak in the model component initialisation!
* Code too slow! Make it faster by far? Faster array calculation? Other approaches?
## Ideas
* Switch to pyFFTW at some time maybe?
* Switch to a faster array calculation?
* Recode all of this in Rust?
## Solutions
- Try to use the FFT standalone and test if this works, if not then check the rest of the code again
## Working-on-ATM

### Plotting
[] Make better uv coordinate plotting colors for different epochs
[] Implement more colors for plotting
[] Fix FOV plotting (reduce the FOV or check the scaling of the fourier axis?)

### Data output
[] Add data about the time the model took to run and when it started
[] Save the best fit data (theta, best_total_fluxes, best_correlated_fluxes, etc.) as data-files as well

### Model coding
[] Look through all of Fourier transform and check where the phase error comes from... Only in Phase?!
[] Write tests for all that has been done
[] Check what FOV is neede, automatically calculate it for highest wl?

## To-Do
[] Look up parallelisation
[] Drop ifs for parallelisation
- Instead of if, multiply result with 0 instead of return or break
[] SIMDI Instructions for faster code? GPU coding?
[] Ignore errors at some point, or warnings that is
[] Finish rework of model.py and implement tests
[] Make tests that compare fluxes to real values (e.g., Jozsef's code see flux values)
[] Implement tests for comparisons between analytical and numerical models -> FFT and all
[] Implement and complete the other components (except delta and ring)
[] Finish the _set_uv_grid method
[] Make function that gives stuff like 'eval_model' automatically docstrings
[] Remove redundancies to improve code speed (for later)
[] Remove pixel scaling from DataHandler and wavelengths from CombinedModel

## Done
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
