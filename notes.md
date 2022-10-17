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
## Problems
* Think about what is interpolated? Is Anthony's interpolation correct?? Maybe meshgrid?
* Think about what the interpolation does??
* Reformat priors only for model input not out of models
* There is a non zero element in the FFT... Where?
## Solutions
- Try to use the FFT standalone and test if this works, if not then check the rest of the code again
## Done
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
## Working-on-ATM
[] Make ring component match priors together, so they are not bigger than outer radius and such
[] Make CombinedModel have the right output for fitting
[] Look through all of Fourier transform and check where the phase error comes from... Only in Phase?!
[] Implement parameter length for parameter refactoring with mod params
## To-Do
[] Finish rework of model.py and implement tests
[] Make tests that compare fluxes to real values (e.g., Jozsef's code see flux values)
[] Implement tests for comparisons between analytical and numerical models -> FFT and all
[] Implement and complete the other components (except delta and ring)
[] Finish the _set_uv_grid method
[] Make the composite model class output in a way that can be taken by the DataHandler class
[] Rework the plotting functionality for the fitted models
[] Make fits adapt to the new scheme and maybe implement tests here as well
[] Make function that gives stuff like 'eval_model' automatically docstrings
[] Make the disc params and the general params setting more modular -> Should be possible
