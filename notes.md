# Notes
## Plan
- Pass everything Model to DataHandler (does the fitting etc.)
- DataHandler also does the FFT
- Make example code where the combined models go
## Done
[x] Reworking model.py, methods to implement:
[x] Reworking model_components
## Working-on-ATM
[] Implement parameter length for parameter refactoring with mod params
[] Implement azimuthal_modulation for every model by way of named tuples
[] Implement the Flux method for the combined model class
[] Implement azimuthal modulation via the params (modulation=true or just handing over params directly)
[] Move a lot of the functionality from the Model class to the combined model class, except azimuthal and so
[] Maybe add _set_ones and _set_zeros to the combined model class
[] Implement fitting in DataHandler Class
[] Write FFT tests and check if it works for the fluxes
## To-Do
[] Implement tests for comparisons between analytical and numerical models -> FFT and all
[] Implement tests for (utils.py, model.py, all component classes)
[] Finish the _set_uv_grid method
[] Finish rework of model.py and implement tests
[] Make the model_components (formerly independent models) work with the new model standard and implement test
[] Create composite_model class (maybe its own file) and implement test
[] Make the composite model class output in a way that can be taken by the DataHandler class
[] Rework fourier.py and implement tests
[] Implement better tests for model.py
[] Rework the plotting functionality for the fitted models
[] Make fits adapt to the new scheme and maybe implement tests here as well
[] Make all theta into named tuples to easily access their values
[] Make tests for utils.py
[] Make function that gives stuff like 'eval_model' automatically docstrings
