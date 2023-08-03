OPTIONS = {}

OPTIONS["data.readouts"] = []
OPTIONS["data.correlated_flux"] = []
OPTIONS["data.correlated_flux_error"] = []
OPTIONS["data.closure_phase"] = []
OPTIONS["data.closure_phase_error"] = []
OPTIONS["data.opacity_files_and_weights"] = {}
OPTIONS["data.constant_parameters"] = {}
OPTIONS["data.model_and_parameters"] = {}

# NOTE: Model. The output can either be 'surface_brightness' or ' jansky_px'
OPTIONS["model.output"] = "jansky_px"

# NOTE: Fourier transform.
OPTIONS["fourier.padding"] = None
OPTIONS["fourier.binning"] = None
OPTIONS["fourier.backend"] = "numpy"

# NOTE: Spectrum.
OPTIONS["spectrum.coefficients"] = {
    "low": [0.10600484,  0.01502548,  0.00294806, -0.00021434],
    "high": [-8.02282965e-05,  3.83260266e-03, 7.60090459e-05, -4.30753848e-07]
}
OPTIONS["spectrum.binning"] = {"low": 7,
                               "high": 7}

# NOTE: Fitting.
OPTIONS["fit.datasets"] = ["flux", "vis", "t3phi"]
OPTIONS["fit.wavelengths"] = None
