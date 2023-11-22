from numpy import float32, complex64

OPTIONS = {}

# NOTE: Data.
OPTIONS["data.binning.window"] = None
OPTIONS["data.gravity.index"] = 20
OPTIONS["data.corr_flux"] = []
OPTIONS["data.corr_flux_err"] = []
OPTIONS["data.corr_flux.ucoord"] = []
OPTIONS["data.corr_flux.vcoord"] = []
OPTIONS["data.cphase"] = []
OPTIONS["data.cphase_err"] = []
OPTIONS["data.cphase.u123coord"] = []
OPTIONS["data.cphase.v123coord"] = []
OPTIONS["data.flux"] = []
OPTIONS["data.flux_err"] = []
OPTIONS["data.vis"] = []
OPTIONS["data.vis_err"] = []
OPTIONS["data.vis.ucoord"] = []
OPTIONS["data.vis.vcoord"] = []
OPTIONS["data.readouts"] = []

# NOTE: Model. The output can either be 'surface_brightness' or ' jansky_px'
OPTIONS["model.components_and_params"] = {}
OPTIONS["model.constant_params"] = {}
OPTIONS["model.dtype.complex"] = complex64
OPTIONS["model.dtype.real"] = float32
OPTIONS["model.flux.factor"] = 1
OPTIONS["model.gridtype"] = "linear"
OPTIONS["model.matryoshka"] = False
OPTIONS["model.matryoshka.binning_factors"] = [2, 1, 2]
OPTIONS["model.modulation.order"] = 0
OPTIONS["model.output"] = "jansky_px"
OPTIONS["model.shared_params"] = {}

# NOTE: Fourier transform.
OPTIONS["fourier.backend"] = "numpy"
OPTIONS["fourier.binning"] = None
OPTIONS["fourier.method"] = "complex"
OPTIONS["fourier.padding"] = None

# NOTE: Spectrum.
OPTIONS["spectrum.binning"] = 7
OPTIONS["spectrum.coefficients"] = {
    "low": [0.10600484,  0.01502548,  0.00294806, -0.00021434],
    "high": [-8.02282965e-05,  3.83260266e-03, 7.60090459e-05, -4.30753848e-07]
}
OPTIONS["spectrum.kernel_width"] = 10

# NOTE: Fitting.
OPTIONS["fit.chi2.weight.corr_flux"] = 1
OPTIONS["fit.chi2.weight.cphase"] = 1
OPTIONS["fit.chi2.weight.flux"] = 1
OPTIONS["fit.data"] = ["flux", "vis", "t3phi"]
OPTIONS["fit.wavelengths"] = None
