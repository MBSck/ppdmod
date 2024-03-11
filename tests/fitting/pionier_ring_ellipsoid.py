from pathlib import Path

import astropy.units as u

from ppdmod import data
from ppdmod import fitting
from ppdmod import plot
from ppdmod.basic_components import assemble_components
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS


DATA_DIR = Path("../data/pionier/nChannels3")
OPTIONS.fit.data = ["vis2"]
OPTIONS.model.output = "non-physical"
