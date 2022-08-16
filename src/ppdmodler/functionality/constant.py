#!/usr/bin/env python3

# Mathematical constants and expressions
I=complex(0, 1)

# Natur constants
PLANCK_CONST: float = 6.62607015e-34                # Planck's constant in SI [J/Hz]
SPEED_OF_LIGHT: int = 299792458                     # Speed of light in SI [m/s]
BOLTZMAN_CONST: float = 1.380649e-23                # Boltzman's constant in SI [J/K]
STEFAN_BOLTZMAN_CONST: float = 5.670374419e-8       # in [W/m^2T^2]

# Specifict constants or values
AU_M: float = 149597870700                          # in [m]
SOLAR_LUMINOSITY: float = 3.828e26                  # in [W]
PARSEC2M: float =  3.085678e16                      # in [m]
ARCSEC2RADIANS: float = 4.8481368110954E-06

if __name__ == "__main__":
    print(1/PARSEC2M)
