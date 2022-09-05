import astropy.units as u

from ppdmod.functionality.utils import set_grid

def test_set_grid():
    output = set_grid(50, 10, 128)
    assert isinstance(output, u.mas)

if __name__ == "__main__":
    ...
