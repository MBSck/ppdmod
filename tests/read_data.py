from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    path = Path("resolution_and_flux")
    df = pd.read_csv(path / "frequency_resolution.csv")
    print(df)
    df = pd.read_csv(path / "frequency_resolution_wavelength.csv")
    print(df)
