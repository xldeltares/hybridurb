import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
from pyproj import CRS

from delft3dfm_utils import Delft3DFM

# Define the root path where the project files are located
root = Path(r"c:\Developments\hybridurb\data\Eindhoven\Delft3DFM")
out_dir = root.joinpath("../Model Building")

# Define the filename of the Rainfall Runoff Model
mdu_fn = "dflowfm\FlowFM.mdu"
mdu = root / mdu_fn

# define region
region = gpd.read_file(
    r"c:\Projects\2023\SITO urban\delft3dfm\dimr_export\geoms\clip_region.shp"
)


# extract geometries from model
model = Delft3DFM()
model.read_model(mdu, crs=28992)
model.extract_geometries(region=region)
model.convert_geometries()
model.save_geometries(out_dir)
