from pathlib import Path
from typing import Optional
import logging
import geopandas as gpd
from hydromt_delft3dfm import DFlowFMModel
from pyproj import CRS
import numpy as np
from delft3dfm_utils import Delft3dFMReader, Delft3DFMConverter


# Define the root path where the project files are located
root = Path(r"c:\Projects\2023\SITO urban\delft3dfm\dimr_export")

# Define the filename of the Rainfall Runoff Model
mdu_fn = "dflowfm\FlowFM.mdu"
mdu = root / mdu_fn

# define region
region = gpd.read_file(
    r"c:\Projects\2023\SITO urban\delft3dfm\dimr_export\geoms\clip_region.shp"
)
# extract geometries from model
model = Delft3dFMReader(mdu, crs=28992)
geometries = model.extract_geometries(region=region)

# Convert
convertor = Delft3DFMConverter()
geometries_converted = convertor.convert(geometries=geometries)
