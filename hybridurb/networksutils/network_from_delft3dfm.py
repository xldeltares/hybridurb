mystring = """

"""
"""
get_geoms_from_model:
    input:  region
    output: # geoms/*.geojson
            
setup_network:
    _config: yaml secifying which files are read as nodes and which as edges
    input: geoms/*.geojson
    output: model/graph.gpickle
    
simplify_network: get flow path
    _config: yaml secifying which files are read as nodes and which as edges
    input: geoms/*.geojson
    output: model/graph.gpickle
  
"""

import logging
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
from pyproj import CRS

from delft3dfm_utils import Delft3DFM


import yaml
from hydromt.config import configread
from hydromt.data_catalog import DataCatalog

logger = logging.getLogger()


class Delft3dfmNetworkWrapper:
    _config_fn = Path(__file__).with_name("config.yml")
    _data_catalog = Path(__file__).with_name("data_catalog.yml")

    def __init__(self, config_fn: Path = _config_fn) -> None:
        self.config = configread(config_fn=config_fn)
        self._init_global()
        # data_catalog = DataCatalog(_data_catalog)

    def _init_global(self):
        _global = self.config.get("global", {})
        self.root = Path(_global.get("root", Path.cwd))
        # FIXME parse region
        region = _global.get("region")
        self.region = None if not region else gpd.read_file(region)
        crs = _global.get("crs")
        self.crs = None if not crs else CRS.from_user_input(crs)

    def get_geoms_from_model(self):
        # get from self properties
        _config = self.config.get("get_geoms_from_model")
        if not _config:
            return None

        model_dir = self.root.joinpath(_config.get("model_dir"))
        geoms_dir = self.root.joinpath(_config.get("geoms_dir"))

        if not any([geoms_dir, model_dir]):
            logger.error("could not perform get_geoms_from_model.")
            return None

        elif (
            (geoms_dir is not None) and geoms_dir.is_dir() and any(geoms_dir.iterdir())
        ):
            logger.info(f"reading geoms from {geoms_dir}")
            geoms = self._read_geoms(geoms_dir, self.crs)
        else:
            logging.info(f"reading model from {model_dir}")
            self.get_geoms(model_dir, self.crs, self.region)

        return geoms

    def get_geoms(
        self, model_dir: Path, crs: CRS, region: Union[None, gpd.GeoDataFrame] = None
    ):
        _geoms_dir = model_dir.joinpath("geoms")
        if _geoms_dir.is_dir() and any(_geoms_dir.iterdir()):
            logger.info(f"read model geoms from {_geoms_dir}.")
            geoms = self._read_geoms(_geoms_dir, self.crs)
            return geoms
        # FIXME harded way of defininig mdu
        mdu_fn = "dflowfm\FlowFM.mdu"
        mdu = model_dir / mdu_fn
        logger.info(f"read model from {mdu}.")
        model = Delft3DFM()
        model.read_model(mdu, crs=crs)
        model.extract_geometries(region=region)
        geoms = model.convert_geometries()
        # model.save_geometries(geoms_dir)
        return geoms

    def _read_geoms(self, geoms_dir: Path, crs: CRS):
        geoms = {}
        for fn in geoms_dir.glob("*.geojson"):
            name = fn.stem
            logger.debug(f"Reading model file {name}.")
            gdf = gpd.read_file(str(fn))
            if gdf.crs != crs:
                logger.debug(f"reprojecting to {crs.to_epsg()}")
                gdf = gdf.to_crs(crs)
            geoms[name] = gdf
        return geoms


mywrapper = Delft3dfmNetworkWrapper()
geoms = mywrapper.get_geoms_from_model()
