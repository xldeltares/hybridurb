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
from typing import Union

import geopandas as gpd
import numpy as np
from pyproj import CRS

from hydromt.config import configread
from hydromt.data_catalog import DataCatalog

from delft3dfm_utils import Delft3DFM


logger = logging.getLogger(__name__)


class Delft3dfmNetworkWrapper:
    """
    This is a wrapper class for the Delft3D flexible mesh (FM) network.
    """

    _config_fn = Path(__file__).with_name("config.yml")
    _data_catalog = Path(__file__).with_name("data_catalog.yml")

    def __init__(self, config_fn: Path = _config_fn) -> None:
        """
        The constructor for the Delft3dfmNetworkWrapper class.

        :param config_fn: The configuration file path. Defaults to _config_fn.
        """
        self.config = configread(config_fn=config_fn)
        self._init_global()

    def _init_global(self):
        """
        Initializes global settings from the configuration file.
        """
        _global = self.config.get("global", {})
        self.root = Path(_global.get("root", Path.cwd))
        region = _global.get("region")
        self.region = None if not region else gpd.read_file(region)
        crs = _global.get("crs")
        self.crs = None if not crs else CRS.from_user_input(crs)

    def get_geoms_from_model(self):
        """
        Retrieves geometries from the model.

        :return: Geometries from the model. If the function is unable to retrieve the geometries, it returns None.
        """
        _config = self.config.get("get_geoms_from_model")
        if not _config:
            return None

        geoms_dir = self.root.joinpath(_config.get("geoms_dir"))
        model_dir = self.root.joinpath(_config.get("model_dir"))

        if self._is_dir_valid(geoms_dir):
            logger.info(f"reading geoms from {geoms_dir}")
            geoms = self._read_geoms(geoms_dir, self.crs)
        elif self._is_dir_valid(model_dir):
            logging.info(f"reading model from {model_dir}")
            geoms = self.get_geoms(model_dir, self.crs, self.region)
        else:
            logger.error("could not perform get_geoms_from_model.")
            return None

        return geoms

    def _is_dir_valid(self, my_dir: Union[Path, None]) -> bool:
        """
        Checks if a given directory is valid.

        :param my_dir: The directory to check.
        :return: True if the directory is valid, False otherwise.
        """
        if (my_dir is not None) and my_dir.is_dir() and any(my_dir.iterdir()):
            return True
        else:
            return False

    def get_geoms(
        self, model_dir: Path, crs: CRS, region: Union[None, gpd.GeoDataFrame] = None
    ):
        """
        Retrieves geometries from a given model directory.

        :param model_dir: The model directory.
        :param crs: The coordinate reference system.
        :param region: The region. Defaults to None.
        :return: The geometries from the model directory.
        """
        mdu_fn = "dflowfm\FlowFM.mdu"
        mdu = model_dir / mdu_fn
        logger.info(f"read model from {mdu}.")
        model = Delft3DFM()
        model.read_model(mdu, crs=crs)
        model.extract_geometries(region=region)
        geoms = model.convert_geometries()
        return geoms

    def _read_geoms(self, geoms_dir: Path, crs: CRS):
        """
        Reads geometries from a given directory.

        :param geoms_dir: The geometries directory.
        :param crs: The coordinate reference system.
        :return: The geometries from the directory.
        """
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

    def setup_network(self):
        pass

    def optimise_network(self):
        pass


mywrapper = Delft3dfmNetworkWrapper()
geoms = mywrapper.get_geoms_from_model()
