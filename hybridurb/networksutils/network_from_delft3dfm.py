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
import os
import logging
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
from pyproj import CRS

from hydromt.config import configread, configwrite
from hydromt.data_catalog import DataCatalog
from hydromt.cli.cli_utils import parse_config
from hydromt.log import setuplog

from delft3dfm_utils import Delft3DFM
from network import NetworkModel


logger = logging.getLogger(__name__)


class Delft3dfmNetworkWrapper:
    """
    This is a wrapper class for the Delft3D flexible mesh (FM) type Network.

    Methods:
    setup_delft3dfm
    setup_network
    """

    _config_fn = Path(__file__).with_name("config.yml")
    _data_catalog = Path(__file__).with_name("data_catalog_delft3dfm.yml")
    _networkopt = Path(__file__).with_name(
        "networkopt_delft3dfm.ini"
    )  # FIXME switch to yml
    _delft3dfmmodel: Delft3DFM = None
    _networkmodel: NetworkModel = None

    def __init__(self, config_fn: Path = _config_fn) -> None:
        """
        The constructor for the Delft3dfmNetworkWrapper class.

        :param config_fn: The configuration file path. Defaults to _config_fn.
        """
        self.config = configread(config_fn=config_fn)
        self._init_global()
        self.logger = setuplog(
            __name__, self.root.joinpath("hybridurb.log"), log_level=10
        )

    def _init_global(self):
        """
        Initializes global settings from the configuration file.
        """
        _global = self.config.get("global", {})
        self.root = Path(_global.get("root", Path.cwd))
        crs = _global.get("crs")
        self.crs = None if not crs else CRS.from_user_input(crs)
        region = _global.get("region")
        self.region = None if not region else gpd.read_file(region)

    def get_geoms_from_model(self):
        """
        Retrieves geometries from the model.

        :return: Geometries from the model. If the function is unable to retrieve the geometries, it returns None.
        """
        _config = self.config.get("get_geoms_from_model")
        if not _config:
            return None

        geoms_dir = self.root.joinpath(_config.get("geoms_dir", "geoms_dir"))
        model_dir = self.root.joinpath(_config.get("model_dir"))

        if self._is_dir_valid(geoms_dir):
            self.logger.info(f"reading geoms from {geoms_dir}")
            geoms = self._read_geoms(geoms_dir, self.crs)
        elif self._is_dir_valid(model_dir):
            logging.info(f"reading model from {model_dir}")
            geoms = self.get_geoms(model_dir, self.crs, self.region, geoms_dir)
        else:
            self.logger.error("could not perform get_geoms_from_model.")
            return None

        self._init_datacatalog(geoms_dir, geoms.keys())
        self._init_networkopt(geoms_dir, geoms.keys())
        self._init_region()

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
        self,
        model_dir: Path,
        crs: CRS,
        region: Union[None, gpd.GeoDataFrame] = None,
        geoms_dir: Path = None,
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
        self.logger.info(f"read model from {mdu}.")
        model = Delft3DFM()
        model.read_model(mdu, crs=crs)
        model.extract_geometries(region=region)
        geoms = model.convert_geometries()
        if geoms_dir is not None:
            model.save_geometries(geoms_dir)
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
            self.logger.debug(f"Reading model file {name}.")
            gdf = gpd.read_file(str(fn))
            if gdf.crs != crs:
                self.logger.debug(f"reprojecting to {crs.to_epsg()}")
                gdf = gdf.to_crs(crs)
            geoms[name] = gdf
        return geoms

    def _init_region(self):
        # FI
        if self.region is not None:
            self.region.to_file(self.root.joinpath("region.geojson"))

    def _init_datacatalog(self, geom_root: Path, geom_keys: list[str] = None):
        # add root
        data_catalog_dict = {"root": str(geom_root.as_posix())}

        # add keys
        _data_catalog_dict = configread(config_fn=self._data_catalog)
        data_catalog_dict.update({key: _data_catalog_dict[key] for key in geom_keys})

        # write
        data_catalog = self.root.joinpath(self._data_catalog.name)
        configwrite(data_catalog, data_catalog_dict)

        self.data_catalog = data_catalog

    def _init_networkopt(self, geom_root: Path, geom_keys: list[str] = None):
        # add report
        networkopt_dict = {
            "setup_basemaps": {
                "region": "{'geom': 'region.geojson'}",
                "report": f"network from delft3dfm geometires: {geom_root}",
            }
        }

        # add keys
        _networkopt_dict = parse_config(self._networkopt)
        networkopt_dict.update({"setup_graph": _networkopt_dict.get("setup_graph")})
        for key, value in _networkopt_dict.items():
            geom_fn = (
                value.get("edges_fn") if "edges_fn" in value else value.get("nodes_fn")
            )
            if geom_fn in geom_keys:
                networkopt_dict.update({key: value})

        # write
        networkopt = self.root.joinpath(self._networkopt.name)
        configwrite(networkopt, networkopt_dict)

        self.networkopt = networkopt

    def setup_network(self):
        """setup network"""

        # FIXME
        # get the current working directory
        cwd = Path.cwd()

        # change the current working directory
        os.chdir(self.root)

        # Initialise
        model = NetworkModel(
            root=self.root,
            mode="w",
            data_libs=self.data_catalog,
            logger=self.logger,
        )
        # Build method options
        opt = parse_config(self.networkopt)
        # Build model
        model.build(opt=opt)

        # change the current working directory back
        os.chdir(cwd)

        pass

    def optimise_network(self):
        pass


mywrapper = Delft3dfmNetworkWrapper()
mywrapper.get_geoms_from_model()
mywrapper.setup_network()
