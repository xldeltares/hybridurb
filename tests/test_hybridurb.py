"""Test for hybridurb"""

import pdb
from os.path import abspath, dirname, join

import pytest
from hydromt.cli.cli_utils import parse_config
from hydromt.log import setuplog

from hybridurb import __version__
from hybridurb.networksutils.network_from_delft3dfm import Delft3dfmNetworkWrapper

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples")


_models = {
    "Eindhoven": {
        "config_fn": "config.yml",
    },
}


def test_version():
    assert __version__ == "0.2.0.dev0"


@pytest.mark.parametrize("model", list(_models.keys()))
def test_delft3dfm_convertor(model):
    _model = _models[model]
    config_fn = join(TESTDATADIR, _model["config_fn"])
    mywrapper = Delft3dfmNetworkWrapper(config_fn)
    # get geoms from delft3dfm
    # setup network from geometry files
    # compute hydraulic params? --> then that requires certain edge and nodes attributes
    # optionmisation --> dag
    # compute flow path
    mywrapper.get_geoms_from_model()
    mywrapper.get_network_from_geoms()
    # mywrapper.get_network_properties()
