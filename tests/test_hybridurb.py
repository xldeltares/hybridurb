"""Test for hybridurb"""

import pdb
from os.path import abspath, dirname, join
from pathlib import Path

import pytest
from hydromt.cli.cli_utils import parse_config
from hydromt.log import setuplog

from hybridurb import __version__
from hybridurb.networksutils.network_from_delft3dfm import Delft3dfmNetworkWrapper
from hybridurb.runners.runner import NowcastRunner

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples")


def test_version():
    assert __version__ == "0.2.0.dev0"


def test_runner():
    _runner = NowcastRunner(
        root_dir=Path(EXAMPLEDIR) / ".\Antwerp\option1",
        t0="201605300900",
    )
    _runner.run()


def test_build_from_delft3dfm():
    config_fn = Path(TESTDATADIR) / "config.yml"
    mywrapper = Delft3dfmNetworkWrapper(config_fn)
    mywrapper.get_geoms_from_model()
    mywrapper.get_network_from_geoms()
    # get geoms from delft3dfm
    # setup network from geometry files
    # compute hydraulic params? --> then that requires certain edge and nodes attributes
    # optionmisation --> dag
    # compute flow path
    # mywrapper.get_network_properties()
