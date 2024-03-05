.. _readme:

===================================================================
HybridUrb: a Python package to perform hybrid urban flood modelling
===================================================================

HybridUrb is a python package to build and run hybrid models for fast and probabilistic urban pluvial flood prediction.

HybridUrb is a package under active development. 

The pre-release version `v0.1.0-alpha` supports:

- Run a pre-calibrated HybridUrb model in nowcast mode.
- Run nowcast mode with pre- and post-adaptor for Delft-FEWS.

The current vertion `v0.2.0.dev0` will be able to:

- Build and Calibrate a hybrid model from Delft3DFM.
- Perform network analysis using graph-theory via a seperate network module for urban drainage networks. 

Quick start using developer install:

.. code-block:: console

    $ git clone git@github.com:xldeltares/hybridurb.git
    $ cd hybridurb
    $ conda create -f envs/environment_dev.yaml
    # in case of error about permission denied libbz2.dll, try admin installation
    $ conda activate hybridurb-dev
    $ pip install -e .

Reference:

Li, X., & Willems, P. (2020). A hybrid model for fast and probabilistic urban pluvial flood prediction. Water Resources Research, 56, e2019WR025128. https://doi.org/10.1029/2019WR025128

.. _Li & Willems (2022): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019WR025128
