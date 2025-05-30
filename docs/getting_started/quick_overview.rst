.. _quick_overview:

==============
Quick overview
==============

Usage
=====
HybridUrb is a python package to build and run simplified 1D models for urban flood nowcasting using hybrid method proposed by `Li & Willems (2022)`_.
HybridUrb is a package under development. The current version only support using the package to run a pre-calibrated HybridUrb model in nowcast mode.

.. _Li & Willems (2022): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019WR025128

Example
=======

Install HybridUrb
-----------------
For now HybridUrb only support developer install in :ref:`Developer install <dev_install>`.

To check if the installation was successful by running the command below. 
This returns the available arguments for HybridUrb.

.. code-block:: console

    $ conda activate hybridurb-dev
    $ python -m hybridurb.runners.main --help

    >>  Usage: python -m hybridurb.runners.main [OPTIONS] ROOT_DIR

        Options:
          --mode TEXT  Mode of the run: hindcast (not implemented), nowcast
          --t0 TEXT    T0 of the run (%Y%m%d%H%M)
          --name TEXT  name of the model.
          --crs TEXT   crs of the model. If not specified, use EPSG:4326
          --fews       Run within FEWS. Include exporting MayLayerFiles for Delft-
                       FEWS. If True, support input format: [t0]_catchemnts.nc. If False,
                       support input format: [t0]_ens*.csv
          --help       Show this message and exit.


For more usage examples, go to the examples directory to check.
