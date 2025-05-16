Examples
========


Local installation
------------------

To run these examples on your local machine, you need a copy of the examples folder
from the repository and an installation of HybridUrb, including some additional
packages required to run the notebooks.

For installation of HybridUrb, check out more details in the `installation guide <https://xldeltares.github.io/hybridurb/getting_started/installation.html>`_.

To run the notebooks, you need to install the ``examples`` version of HybridUrb using pip. The examples version installs additional dependencies
such as Jupyter Notebook to run the notebooks, matplotlib to plot, etc.
To install or update in an existing environment (hybridurb environment), do:

.. code-block:: console

  $ conda activate hybridurb
  $ pip install "hybridurb[examples]"

Running the notebooks
---------------------

To run the example notebooks, first activate the ``hybridurb`` environment and navigate to the ``examples`` directory:

.. code-block:: console

  $ conda activate hybridurb
  $ cd hybridurb/examples

Then, start a Jupyter Notebook or JupyterLab server:

.. code-block:: console

  $ jupyter notebook
  # or
  $ jupyter lab

Alternatively, you can open and run the notebooks directly in `Visual Studio Code <https://code.visualstudio.com/download>`_ with the Python and Jupyter extensions installed.

Use case 1: Build a graph model from Delft3D FM Simulation
----------------------------------------------------------

One of the main topology building blocks of the HybridUrb model is the graph model.
In the latest version of the code, the graph model is extended to Delft3D FM models.
More specifically, the graph model is built from the Delft3D FM simulation results (``*_map.nc`` files) and shapefiles, using the `networkx`_ package.

.. note::

    In its original design, the graph representation of the urban drainage model only contains static features, which were the features extracted from timeseries/hydrological processes to represent a critical state of the urban drainage system (approaching flooding).
    To further extend the usage of the graph model for other types of data-driven models, such as graph neural networks (see `SWMM_GNN_Repository_Paper_version <https://github.com/alextremo0205/SWMM_GNN_Repository_Paper_version>`_ by Alexander Garzón),
    the code also includes the option to add dynamic features directly to the graph model from simulation results.

As an example, the steps to build a graph model from a Delft3D FM Simulation are provided for Eindhoven.
The example is provided in the Jupyter notebook ``build_Eindhoven_from_simulations.ipynb``.

    **Important:**
    Preparing the graph requires the simulation results of the Delft3D FM model.
    For Eindhoven, a series of simulations are made available for testing data-driven approaches.
    Due to the file size of the simulation results exceeding the GitHub limit, the simulation results are not published.
    Please contact the authors for access to the simulation results.
    There is also an example that helps to understand the Eindhoven model simulations; this can be found in the Jupyter notebook ``Understand_Eindhoven_simulations.ipynb``.

Use case 2: Run a model for Antwerp
-----------------------------------

This example folder contains calibrated HybridUrb models for Antwerp and Gent, as published in `Li & Willems (2022)`_. The calibrated examples can be run on your local machine as **standalone**, or as an integrated model (with pre- and post-adaptor) in **`Delft-FEWS`_**.


Run the Antwerp model in nowcast mode using nowcasts produced at 201605300900:

.. code-block:: console

    $ python -m hybridurb.runners.main .\examples\Antwerp\option1 --mode nowcast --t0 201605300900


Running a model in the `Delft-FEWS`_ context involves running also the pre- and post-adaptor (example not enclosed):

.. code-block:: console

    $ python -m hybridurb.runners.main .\examples\Antwerp\option1 --mode nowcast --t0 201605300900 --fews

.. _Delft-FEWS: https://oss.deltares.nl/web/delft-fews

.. _Li & Willems (2022): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019WR025128

.. _networkx: https://networkx.org/
