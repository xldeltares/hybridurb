examples
========

This folder contains calibrated HybirdUrb model for Antwerp and Gent, as published in `Li & Willems (2022)`_, as well as iPython notebook examples for **building a new HybridUrb model from Delft3DFM model - an example of Eindhoven** (in progress). 

The calibrated examples can be run on your local machine as **standalone**, or as intergrated model (with pre- and post-adaptor) in **`Delft-FEWS`_**. 

Local installation
------------------

To run these examples on your local machine you need a copy of the examples folder 
of the repository and an installation of HybirdUrb including some additional 
packages required to run the notebooks. 

For installation of HybirdUrb, checkout more details in the `installation guide. <https://xldeltares.github.io/hybridurb/getting_started/installation.html>`_

To run the notebooks, you need to install the ``examples`` version of HybirdUrb using pip. The examples version installs additional dependencies
such as jupyter notebook to run the notebooks, matplotlib to plot etc. 
To install or update in an existing environment (hybridurb environment), do:

.. code-block:: console

  $ conda activate hybridurb-dev
  $ pip install "hybridurb[examples]"


Build a model from Delft3D FM Simulation
----------------------------------------

Building a HybirdUrb model from a Delft3D FM Simulation is provided as iPython notebook for Eindhoven. 

Finally, start a jupyter lab server inside the **examples** folder 
after activating the **hybridurb** environment, see below.

Alternatively, you can run the notebooks from `Visual Studio Code <https://code.visualstudio.com/download>`_.

.. code-block:: console

  $ conda activate hybridurb
  $ cd hybridurb/examples
  $ jupyter notebook


Run a model
-----------
Run the Antwerp model in nowcast mode using nowcasts produced at 201605300900:

.. code-block:: console

    $ python -m hybridurb.runners.main .\examples\Antwerp\option1 --mode nowcast --t0 201605300900


Run a model in Delft-FEWS
-------------------------
Run a model in `Delft-FEWS`_ context involves running also the pre- and post-adaptor (example not enclosed):

.. code-block:: console

    $ python -m hybridurb.runners.main .\examples\Antwerp\option1 --mode nowcast --t0 201605300900 --fews

.. _Delft-FEWS: https://oss.deltares.nl/web/delft-fews>


.. _Li & Willems (2022): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019WR025128

