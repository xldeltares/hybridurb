.. _dev_install:

Developer installation guide
============================

HybridUrb provides two ways to develop the package, using conda or poetry.

Developer install using conda
-----------------------------

First, clone the Hybridurb ``git`` repo from `github <https://github.com/Deltares/hydromt.git>`_.

.. code-block:: console

    $ git clone git@github.com:xldeltares/hybridurb.git
    $ cd hybridurb

Then, navigate into the the code folder (where the envs folder and pyproject.toml are located), creates a development environment with all the dependencies required by HybridUrb.:

.. code-block:: console

    $ conda env create -f envs/environment_dev.yaml
    $ conda activate hybridurb-dev

Finally, create a developer installation of HybridUrb:

.. code-block:: console

    $ conda develop .
	
Developer install using poetry
------------------------------
First, clone the Hybridurb ``git`` repo from `github <https://github.com/Deltares/hydromt.git>`_.

.. code-block:: console

    $ git clone git@github.com:xldeltares/hybridurb.git
    $ cd hybridurb

Then, navigate into the the code folder (where the pyproject.toml are located), creates an environment for HybridUrb:

.. code-block:: console

    $ conda env create -f environment.yaml
    $ conda activate hybridurb
	
Finally, create a developer installation of HybridUrb:

.. code-block:: console

    $ poetry install
