.. _dev_install:

Developer installation guide
============================

HybridUrb provides two ways to develop the package, using conda or poetry.

Developer install using conda
-----------------------------

First, clone the Hybridurb ``git`` repo from `github <https://github.com/Deltares/hydromt.git>`_. 
Navigate into the the code folder (where the envs folder and pyproject.toml are located):


.. code-block:: console

    $ git clone git@github.com:xldeltares/hybridurb.git
    $ cd hybridurb


Then, create and activate a development environment for HybridUrb:

.. code-block:: console

    $ conda create --name hybridurb-dev python=3.11 -c conda-forge -y
    $ conda activate hybridurb-dev

If you wish to make changes in HybridUrb, you should make an editable install of the plugin.
This can be done with:

.. code-block:: console

    $ pip install -e.
	
If you encounter issues with the installation of some packages, you might consider cleaning conda to remove unused packages and caches.
This can be done through the following command from your base environment:

.. code-block:: console

    $ conda clean -a