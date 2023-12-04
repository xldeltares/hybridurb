.. _dev_install:

Developer installation guide
============================


Conda as environment manager
----------------------------
We recomment using conda as environment manager. You can download and install it `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`_. 


After installation, we recommond install libmamba as the solver for its faster performance. See explaination `here <https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community>`_.

.. code-block:: console

  $ conda install -n base conda-libmamba-solver
  $ conda config --set solver libmamba


Developer install of Hybridurb using conda
------------------------------------------

First, clone the Hybridurb ``git`` repo from `github <https://github.com/Deltares/hydromt.git>`_. 
Navigate into the the code folder (where the envs folder and pyproject.toml are located):


.. code-block:: console

    $ git clone git@github.com:xldeltares/hybridurb.git
    $ cd hybridurb



Then, make and activate a new hybridurb-dev conda environment based on the envs/environment_dev.yml
file contained in the repository:

.. code-block:: console

    $ conda create -f envs/environment_dev.yaml
    $ conda activate hybridurb-dev

If you wish to make changes in HybridUrb, you should make an editable install of the plugin.
This can be done with:

.. code-block:: console

    $ pip install -e.
	
If you encounter issues with the installation of some packages, you might consider cleaning conda to remove unused packages and caches.
This can be done through the following command from your base environment:

.. code-block:: console

    $ conda clean -a