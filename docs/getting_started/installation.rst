.. _installation_guide:

==================
Installation guide
==================

Prerequisites
=============

Python and conda
-----------------------
You'll need **Python 3.9 or greater** and conda as an environment manager.
These package managers help you to install (Python) packages and manage environments
to prevent conflicts between different installations.

Check out Miniconda for conda installation.

* `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__

Dependencies
------------

The HybridUrb Python package makes extensive use of the modern scientific Python
ecosystem. The most important dependencies are listed here (for a complete list,
see the pyproject.toml file in the repository root). These dependencies are automatically installed by 
following the installation steps provided on this page.

Network analysis:

* `NetworkX  <https://networkx.org/>`__

Statistical operations:

* `statsmodels <https://www.statsmodels.org/>`__
* `scikit-learn <https://scikit-learn.org/>`__

Unstructured grid manipulations:

* `meshkernel <https://deltares.github.io/MeshKernelPy/>`__
* `xugrid <https://deltares.github.io/xugrid/>`__

Hydrological model building libraries:

* `hydromt-delft3dfm <https://deltares.github.io/hydromt_delft3dfm>`__


Installation
============

For now, HybridUrb only supports a developer installation, which means setting up the package in a way that allows you to modify its source code. This is typically used by contributors or advanced users who want to work on the development of the package itself.

.. code-block:: console

    git clone https://github.com/xldeltares/hybridurb.git
    cd hybridurb
    conda create -n hybridurb python=3.11 -c conda-forge
    conda activate hybridurb
    pip install hydromt-delft3dfm
    pip install -e .

