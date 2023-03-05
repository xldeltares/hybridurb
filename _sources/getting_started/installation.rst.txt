.. _installation_guide:

==================
Installation guide
==================

Prerequisites
=============

Python and conda
-----------------------
You'll need **Python 3.8 or greater** and conda as an environment manager.
These package managers help you to install (Python) packages and manage environment
such that different installations do not conflict.

Check out Miniconda for conda installation.

* `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__

Dependencies
------------

The HybridUrb Python package makes extensive use of the modern scientific Python
ecosystem. The most important dependencies are listed here (for a complete list,
see the pyproject.toml file in the repository root). These dependencies are automatically installed when 
installing HybridUrb following the steps enclosed in this page.

Network analysis:

* `NetworkX  <https://networkx.org/>`__

Statistical operations:

* `statsmodels <https://www.statsmodels.org/>`__
* `scikit-learn <https://scikit-learn.org/>`__

Unstructured grid maniputations:

* `meshkernel <https://deltares.github.io/MeshKernelPy/>`__
* `xugrid <https://deltares.github.io/xugrid/>`__

Hydrological model building libraries:

* `hydromt <https://deltares.github.io/hydromt>`__


Installation
============

For now HybridUrb only supports developer install.
Please refer to the steps mentioned here: :ref:`Developer install <dev_install>`.
