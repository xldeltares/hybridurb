.. _dev_install:

Developer installation guide
============================

Conda as environment manager
----------------------------
We recommend using conda as environment manager. You can download and install it `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`_. 

If you do not have Conda installed, please follow the instructions `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ to install it.

.. code-block:: console
    git clone https://github.com/xldeltares/hybridurb.git
    cd hybridurb
    conda create -n hybridurb python=3.11 -c conda-forge
    conda activate hybridurb
    pip install hydromt-delft3dfm
    pip install -e .

