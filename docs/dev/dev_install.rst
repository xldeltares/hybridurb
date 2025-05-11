.. _dev_install:

Developer installation guide
============================

Conda as environment manager
----------------------------
We recomment using conda as environment manager. You can download and install it `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`_. 

.. code-block:: console
    $ git clone git@github.com:xldeltares/hybridurb.git
    $ cd hybridurb
    $ conda create -n hybridurb python=3.11 -c conda-forge
    $ conda activate hybridurb
    $ pip install hydromt-delft3dfm
    $ pip install -e .

