.. _installation:

Installation
******************

We recommend using `mamba`_, a fast and robust package manager.
It is fully compatible with ``conda`` packages and supports most of ``conda``’s commands.

In the following we create a new environment called ``NewEnv``:

.. code-block:: shell
   
   mamba create -n NewEnv python=3 pandas uproot pytables h5py plotly
   mamba activate NewEnv

Occasionally, you might find some packages in the `PyPI <https://pypi.org/>`_ that are not present in the ``conda`` repositories.
Assuming the package is called ``NewPackage``, one solution is to run:

.. code-block:: shell
   
   mamba activate NewEnv # if not done before
   python -m pip install NewPackage

which will associate ``NewPackage`` to the ``python`` version (``python --version``) defined in the activated ``mamba`` environment.

.. _mamba: https://mamba.readthedocs.io/en/latest/index.html
