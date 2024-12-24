Introduction
***********************

Thanks for considering contributing to this repository!
In the :ref:`reco_chain_overview` section you should have understood **what** this framework aims to do.
Here I will cover **how** it does it.

The code's structure is simple, and can be found under the ``bye_splits`` folder in the repository root folder:

+ ``production``: skimming code to speed-up the executables, which benefit form smaller inputs;
+ ``data_handle``: handles most pure data processing operations;
+ ``plot``: contains most data plotting code, either 2D or 3D, static or interactive, to be used as a standalone or as part of an executable;
+ ``scripts``: contains the executables, which will in turn run certain data production steps, plotting steps, or tasks;
+ ``tasks``: single tasks as performed by the TPG, as described in the :ref:`reco_chain_overview` section, where each task ca nalso be run independently, given inputs in the correct format

In the next subsections we detail each of the above in detail.
