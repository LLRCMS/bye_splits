.. bye_splits documentation master file, created by
   sphinx-quickstart on Wed Nov 20 10:44:48 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation 
===============

Implementation of the CMS HGCAL Trigger Backend Stage 2 in Python, following the CMSSW implementation.

It can be used for quick prototyping, debugging and optimization.
It can also generate a 2D and 3D event visualization apps, since it includes a (very) simplified version of the HGCA geometry.
It was originally used for understanding and fixing the observed cluster splitting.

Various presentations based on the framework are available `here <https://github.com/LLRCMS/bye_splits/wiki/Resources>`_.

.. toctree::
   :maxdepth: 2
   :caption: Installation

   source/usage/installation
   
.. toctree::
   :maxdepth: 1
   :caption: Overview

   source/usage/reconstruction_chain
   source/usage/data_production

.. toctree::
   :maxdepth: 1
   :caption: For Developers

   source/usage/dev_intro
   source/usage/dev_produce
   source/usage/dev_data_handle
   source/usage/dev_plot
   source/usage/dev_scripts
   source/usage/dev_tasks
   source/usage/dev_utils
   
.. toctree::
   :maxdepth: 1
   :caption: Miscellaneous

   source/usage/event_visualization
   source/usage/cluster_radii_studies
   source/usage/tikz

.. toctree::
   :maxdepth: 2
   :caption: Source code

   modules
   
