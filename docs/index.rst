Documentation 
===============

Implementation of the CMS HGCAL Trigger Backend Stage 2 in Python, mimicking the CMSSW implementation.

It can be used for quick prototyping, debugging and optimization.
It can also generate event visualization apps, since it includes a (very) simplified version of the HGCA geometry.
It was originally used for understanding and fixing the observed cluster splitting.

Various presentations based on the framework are available `here <https://github.com/LLRCMS/bye_splits/wiki/Resources>`_.

.. warning::
   The project is in a work-in-progress status.
   This means that no guarantees are provided for anything to work out-of-the-box (though most of it should, with little to no change 🙂).

   If you spot clearly outdated code, or if something does not run, please :ref:`contact` me.
   Of course, proposing a code change directly via a `PR <https://github.com/LLRCMS/bye_splits/pulls/>`_ is much welcomed and preferred.

.. toctree::
   :maxdepth: 1
   :caption: Overview

   source/usage/reconstruction_chain
   source/usage/contact
   
.. toctree::
   :maxdepth: 1
   :caption: First Steps

   source/usage/first_installation
   source/usage/first_running
   source/usage/first_write_chain

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

   source/usage/cluster_radii_studies
   source/usage/tikz

.. toctree::
   :maxdepth: 2
   :caption: Source code

   modules
   
