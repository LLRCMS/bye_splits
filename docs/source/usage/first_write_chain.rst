.. _write_new_reco_chain:
Write a new Reconstruction Chain
***************************************

By **reconstruction chain** we mean a series of algorithmic steps, or :ref:`tasks`, which lead from the input datasets to an output later exploited by the L1 trigger. In other words, leading to useful TPG outputs.
The ``scripts/run_default_chain.py`` chain mimics what CMSSW does, and can be used as template for future studies.
Other chain do **not** replicate CMSSW; they are precisely meant to explore what can be done beyond the original ideas.

If you want to write a new reconstruction chain, you should define a new file under ``scripts/``, just like ``scripts/run_default_chain.py``.
This is not enforced, but helps keeping the repository's structure under control.

Recipe
------

Within your new ``run_awesome_chain()`` function (defined within a new homonymous file), define the following, in order:

+ a ``validation.Collector()`` object to group statistics of interest;
+ a ``chain_plotter.ChainPlotter()`` object to handle all required plotting steps;
+ use the ``get_data_reco_chain_start()`` function to access the input datasets, where ``nevents`` sets the number of events to consider (the more events one uses, the slowest the chain becomes), and ``particles`` and ``pu`` set the particle type (photons, pions, ...) and pile-up number, respectively, calling the appropriate ``ROOT`` input files;
+ call whichever :ref:`tasks` you want, sequentially; make sure each task can be treated as a standalone executable.

.. note::
   None of these steps are enforced, i.e., there is no mandatory *interface*.
   This should be changed in case the project grows significantly, but for the moment its absence provides more flexibility.

If you need to write new plotting scripts, add them to the ``plot/`` folder.
For new :ref:`tasks`, use the ``tasks/`` folder.

