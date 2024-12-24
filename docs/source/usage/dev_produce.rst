Production (skimming)
***********************

This section refers to the ``bye_splits/produce`` directory. The ``C++`` code is kept for reference only; you should use the ``produce.py`` (single) executable.

To get instructions on **how** to run this step, please go to the :ref:`data_production` section. In this section the ``produce.py`` scripts is described.

The code is implemented in ``Python`` to ease the production task for the user, skipping the need to compile the source code with the required ``ROOT`` packages, and avoiding the usage of two programming languages in the same context.
This enables the user, for instance, to use the same ``config.yaml`` file for everything, without additional conversion steps.
On the negative side, the ``ROOT.gInterpreter.Declare(...)`` syntax is employed for some ``RDataFrame`` operations which require ``C++`` code, which are harder to debug.

.. note::
   ``RDataFrame`` is preferred over ``uproot`` due to its loading speed and parallel processing. It is instead inferior from a convenience, readability an functionality point of view, which explains why ``uproot`` is preferred for all the :ref:`tasks`.
   The conversion from one to the other requires the ``convertInt/Uint/Float(...)`` functions, otherwise ``uproot`` cannot read the skim output files.

The aim of the code is to reduce the size of the input files.
This is more important the larger the original files are, thus being more relevant for files including pile-up.
The selection cuts are applied both *per-event* and *within an event*.
For instance, events without clusters are discarded, and clusters with negative pseudo-rapidity *within each event* are as well removed (in this case to focus on a single endcap).

Selection within an event are exploited using the following characteristic (and convoluted) syntax:

.. code-block:: python

	df = df.Define("filtered_object", "object[mask]")

where ``object`` is a vector referring to a single event and containing some event property (energy, momentum, ...) and ``mask`` is a boolean vector with the same length as ``object`` which selects only the events passing a given condition.
We deal mostly with vectors since each event can have multiple generated particles, clusters, and so on.

.. warning::
   The choice of the used ``RDataFrame`` methods was often constrained by the version of ``RDataFrame`` that was available.
   Keep in mind that recent versions often include more intuitive methods to apply the same selections.
   Recent versions are also better for other functionalities, as for instance the inclusion of a progress bar, which is quite handy during the (somewhat lengthy) processing of hundreds of thousands of events.

The following selections are applied:

+ positive endcap, for simplicity, with ``tc_zside == 1`` and ``cl3d_eta > 0``
+ look only at odd layers, where TCs are located, using ``disconnectedTriggerLayers``
+ remove various issues in the simulation step with ``genpart_gen != -1``
+ select only particles with a specific PID with ``genpart_pid``
+ select converted or unconverted photons, based on the ``reachedEE`` variable
+ perform generator matching with the ``calcDeltaR`` function, using ``deltarThreshold`` as maximum radial distance threshold
+ minimum TC energy with ``tc_mipPt``, to mimic what the actual TPG does

.. warning::
   The framework was never tested with the negative endcap.
   Small element coordinate's misalignments might introduce unexpected effects, as well as unwanted dependences on the sign of the *x*, *y*, or *z*, coordinates.
   We recommend anyone testing the negative endcap to verify the outputs of every single task in the :ref:`tasks` section.

.. note::
   Vectors with the ``gen_`` suffix refer to the generator step, while the ``genpart_`` suffix refer to ``Geant4`` (simulation). As an example, the Higgs boson is included in the former but not in the latter.

.. note::
   The ``reachedEE`` selection can take three values:

   + ``0``: converted photons (photons converting before reaching the surface of HGCAL)
   + ``1``: photons that missed HGCAL
   + ``2``: photons that hit HGCAL
