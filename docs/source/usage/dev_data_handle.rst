Handling the data
***********************

This section refers to the ``bye_splits/data_handle`` directory. It includes the definition and functionality of event and geometry data, within appropriate classes, both inheriting form the same base class. Additional helper functions are also included.

.. figure:: ../img/flowchart.png
   :align: center

   Code dataflow structure. At an initial stage, event and geometry data are separately processed. A "cache" system is put in place to minimize expensive reads from file. The two data sources are combined to provide a basis for all algorithms, which can be exploited and extended by the user ("client").


Event data
===========

Implemented in ``data_handle/event.py``, in the ``EventData`` class.

The core idea is to provide event-related data as fast as possible while minimizing the number of events to be loaded at any given time.
For an algorithm where one knows in advance which events will be processes, one wants to load as many events as possible at once.
However, for an interactive event display (see :ref:`plots`), one does not know in advance which specific events will be needed, and loading all might not be feasible, or fast.
Thus, a simple system was set in place which detects whether a new event being requested has been requested before, reading it from the input file only if it is being requested for the first time.
Events to load at start can be specifically mentioned via the ``default_events`` argument.

.. note::
   Some ``EventData`` methods differ in their name by a single "s".
   This singular/plural difference aims at promoting code reuse as much as possible, while providing a set of convenience functions for the user, which can process one or mor events using a (hopefully) clear syntax.
   See for instance ``provide_event(...)`` and ``provide_events(...)``.
   

Geometry data
=============

Implemented in ``data_handle/geometry.py``, in the ``GeometryData`` class.

It aims to mimic the official CMSSW HGCAL geometry, originally implemented in ``C++``.
Since it is conceived to be used for visualization only, only *topological* information is processed, while all (extremely complex) details on physics simulations are not present.
Even the shapes are considered simplistically; the result should **not** be seen in any way as a replacement to the original geometry.
Its goal is to provide a reasonable hand-wavy visualization of what an algorithm is doing, potentiating faster algorithmic developments.

The geometry encodes the location of all TCs, silicon modules and scintillator tiles.
It has been validated with respect to the CMSSW implementation.
TCs have a fixed diamond shape for the silicon section, and a variable-sized annular-wedge shape for the scintillator section.
`This PR <https://github.com/LLRCMS/bye_splits/pull/26>`_ provides a good discussion on both shapes. `This one <https://github.com/LLRCMS/bye_splits/pull/8>`_ instead discusses a compromise on the positioning of silicon TCs, given the fact that real TCs can have varying shapes.

