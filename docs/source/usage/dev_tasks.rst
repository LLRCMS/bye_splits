.. _tasks:

Tasks
*******

This section refers to the ``bye_splits/tasks`` directory.
The individual tasks are described in :ref:`reco_chain_overview`.

Each task is defined as **standalone**, and always stores its outputs to a persistent format (usually `HDF5 <https://docs.h5py.org/en/stable/index.html>`_).
These two characteristics should ensure that any task can be run from the command line as a ``Python`` executable (on top of being called as a function), provided that upstream tasks have already produe their expected inputs.
This is advantageous when a particular task has to be optimized, as it avoids rerunning all tasks that came before.
It also helps in the thought process during implementation, as every task must be conceived as a clearly defined step that takes a specific input and produces a specific output, and those inputs and outputs must match when exploring several alternative tasks in parallel.

.. note::

   Once again, none of the above is enforced, so it is up to the programmer to follow these general guidelines.


Whenever needed, new tasks should be placed under the same ``tasks/`` folder, noting that the same task should be written as to be potentially used by more than just one reconstruction chain.
