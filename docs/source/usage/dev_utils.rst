Utilities
***********************

This section refers to the ``bye_splits/utils`` directory.
It encodes all functions and classes which are used to support the main operations of the framework.
In the following the list the most relevant ones.

Common parameters
-----------------

Stored in ``common.py``.
Stores most functions and classes that are not clearly associated to a given task, and which can thus be used in a variety of scenarios.

Taks parameters
-----------------

Stored in ``params.py``.
Handles the input parameters for each task.
When implementing a new task, one should make sure it is also supported here, which amounts to adding the corresponding options to the ``config.yaml`` configuration file.

Parameter parsing
-----------------

Stored in ``parsing.py``.
Common location to store all ``argparse.ArgumentParser()`` objects used across different reconstruction chains.
Many options can be shared across different chains, as chains can share most of their tasks, which are the functions that consume most of the parameters being passed by the user.

Bye-splits algorithm
--------------------

The original implementation of the bye-splits algorithm is stored in ``iterative_optimization.py``.

.. warning::

   Given the project's timeline, the ``bye-splits`` macro is currently not running.
   A successfull update would not alter the algorithm, but would have to take into account I/O and structure changes that were introduced since the algorithm was originally implemented and tested.
