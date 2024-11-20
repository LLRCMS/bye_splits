Reconstruction Chain
********************

The reconstruction chain is implemented in Python. To run it:

.. code-block:: shell
				
    python bye_splits/run_chain.py

where one can use the ``-h`` flag to visualize available options. To use the steps separately in your own script use the functions defined under ``bye_splits/tasks/``, just as done in the ``iterative_optimization.py`` script.

For plotting results as a function of the optimization trigger cell parameter:

.. code-block:: shell
				
    python plot/meta_algorithm.py

The above will create ``html`` files with interactive outputs.


Cluster Size Studies
======================

The script ``bye_splits/scripts/cluster_size.py`` reads a configuration file ``bye_splits/scripts/cl_size_params.yaml`` and runs the Reconstruction Chain on the ``.root`` inside corresponding to the chosen particle, where the clustering step is repeated for a range of cluster radii that is specified in the parameter file under ``cl_size: Coeffs``.

The most convenient way of running the study is to do:


.. code-block:: shell
				
    bash run_cluster_size.sh <username>

where ``<username>`` is your lxplus username, creating ``.hdf5`` files containing Pandas DFs containing cluster properties (notably energy, eta, phi) and associated gen-level particle information for each radius.
The bash script acts as a wrapper for the python script, setting a few options that are convenient for the cluster size studies that are not the default options for the general reconstruction chain.
As of now, the output ``.hdf5`` files will be written to your local directory using the structure:

.. code-block:: shell
				
    ├── /<base_dir>
    │            ├── out
    │            ├── data
    │            │   ├──new_algos

with the files ending up in ``new_algos/``. Currently working on implementing an option to send the files directly to your ``eos/`` directory, assuming the structure:

.. code-block:: shell
				
    ├── /eos/user/<first_letter>/<username>
    │                                   ├── out
    │                                   ├── data
    │                                   │   ├──PU0
    │                                   │   │   ├──electrons
    │                                   │   │   ├──photons
    │                                   │   │   ├──pions
    │                                   │   ├──PU200
    │                                   │   │   ├──electrons
    │                                   │   │   ├──photons
    │                                   │   │   ├──pions
