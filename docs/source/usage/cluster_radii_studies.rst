Cluster Size Studies
***********************

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


Visualization
==============

A DashApp has been built to interactively explore the effect of cluster size on various cluster properties, which is currently hosted at https://bye-splits-app-hgcal-cl-size-studies.app.cern.ch/.
To run the app locally, you can do:

.. code-block:: shell
				
    bash run_cluster_app.sh <username>

where ``<username>`` is your lxplus username.
The app reads the configuration file ``bye_splits/plot/display_clusters/config.yaml`` and assumes that you have a directory structure equivalent to the directories described in the cluster size step (depending on your choice of ``Local``).

It performs the necessary analysis on the files in the specified directory to generate the data for each page, which are themselves written to files in this directory.
In order to minimize duplication and greatly speed up the user experience, if one of these files does not exist in your own directory, it looks for it under the appropriate directories (listed in our Data Sources), where a large number of the possible files already exist.
The same procedure is used for reading the generated cluster size files, so you can use the app without having had to run the study yourself.
