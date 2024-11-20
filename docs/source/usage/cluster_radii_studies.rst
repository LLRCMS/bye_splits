Cluster Radii Studies
**********************

A DashApp has been built to interactively explore the effect of cluster size on various cluster properties, which is currently hosted at https://bye-splits-app-hgcal-cl-size-studies.app.cern.ch/.
To run the app locally, you can do:

.. code-block:: shell
				
    bash run_cluster_app.sh <username>

where ``<username>`` is your lxplus username.
The app reads the configuration file ``bye_splits/plot/display_clusters/config.yaml`` and assumes that you have a directory structure equivalent to the directories described in the cluster size step (depending on your choice of ``Local``).

It performs the necessary analysis on the files in the specified directory to generate the data for each page, which are themselves written to files in this directory.
In order to minimize duplication and greatly speed up the user experience, if one of these files does not exist in your own directory, it looks for it under the appropriate directories (listed in our Data Sources), where a large number of the possible files already exist.
The same procedure is used for reading the generated cluster size files, so you can use the app without having had to run the study yourself.
