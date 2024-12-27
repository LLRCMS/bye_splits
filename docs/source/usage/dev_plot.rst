.. _plots:

Data Plotting
***********************

This section refers to the ``bye_splits/plot`` directory. It is used to plot both the products of the :ref:`scripts`, but also to display the 2D or 3D geometry independently.
Its inner structure is:

+ ``display/``: generic 2D interactive pots using `Bokeh <https://bokeh.org/>`_;
+ ``display_plotly/``: generic 3D interactive plots using `Plotly <https://plotly.com/python/>`_;
+ ``join/``: a working example of how to merge the two plots above using `Flask <https://flask.palletsprojects.com/en/stable/>`_;
+ ``display_clusters/``: 2D interactive plots relating to specific cluster studies, using `Plotly <https://plotly.com/python/>`_ too;
+ a series of independent ``Python`` scripts or classes:

  + ``chain_plotter.py``: where ``ChainPlotter`` is defined, a helper class to gather all plotting-related information in a specific reconstruction chain
  + ``trigger_cell_plotter.py``: which provides a binned visualization of TCs;
  + other scripts, defined for specific reconstruction chains

	
Event Visualization
=====================

The repository creates two web apps that can be visualized in a browser.
The code is stored under ``bye_splits/plot``.

Setup
------

Please install the following from within the ``conda`` environment installed in :ref:`installation`:

.. code-block:: shell
				
    conda install -c conda-forge pyarrow # if the above fails: python -m pip install pyarrow
    python3 -m pip install --upgrade pip setuptools # to avoid annoying "Setuptools is replacing distutils." warning

	
Setup in local browser
----------------------

Since browser usage directly in the server will necessarily be slow, we can use LLR's intranet at ``llruicms01.in2p3.fr:<port>/display``, and forward it to our local machines via ``ssh``.
To establish a connection between the local machine and the remote ``llruicms01`` server, passing by the gate, use:

.. code-block:: shell
				
    ssh -L <port>:llruicms01.in2p3.fr:<port> -N <llr_username>@llrgate01.in2p3.fr

for instance

.. code-block:: shell

   ssh -L 8080:lruicms01.in2p3.fr:8080 -N alves@llrgate01.in2p3.fr

The two ports do not have to be the same, but it avoids possible confusion.
Leave the terminal open and running (it will not produce any output).

Visualization in local browser
==============================

2D display app
--------------

In a new terminal window go to the ``llruicms01`` machines and launch one of the apps, for instance:

.. code-block:: shell
				
    bokeh serve bye_splits/plot/display/ --address llruicms01.in2p3.fr --port <port>  --allow-websocket-origin=localhost:<port>

if visualizing directly at LLR use ``--allow-websocket-origin=llruicms01.in2p3.fr:<port>``.

This uses the server-creation capabilities of ``bokeh``, a ``python`` package for interactive visualization (`docs`_). Note the port number must match.
For further customisation of ``bokeh serve`` see `the serve documentation`_.
The above command should give access to the visualization under ``http://localhost:8080/display``.
For debugging, just run ``python bye_splits/plot/display/main.py``  and see that no errors are raised.

3D display app
--------------

Make sure you have activated your ``conda`` environment.

.. code-block:: shell
				
    conda activate <Env>

Run the following lines. With these commands, some useful packages to run the web application (e.g. ``dash``, ``uproot``, ``awkward``, etc) will be installed in your ``conda`` environment:

.. code-block:: shell
				
    conda install dash
    python3 -m pip install dash-bootstrap-components
    python3 -m pip install dash-bootstrap-templates
    conda install pandas pyyaml numpy bokeh awkward uproot h5py pytables
    conda install -c conda-forge pyarrow fsspec

Then go to the ``llruicms01`` machine (if you are indide LLR intranet) or to your preferred machine and launch:

.. code-block:: shell
				
    python bye_splits/plot/display_plotly/main.py --port 5004 --host localhost

In a browser, go to http://localhost:5004/.
Make sure you have access to the geometry and event files, to be configured in ``config.yaml``.


Visualization with OpenShift OKD4
==================================

We use the `S2I`_ (Source to Image) service via CERN's `PaaS`_ (Platform-as-a-Service) using OpenShift to deploy and host web apps in the `CERN computing environment`_.
There are three ways to deploys such an app: S2I represents the easiest (but less flexible) of the three; `instructions`_.
It effectively abstracts away the need for Dockerfiles.

We will use S2I's simplest configuration possible under ``app.sh``. The image is created alongside the packages specified in ``requirements.txt``. The two latter definitions are `documented`_.

We are currently running a pod at <https://viz2-hgcal-event-display.app.cern.ch/>.
The port being served by ``bokeh`` in ``app.sh`` must match the one the pod is listening to, specified at configuration time before deployment in the `OpenShift management console`_ at CERN.
The `network visibility`_ was also updated to allow access from outside the CERN network.

Additional information
------------------------

+ `What is a pod <https://cloud.google.com/kubernetes-engine/docs/concepts/pod>`_?
+ `How to mount EOS at CERN so that it is accessible by a pod? <https://paas.docs.cern.ch/3._Storage/eos/>`_

Using Flask
============

Flask is a python micro web framework to simplify web development.
It is considered "micro" because it’s lightweight and only provides essential components.
Given that ``plotly``'s dashboard framework, ``dash``, runs on top of ``flask``, and that ``bokeh`` can produce html components programatically (which can be embedded in a ``flask`` app), it should be possible to develop a ``flask``-powered web app mixing these two plotting packages.
Having a common web framework also simplifies future integration.


Flask embedding
-----------------

The embedding of bokeh and plotly plots within flask is currently demonstrated in ``plot/join/app.py``. Two servers run: one from ``flask`` and the other from ``bokeh``, so special care is required to ensure the browser where the app is being served listens to both ports. Listening to ``flask``'s port only will cause the html ``plot/join/templates/embed.html`` to be rendered without bokeh plots.

Note
-----

Running a server is required when more advanced callbacks are needed.
Currently only ``bokeh`` has a server of its own; ``plotly`` simply creates an html block with all the required information.
If not-so-simple callbacks are required for ``plotly`` plots, another port will have to be listened to.

  
.. _docs: https://docs.bokeh.org/en/latest/index.html
.. _the serve documentation: https://docs.bokeh.org/en/latest/docs/reference/command/subcommands/serve.html
.. _S2I: https://docs.openshift.com/container-platform/3.11/creating_images/s2i.html
.. _PaaS: https://paas.docs.cern.ch/
.. _CERN computing environment: https://paas.cern.ch/
.. _instructions: https://paas.docs.cern.ch/2._Deploy_Applications/Deploy_From_Git_Repository/2-deploy-s2i-app/
.. _documented: https://github.com/kubesphere/s2i-python-container/blob/master/2.7/README.md#source-repository-layout
.. _OpenShift management console: https://paas.cern.ch/
.. _network visibility: https://paas.docs.cern.ch/5._Exposing_The_Application/2-network-visibility/
