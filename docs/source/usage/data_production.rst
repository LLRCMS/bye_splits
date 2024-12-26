.. _data_production:
Data Production
******************
   
Skimming
===============

To make the size of the files more manageable, a skimming step was implemented relying on ``ROOT``'s ``RDataFrame``.
Several cuts are applied, and additionally many type conversions are run for ``uproot`` usage at later steps.
To run it:

.. code-block:: shell

   python bye_splits/production/produce.py --nevents -1 --particles photons

where ``--nevents -1`` represents all events, and the input file is defined in ``config.yaml``. 

The output files include, among many others, the following variables:

+---------------+-------------------------------------------------------------------+
| Variable name | Meaning                                                           |
+===============+===================================================================+
| ``tc_*``      | relative to TCs                                                   |
+---------------+-------------------------------------------------------------------+
| ``tc_x/y/z``  | Cartesian coordinates :math:`x/y/z` of the TC                     |
+---------------+-------------------------------------------------------------------+
| ``tc_wu``     | Coordinate :math:`U` of the module where the TC belongs           |
+---------------+-------------------------------------------------------------------+
| ``tc_wv``     | Coordinate :math:`V` of the module where the TC belongs           |
+---------------+-------------------------------------------------------------------+
| ``tc_cu``     | Coordinate :math:`u` of the cell in a module where the TC belongs |
+---------------+-------------------------------------------------------------------+
| ``tc_cv``     | Coordinate :math:`v` of the cell in a module where the TC belongs |
+---------------+-------------------------------------------------------------------+
| ``tc_mipPt``  | Energy of a TC in transverse MIP units                            |
+---------------+-------------------------------------------------------------------+


Data sources
==============

This framework relies on photon-, electron- and pion-gun samples produced via CRAB.
The most up to date versions are currently stored under:

+------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Particle and pile-up   | Path                                                                                                                                                                                                 |
+========================+======================================================================================================================================================================================================+
| Photons (PU0)          | ``/dpm/in2p3.fr/home/cms/trivcat/store/user/lportale/DoublePhoton_FlatPt-1To100/GammaGun_Pt1_100_PU0_HLTSummer20ReRECOMiniAOD_2210_BCSTC-FE-studies_v3-29-1_realbcstc4/221025_153226/0000/``         |
+------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Electrons (PU0)        | ``/dpm/in2p3.fr/home/cms/trivcat/store/user/lportale/DoubleElectron_FlatPt-1To100/ElectronGun_Pt1_100_PU200_HLTSummer20ReRECOMiniAOD_2210_BCSTC-FE-studies_v3-29-1_realbcstc4/221102_102633/0000/``  |
+------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Pions (PU0)            | ``/dpm/in2p3.fr/home/cms/trivcat/store/user/lportale/SinglePion_PT0to200/SinglePion_Pt0_200_PU0_HLTSummer20ReRECOMiniAOD_2210_BCSTC-FE-studies_v3-29-1_realbcstc4/221102_103211/0000``               |
+------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Photons (PU200)        | ``/eos/user/i/iehle/data/PU200/photons/ntuples``                                                                                                                                                     |
+------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Electrons (PU200)      | ``/eos/user/i/iehle/data/PU200/electrons/ntuples``                                                                                                                                                   |
+------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
