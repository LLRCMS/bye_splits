import os
import numpy as np
import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from filling import filling
from smoothing import smoothing
from seeding import seeding
from clustering import clustering
from validation import validation

import random
random.seed(18) # fix seed for reproducibility

#### Useful Airflow Commands ######################
# initialize database: `airflow db init`
# check the dag is loaded: `airflow dags list`
# print tasks associatedto a dag: `airflow tasks list <dag_id> --tree`
# test a single dag task: `airflow tasks test stage2_reconstruction_default filling 2022-02-25`
# https://predictivehacks.com/?all-tips=how-to-interact-with-airflow-via-the-command-line
###################################################

Nevents = {{ dag_run.conf.nevents }}
print(Nevents)
quit()


NbinsRz = 42
NbinsPhi = 216
MinROverZ = 0.076
MaxROverZ = 0.58
DataFolder = 'data'

base_kwargs = {
    'NbinsRz': NbinsRz,
    'NbinsPhi': NbinsPhi,
    'MinROverZ': MinROverZ,
    'MaxROverZ': MaxROverZ,
    'LayerEdges': [0,28],
    'IsHCAL': False,

    'Debug': False,
    'DataFolder': DataFolder,
    'FesAlgos': ['ThresholdDummyHistomaxnoareath20'],
    'BasePath': os.path.join(os.environ['PWD'], DataFolder),

    'RzBinEdges': np.linspace( MinROverZ, MaxROverZ, num=NbinsRz+1 ),
    'PhiBinEdges': np.linspace( -np.pi, np.pi, num=NbinsPhi+1 ),
}
if len(base_kwargs['FesAlgos'])!=1:
    raise ValueError('The event number in the clustering task'
                     ' assumes there is only on algo.\n'
                     'The script must be adapted.')


def setDictionary(adict):
    adict.update(base_kwargs)
    return adict
    
_fillBasePath = lambda x : os.path.join( base_kwargs['BasePath'], x)

# filling task
filling_kwargs = setDictionary(
    { 'Nevents': Nevents,
      'FillingIn': _fillBasePath('gen_cl3d_tc.hdf5'),
      'FillingOut': _fillBasePath('filling.hdf5') }
)

# smoothing task
smoothing_kwargs = setDictionary(
    { #copied from L1Trigger/L1THGCal/python/hgcalBackEndLayer2Producer_cfi.py
        'BinSums': (13,               # 0
                    11, 11, 11,       # 1 - 3
                    9, 9, 9,          # 4 - 6
                    7, 7, 7, 7, 7, 7,  # 7 - 12
                    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,  # 13 - 27
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3  # 28 - 41
                    ),
        'SeedsNormByArea': False,
        'AreaPerTriggerCell': 4.91E-05,
        'SmoothingIn': filling_kwargs['FillingOut'],
        'SmoothingOut': _fillBasePath('smoothing.hdf5') }
    )

# seeding task
seeding_kwargs = setDictionary(
    { 'SeedingIn': smoothing_kwargs['SmoothingOut'],
      'SeedingOut': _fillBasePath('seeding.hdf5'),
      'histoThreshold': 20. }
    )

# clustering task
clustering_kwargs = setDictionary(
    { 'ClusteringInTC': filling_kwargs['FillingOut'],
      'ClusteringInSeeds': seeding_kwargs['SeedingOut'],
      'ClusteringOut': _fillBasePath('clustering.hdf5'),
      'CoeffA': ( (0.015,)*7 + (0.020,)*7 + (0.030,)*7 + (0.040,)*7 + #EM
                  (0.040,)*6 + (0.050,)*6 + # FH
                  (0.050,)*12 ), # BH
      'CoeffB': 0,
      'MidRadius': 2.3,
      'PtC3dThreshold': 0.5 }
)

# validation task
validation_kwargs = setDictionary(
    { 'ClusteringOut': clustering_kwargs['ClusteringOut'],
      'FillingOut': filling_kwargs['FillingOut'] }
)

with DAG(
    dag_id='stage2_reconstruction_default',
    default_args={
        'depends_on_past': False,
        'email': ['bruno.alves@cern.ch'],
        'email_on_failure': False,
        'email_on_retry': False,
        'start_date': datetime.datetime(2022, 2, 25, 00, 00),
        'retries': 1,
        'retry_delay': datetime.timedelta(minutes=5),
        # 'queue': 'bash_queue',
        # 'pool': 'backfill',
        # 'priority_weight': 10,
        # 'end_date': datetime(2016, 1, 1),
        # 'wait_for_downstream': False,
        # 'sla': timedelta(hours=2),
        # 'execution_timeout': timedelta(seconds=300),
        # 'on_failure_callback': some_function,
        # 'on_success_callback': some_other_function,
        # 'on_retry_callback': another_function,
        # 'sla_miss_callback': yet_another_function,
        # 'trigger_rule': 'all_success'
    },
    description='Seeding + Clustering',
    catchup=False,
    tags=['v1'],
) as dag:

    fill = PythonOperator(
        task_id='filling',
        python_callable=filling,
        op_kwargs=filling_kwargs,
    )

    smooth = PythonOperator(
        task_id='smoothing',
        python_callable=smoothing,
        op_kwargs=smoothing_kwargs,
    )

    seed = PythonOperator(
        task_id='seeding',
        python_callable=seeding,
        op_kwargs=seeding_kwargs,
    )

    cluster = PythonOperator(
        task_id='clustering',
        python_callable=clustering,
        op_kwargs=clustering_kwargs,
    )

    valid = PythonOperator(
        task_id='validation',
        python_callable=validation,
        op_kwargs=validation_kwargs,
    )

    fill >> smooth >> seed >> cluster >> valid
