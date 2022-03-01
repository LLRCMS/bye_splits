import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Plot trigger cells occupancy.')
parser.add_argument('--nphibins', help='number of uniform phi bins',
                    default=216, type=int)
parser.add_argument('--nrzbins', help='number of uniform R/z bins',
                    default=42, type=int)
parser.add_argument('--nevents', help='number of events to display',
                    default=8, type=int)
parser.add_argument('--ledges', help='layer edges (if -1 is added the full range is also included)', default=[0,28], nargs='+', type=int)
parser.add_argument('--pos_endcap', help='Use only the positive endcap.',
                    default=True, type=bool)
parser.add_argument('--hcal', help='Consider HCAL instead of default ECAL.', action='store_true')
parser.add_argument('--minROverZ', help='Minimum value of R/z, as defined in CMSSW.',
                    default=0.076, type=float)
parser.add_argument('--maxROverZ', help='Maximum value of R/z, as defined in CMSSW.',
                    default=0.58, type=float)
parser.add_argument('-d', '--debug', help='debug mode', action='store_true')
parser.add_argument('-l', '--log', help='use color log scale', action='store_true')

_flags = parser.parse_args()

Nevents = _flags.nevents
MaxROverZ = _flags.maxROverZ
MinROverZ = _flags.minROverZ

NbinsRz = _flags.nrzbins
NbinsPhi = _flags.nphibins
RzBinEdges = np.linspace( _flags.minROverZ, _flags.maxROverZ, num=NbinsRz+1 )
PhiBinEdges = np.linspace( -np.pi, np.pi, num=NbinsPhi+1 )
BinDistRz = RzBinEdges[1] - RzBinEdges[0] #assumes the binning is regular
BinDistPhi = PhiBinEdges[1] - PhiBinEdges[0] #assumes the binning is regular

Debug = _flags.debug
DataFolder = 'data'
FesAlgos = ['ThresholdDummyHistomaxnoareath20']
Seed = 18
BasePath = os.path.join(os.environ['PWD'], DataFolder)
_fillBasePath = lambda x : os.path.join( BasePath, x)

# filling task
FillingIn = _fillBasePath('gen_cl3d_tc.hdf5')
FillingOut = _fillBasePath('filling.hdf5')

# smoothing task
BinSums = (13,               # 0
           11, 11, 11,       # 1 - 3
           9, 9, 9,          # 4 - 6
           7, 7, 7, 7, 7, 7,  # 7 - 12
           5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,  # 13 - 27
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3  # 28 - 41
           ) #copied from L1Trigger/L1THGCal/python/hgcalBackEndLayer2Producer_cfi.py
SeedsNormByArea = False
areaPerTriggerCell = 4.91E-05
SmoothingOut = _fillBasePath('smoothing.hdf5')

# seeding task
SeedingOut = _fillBasePath('seeding.hdf5')
histoThreshold = 20.

# clustering task
ClusteringOut = _fillBasePath('clustering.hdf5')
CoeffA = ( (0.015,)*7 + (0.020,)*7 + (0.030,)*7 + (0.040,)*7 + #EM
           (0.040,)*6 + (0.050,)*6 + # FH
           (0.050,)*12 ) # BH
CoeffB = 0
MidRadius = 2.3
PtC3dThreshold = 0.5
