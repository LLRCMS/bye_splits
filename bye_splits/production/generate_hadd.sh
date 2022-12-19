#!/bin/bash

# Base paths to root files on /dpm
photon_base_path='root://polgrid4.in2p3.fr//dpm/in2p3.fr/home/cms/trivcat/store/user/lportale/DoublePhoton_FlatPt-1To100/GammaGun_Pt1_100_PU200_HLTSummer20ReRECOMiniAOD_2210_BCSTC-FE-studies_v3-29-1_realbcstc4/221102_143035/0000/'
electron_base_path='root://polgrid4.in2p3.fr//dpm/in2p3.fr/home/cms/trivcat/store/user/lportale/DoubleElectron_FlatPt-1To100/ElectronGun_Pt1_100_PU200_HLTSummer20ReRECOMiniAOD_2210_BCSTC-FE-studies_v3-29-1_realbcstc4/221102_102633/0000/'

# List of root files
photon_files=`gfal-ls $photon_base_path`
electron_files=`gfal-ls $electron_base_path`

# Output path
out_path='/data_CMS/cms/ehle/L1HGCAL'

# Output names
photon_out='photon_200PU_bc_stc_hadd.root'
electron_out='electron_200PU_bc_stc_hadd.root'

# Combine files into one.
# Options: -k = Skip corrupt or non-existent files, do not exit
#          -j = Parallelize the execution in multiple processes
photon_comm="hadd -k -j ${out_path}/${photon_out}"
#echo "Combining photon files to be placed into: ${photon_comm}."
for file in $photon_files
do
  photon_comm+=" ${photon_base_path}${file}"
done

photon_command=`$photon_comm`
#echo $photon_command

electron_comm="hadd -k -j ${out_path}/${electron_out}"
echo "Combining electron files to be placed into: ${electron_comm}."
for file in $electron_files
do
  electron_comm+=" ${electron_base_path}${file}"
done

electron_command=`$electron_comm`
echo $electron_command

echo "Files have been combined."
