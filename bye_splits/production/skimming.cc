#include <iostream>
#include "include/skim.h"

int main() {
  std::string dir = "../../data/new_algos/";
  std::string tree_name = "FloatingpointMixedbcstcrealsig4DummyHistomaxxydr015GenmatchGenclustersntuple/HGCalTriggerNtuple";
  std::string infile = "photon_0PU_bc_stc_hadd.root";
  std::string outfile = "skim_" + infile;

  skim(tree_name, dir + infile, dir + outfile);
  return 0;
}
