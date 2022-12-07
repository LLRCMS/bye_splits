#include <iostream>
#include "include/skim.h"

//Run with ./skimming.exe photon
int main(int argc, char **argv) {
  std::string dir = "/eos/user/b/bfontana/FPGAs/new_algos/";
  std::string tree_name = "FloatingpointMixedbcstcrealsig4DummyHistomaxxydr015GenmatchGenclustersntuple/HGCalTriggerNtuple";
  std::string particle = std::string(argv[1]);
  std::string infile = particle + "_0PU_bc_stc_hadd.root";
  std::string outfile = "skim_" + infile;
  skim(tree_name, dir + infile, dir + outfile, particle);
  return 0;
}
