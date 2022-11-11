#include "include/skim.h"

void skim(std::string tn, std::string inf, std::string outf) {

  YAML::Node config = YAML::LoadFile("bye_splits/production/prod_params.yaml");
  std::vector<int> vec;
  if (config["disconnectedTriggerLayers"]) {
	vec = config["disconnectedTriggerLayers"].as<std::vector<int>>();
  }
  for(auto& x : vec)
	std::cout << x << std::endl;
  std::exit(0);


 
  ROOT::EnableImplicitMT();
  ROOT::RDataFrame df(tn, inf);

  std::vector<std::string> genvars = {
	"genpart_pid", "genpart_exphi", "genpart_exeta", "genpart_energy"
  };
  std::string condgen = "genpart_gen != -1 && genpart_reachedEE == 2 && genpart_pid == 22 && genpart_exeta > 0";
  auto dd = df.Define("good_gens", condgen);
  for(auto& v : genvars)
	dd = dd.Define("good_" + v, v + "[good_gens]");

  std::vector<std::string> tcvars = {
	"tc_energy", "tc_mipPt", "tc_pt", "tc_layer",
	"tc_x", "tc_y", "tc_z", "tc_phi", "tc_eta",
  };
  std::string condtc = "tc_zside == 1 && tc_layer%2 == 0";
  dd = dd.Define("good_tcs", condtc);
  for(auto& v : tcvars)
	dd = dd.Define("good_" + v, v + "[good_tcs]");

  std::vector<std::string> clvars = {
	"cl3d_energy", "cl3d_pt", "cl3d_eta", "cl3d_phi"
  };
  std::string condcl = "cl3d_eta > 0"; //dummy selection
  dd = dd.Define("good_cl", condcl);
  for(auto& v : clvars)
	dd = dd.Define("good_" + v, v + "[good_cl]");

  std::vector<std::string> allvars;
  allvars.insert(allvars.end(), genvars.begin(), genvars.end());
  allvars.insert(allvars.end(), tcvars.begin(),  tcvars.end());
  allvars.insert(allvars.end(), clvars.begin(),  clvars.end());
  std::vector<std::string> good_allvars = {"event"};
  for(auto& v : allvars)
	good_allvars.push_back("good_" + v);
  dd.Snapshot(tn, outf, good_allvars);
}
