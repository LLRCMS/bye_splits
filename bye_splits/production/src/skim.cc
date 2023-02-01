#include "include/skim.h"

void skim(string tn, string inf, string outf, string particle) {

  std::cout << "\nTree Name: " << tn << "\n\n";
  std::cout << "File Name: " << inf << "\n\n";
  
  YAML::Node config = YAML::LoadFile("bye_splits/production/prod_params.yaml");
  vector<int> vec;
  if (config["disconnectedTriggerLayers"]) {
	vec = config["disconnectedTriggerLayers"].as<std::vector<int>>();
  }
  string reachedEE = "x";
  if (config["reachedEE"]) {
	reachedEE = config["reachedEE"].as<string>();
  }

  std::vector<string> genvars = {
	"event", "genpart_pid", "genpart_exphi", "genpart_exeta", "genpart_energy", "genpart_pt"
  };

  ROOT::EnableImplicitMT();
  ROOT::RDataFrame df(tn, inf, genvars);

  //df.Define("good_gen", "genpart_gen[genpart_gen != -1]"); // THIS WORKS!!

  unordered_map<string,string> pmap = {{"photon", "22"}, {"electron", "11"}};

  string condgen = "genpart_gen != -1 && ";
  condgen += "genpart_reachedEE == " + reachedEE;
  condgen += " && genpart_pid == " + pmap[particle];
  condgen += " && genpart_exeta > 0";

  auto dd = df.Define("good_gens", condgen);
  
  // This produces an error: the second expression needs to be a C++ function, written as a string, to be used with RDataFrame's .Define() method.
  for(auto& v : genvars)
  {
	  dd = dd.Define("good_" + v, v + "[good_gens]");
  }
  
  vector<string> tcvars = {
	"tc_energy", "tc_mipPt", "tc_pt", "tc_layer",
	"tc_x", "tc_y", "tc_z", "tc_phi", "tc_eta",
	"tc_cellu", "tc_cellv", "tc_waferu", "tc_waferv",
  };
  string condtc = "tc_zside == 1 && tc_layer%2 == 0";
  dd = dd.Define("good_tcs", condtc);
  for(auto& v : tcvars)
  {
    dd = dd.Define("good_" + v, v + "[good_tcs]");
  }

  vector<string> clvars = {
	"cl3d_energy", "cl3d_pt", "cl3d_eta", "cl3d_phi"
  };
  string condcl = "cl3d_eta > 0"; //dummy selection
  dd = dd.Define("good_cl", condcl);
  for(auto& v : clvars)
  {
    dd = dd.Define("good_" + v, v + "[good_cl]");
  }

  vector<string> allvars;
  allvars.insert(allvars.end(), genvars.begin(), genvars.end());
  allvars.insert(allvars.end(), tcvars.begin(),  tcvars.end());
  allvars.insert(allvars.end(), clvars.begin(),  clvars.end());
  vector<string> good_allvars = {"event"};
  for(auto& v : allvars)
  {
    good_allvars.push_back("good_" + v);
  }
  dd.Snapshot(tn, outf, good_allvars);
  
}
