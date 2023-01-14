#include "include/skim.h"

ROOT::VecOps::RVec<float> calcDeltaR(ROOT::VecOps::RVec<float> geta, ROOT::VecOps::RVec<float> gphi,
									 ROOT::VecOps::RVec<float> cleta,  ROOT::VecOps::RVec<float> clphi)
  
{
  if(geta.size()==0) // empty event (filtered before)
	return ROOT::VecOps::RVec<float>();

  assert(geta.size() == 1); //consistency check
  unsigned ncl = cleta.size();
  ROOT::VecOps::RVec<float> deltaR(ncl);
  float deta, dphi;
  
  for(unsigned j=0; j<ncl; ++j) {
	deta = fabs(cleta[j] - geta[0]);
	dphi = fabs(clphi[j] - gphi[0]);
	if(dphi > M_PI) dphi -= (2 * M_PI);
	deltaR[j] = sqrtf(dphi*dphi + deta*deta);
  }

  return deltaR;
}

void skim(string tn, string inf, string outf, string particle) {

  YAML::Node config = YAML::LoadFile("bye_splits/production/prod_params.yaml");
  vector<int> discLayers;
  if (config["disconnectedTriggerLayers"]) {
	discLayers = config["disconnectedTriggerLayers"].as<vector<int>>();
  }
  string reachedEE="", deltarThreshold="", mipThreshold="";
  if (config["reachedEE"]) reachedEE = config["reachedEE"].as<string>();
  if (config["deltarThreshold"]) deltarThreshold = config["deltarThreshold"].as<string>();
  if (config["mipThreshold"]) mipThreshold = config["mipThreshold"].as<string>();

  // variables
  string vtmp = "tmp_good";
  
  //ROOT::EnableImplicitMT();
  ROOT::RDataFrame df(tn, inf);
  auto dd = df.Range(0, 100);
  
  vector<string> genvars_int = {"genpart_pid"};
  vector<string> genvars_float = {"genpart_exphi", "genpart_exeta", "genpart_energy"};
  vector<string> genvars = join_vars(genvars_int, genvars_float);
  
  unordered_map<string,string> pmap = {{"photons", "22"},
									   {"electrons", "11"}};
  string condgen = "genpart_gen != -1 && ";
  condgen += "genpart_reachedEE == " + reachedEE;
  condgen += " && genpart_pid == " + pmap[particle];
  condgen += " && genpart_exeta > 0";

  dd = dd.Define(vtmp + "_gens", condgen);
  for(auto& v : genvars)
   	dd = dd.Define(vtmp + "_" + v, v + "[" + vtmp + "_gens]");

  vector<string> tcvars_int = {"tc_layer", "tc_cellu", "tc_cellv", "tc_waferu", "tc_waferv"};
  vector<string> tcvars_float = {"tc_energy", "tc_mipPt", "tc_pt", 
								 "tc_x", "tc_y", "tc_z", "tc_phi", "tc_eta"};
  vector<string> tcvars = join_vars(tcvars_int, tcvars_float);

  string condtc = "tc_zside == 1 && tc_mipPt > " + mipThreshold;// && tc_layer%2 == 0";
  dd = dd.Define(vtmp + "_tcs", condtc);
  for(auto& v : tcvars)
	dd = dd.Define(vtmp + "_" + v, v + "[" + vtmp + "_tcs]");

  vector<string> clvars_uint = {"cl3d_id"};
  vector<string> clvars_float = {"cl3d_energy", "cl3d_pt", "cl3d_eta", "cl3d_phi"};
  vector<string> clvars = join_vars(clvars_uint, clvars_float);
	
  string condcl = "cl3d_eta > 0";
  dd = dd.Define(vtmp + "_cl", condcl);
  for(auto& v : clvars)
	dd = dd.Define(vtmp + "_" + v, v + "[" + vtmp + "_cl]");

  // matching
  vector<string> matchvars = {"deltaR", "matches"};
  string cond_deltaR = matchvars[0] + " <= " + deltarThreshold;
  dd = dd.Define(matchvars[0], calcDeltaR, {vtmp + "_genpart_exeta", vtmp + "_genpart_exphi",
											vtmp + "_cl3d_eta", vtmp + "_cl3d_phi"})
	.Define(matchvars[1], cond_deltaR);

  // convert root vector types to vector equivalents (uproot friendly)
  vector<string> intvars = join_vars(genvars_int, tcvars_int);
  for(auto& var : intvars) {
	dd = dd.Define("good_" + var,
				   [](const ROOT::VecOps::RVec<int> &v) {
					 return vector<int>(v.begin(), v.end());
				   },
				   {vtmp + "_" + var});
  }
  vector<string> uintvars = join_vars(clvars_uint);
  for(auto& var : uintvars) {
    dd = dd.Define("good_" + var,
		   [](const ROOT::VecOps::RVec<unsigned> &v) {
		     return vector<unsigned>(v.begin(), v.end());
		   },
		   {vtmp + "_" + var});
  }
  vector<string> floatvars = join_vars(genvars_float, tcvars_float, clvars_float);
  for(auto& var : floatvars) {
	dd = dd.Define("good_" + var,
				   [](const ROOT::VecOps::RVec<float> &v) {
					 return vector<float>(v.begin(), v.end());
				   },
				   {vtmp + "_" + var});
  }
  
  // define stored variables (and rename some)
  vector<string> allvars = join_vars(genvars, tcvars, clvars);
  vector<string> good_allvars = {"event"};
  good_allvars.insert(good_allvars.end(), matchvars.begin(), matchvars.end());
  for(auto& v : allvars)
	good_allvars.push_back("good_" + v);

  // store skimmed file
  dd.Snapshot(tn, outf, good_allvars);
}
