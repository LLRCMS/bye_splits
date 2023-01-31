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

ROOT::RDF::RResultPtr<long long unsigned> addProgressBar(ROOT::RDF::RNode df) {
  auto c = df.Count();
  c.OnPartialResult(/*every=*/100,
					[] (long long unsigned e) { cout << "Progress: " << e << endl; });
  return c;
}

void skim(string tn, string inf, string outf, string particle, int nevents) {

  // read input parameters
  YAML::Node config = YAML::LoadFile("bye_splits/production/prod_params.yaml");
  vector<int> discLayers;
  if (config["selection"]["disconnectedTriggerLayers"]) {
	discLayers = config["selection"]["disconnectedTriggerLayers"].as<vector<int>>();
  }
  string reachedEE="", deltarThreshold="", mipThreshold="";
  if (config["selection"]["reachedEE"])
	reachedEE = config["selection"]["reachedEE"].as<string>();
  if (config["selection"]["deltarThreshold"])
	deltarThreshold = config["selection"]["deltarThreshold"].as<string>();
  if (config["selection"]["mipThreshold"])
	mipThreshold = config["selection"]["mipThreshold"].as<string>();
  // variables
  string vtmp = "tmp_good";
  
  if(nevents==-1) { //RDataFrame.Range() does not work with multithreading
	ROOT::EnableImplicitMT();
	cout << "Multithreaded..." << endl;
  }
  ROOT::RDataFrame dataframe(tn, inf);
  
  // gen-related variables
  vector<string> gen_intv = {"genpart_pid"};
  vector<string> gen_floatv = {"genpart_exphi", "genpart_exeta", "genpart_energy"};
  vector<string> gen_floatv2 = {"genpart_posx", "genpart_posy", "genpart_posz"};
  vector<string> gen_v = join_vars(gen_intv, gen_floatv, gen_floatv2);

  // selection on generated particles (within each event)
  unordered_map<string,string> pmap = {
	{"photons", "22"},
	{"electrons", "11"},
	{"pions", "211"}};
  string condgen = "genpart_gen != -1 && ";
  condgen += "genpart_reachedEE == " + reachedEE;
  condgen += " && genpart_pid == abs(" + pmap[particle] + ")";
  condgen += " && genpart_exeta > 0";

  auto df = dataframe.Define(vtmp + "_gens", condgen);
  for(auto& v : gen_v) { df = df.Define(vtmp + "_" + v, v + "[" + vtmp + "_gens]"); }

  //remove events with zero generated particles
  auto dfilt = df.Filter(vtmp + "_genpart_pid.size()!=0");

  // trigger cells-related variables
  vector<string> tc_uintv = {"tc_cluster_id"};
  vector<string> tc_intv = {"tc_layer", "tc_cellu", "tc_cellv", "tc_waferu", "tc_waferv"};
  vector<string> tc_floatv = {"tc_energy", "tc_mipPt", "tc_pt", 
							  "tc_x", "tc_y", "tc_z", "tc_phi", "tc_eta"};

  // selection on trigger cells (within each event)
  vector<string> tc_v = join_vars(tc_uintv, tc_intv, tc_floatv);
  string condtc = "tc_zside == 1 && tc_mipPt > " + mipThreshold + " && tc_layer <= 28";
  auto dd1 = dfilt.Define(vtmp + "_tcs", condtc);
  for(auto& v : tc_v)
	dd1 = dd1.Define(vtmp + "_" + v, v + "[" + vtmp + "_tcs]");

  // // module sums-related variables
  // vector<string> tsum_intv = {"ts_layer", "ts_waferu", "ts_waferv"};
  // vector<string> tsum_floatv = {"ts_energy", "ts_mipPt", "ts_pt"};
  // vector<string> tsum_v = join_vars(tsum_intv, tsum_floatv);
  
  // // selection on module trigger sums (within each event)
  // string condtsum = "ts_zside == 1 && ts_mipPt > " + mipThreshold;
  // dd1 = dd1.Define(vtmp + "_tsum", condtsum);
  // for(auto& v : tsum_v)
  // 	dd1 = dd1.Define(vtmp + "_" + v, v + "[" + vtmp + "_tsum]");

  // cluster-related variables
  vector<string> cl_uintv = {"cl3d_id"};
  vector<string> cl_floatv = {"cl3d_energy", "cl3d_pt", "cl3d_eta", "cl3d_phi"};
  vector<string> cl_v = join_vars(cl_uintv, cl_floatv);

  // selection on clusters (within each event)
  string condcl = "cl3d_eta > 0";
  dd1 = dd1.Define(vtmp + "_cl", condcl);
  for(auto& v : cl_v)
	dd1 = dd1.Define(vtmp + "_" + v, v + "[" + vtmp + "_cl]");

  //remove events with zero clusters
  auto dfilt2 = dd1.Filter(vtmp + "_cl3d_id.size()!=0");

  // matching
  vector<string> matchvars = {"deltaR", "matches"};
  string cond_deltaR = matchvars[0] + " <= " + deltarThreshold;
  auto dd2 = dfilt2.Define(matchvars[0], calcDeltaR, {vtmp + "_genpart_exeta", vtmp + "_genpart_exphi",
													  vtmp + "_cl3d_eta", vtmp + "_cl3d_phi"})
	.Define(matchvars[1], cond_deltaR);

  // convert root vector types to vector equivalents (uproot friendly)
  // vector<string> intv = join_vars(gen_intv, tc_intv, tsum_intv);
  vector<string> intv = join_vars(gen_intv, tc_intv);
  for(auto& var : intv) {
	dd2 = dd2.Define("good_" + var,
					 [](const ROOT::VecOps::RVec<int> &v) {
					   return vector<int>(v.begin(), v.end());
					 },
					 {vtmp + "_" + var});
  }
  vector<string> uintv = join_vars(cl_uintv, tc_uintv);
  for(auto& var : uintv) {
    dd2 = dd2.Define("good_" + var,
					 [](const ROOT::VecOps::RVec<unsigned> &v) {
					   return vector<unsigned>(v.begin(), v.end());
					 },
					 {vtmp + "_" + var});
  }
  //vector<string> floatv = join_vars(gen_floatv, tc_floatv, cl_floatv, tsum_floatv);
  vector<string> floatv = join_vars(gen_floatv, tc_floatv, cl_floatv);
  for(auto& var : floatv) {
	dd2 = dd2.Define("good_" + var,
					 [](const ROOT::VecOps::RVec<float> &v) {
					   return vector<float>(v.begin(), v.end());
					 },
					 {vtmp + "_" + var});
  }
  vector<string> floatv2 = join_vars(gen_floatv2);
  for(auto& var : floatv2) {
	dd2 = dd2.Define("good_" + var,
					 [](const ROOT::VecOps::RVec<vector<float>> &v) {
					   vector<vector<float>> vec(v.size());
					   for(unsigned i=0; i<v.size(); ++i) {
						 vec[i] = vector<float>(v[i].begin(), v[i].end());
					   }
					   return vec;
					 },
					 {vtmp + "_" + var});
  }
  
  // define stored variables (and rename some)
  // vector<string> allvars = join_vars(gen_v, tc_v, cl_v, tsum_v);
  vector<string> allvars = join_vars(gen_v, tc_v, cl_v);
  vector<string> good_allvars = {"event"};
  good_allvars.insert(good_allvars.end(), matchvars.begin(), matchvars.end());
  for(auto& v : allvars)
	good_allvars.push_back("good_" + v);

  // store skimmed file
  if(nevents>0) {
	dd2.Range(0, nevents).Snapshot(tn, outf, good_allvars);
  }
  else {
	dd2.Snapshot(tn, outf, good_allvars);
  }
  
  // display event processing progress
  ROOT::RDF::RResultPtr<long long unsigned> count = addProgressBar(ROOT::RDF::RNode(dd2));
  count.GetValue();
}
