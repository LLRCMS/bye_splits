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
					[] (long long unsigned e) { std::cout << "Progress: " << e << "\t\r" << std::endl; });
  return c;
}

void skim(std::string tn, std::string inf, std::string outf, std::string particle) {

  // read input parameters
  YAML::Node config = YAML::LoadFile("bye_splits/production/prod_params.yaml");
  std::vector<int> discLayers;
  if (config["disconnectedTriggerLayers"]) {
	discLayers = config["disconnectedTriggerLayers"].as<std::vector<int>>();
  }
  std::string reachedEE="", deltarThreshold="", mipThreshold="";
  if (config["reachedEE"]) reachedEE = config["reachedEE"].as<std::string>();
  if (config["deltarThreshold"]) deltarThreshold = config["deltarThreshold"].as<std::string>();
  if (config["mipThreshold"]) mipThreshold = config["mipThreshold"].as<std::string>();

  // variables
  std::string vtmp = "tmp_good";
  
  //ROOT::EnableImplicitMT();
  ROOT::RDataFrame dataframe(tn, inf);
  auto df = dataframe.Range(0, 500);
  
  // gen-related variables
  std::vector<std::string> gen_intv = {"genpart_pid"};
  std::vector<std::string> gen_floatv = {"genpart_exphi", "genpart_exeta", "genpart_energy"};
  std::vector<std::string> gen_v = join_vars(gen_intv, gen_floatv);

  // selection on generated particles (within each event)
  std::unordered_map<std::string,std::string> pmap = {{"photons", "22"}, {"electrons", "11"}};
  std::string condgen = "genpart_gen != -1 && ";
  condgen += "genpart_reachedEE == " + reachedEE;
  condgen += " && genpart_pid == " + pmap[particle];
  condgen += " && genpart_exeta > 0";

  df = df.Define(vtmp + "_gens", condgen);
  for(auto& v : gen_v) { df = df.Define(vtmp + "_" + v, v + "[" + vtmp + "_gens]"); }

  //remove events with zero generated particles
  auto dfilt = df.Filter(vtmp + "_genpart_pid.size()!=0");

  // trigger cells-related variables
  std::vector<std::string> tc_uintv = {"tc_cluster_id"};
  std::vector<std::string> tc_intv = {"tc_layer", "tc_cellu", "tc_cellv", "tc_waferu", "tc_waferv"};
  std::vector<std::string> tc_floatv = {"tc_energy", "tc_mipPt", "tc_pt", 
							  "tc_x", "tc_y", "tc_z", "tc_phi", "tc_eta"};

  // selection on trigger cells (within each event)
  std::vector<std::string> tc_v = join_vars(tc_uintv, tc_intv, tc_floatv);
  std::string condtc = "tc_zside == 1 && tc_mipPt > " + mipThreshold;// && tc_layer%2 == 0";
  auto dd1 = dfilt.Define(vtmp + "_tcs", condtc);
  for(auto& v : tc_v)
	dd1 = dd1.Define(vtmp + "_" + v, v + "[" + vtmp + "_tcs]");

  // module sums-related variables
  std::vector<std::string> tsum_intv = {"ts_layer", "ts_waferu", "ts_waferv"};
  std::vector<std::string> tsum_floatv = {"ts_energy", "ts_mipPt", "ts_pt"};
  std::vector<std::string> tsum_v = join_vars(tsum_intv, tsum_floatv);
  
  // selection on module trigger sums (within each event)
  std::string condtsum = "ts_zside == 1 && ts_mipPt > " + mipThreshold;
  dd1 = dd1.Define(vtmp + "_tsum", condtsum);
  for(auto& v : tsum_v)
	dd1 = dd1.Define(vtmp + "_" + v, v + "[" + vtmp + "_tsum]");

  // cluster-related variables
  std::vector<std::string> cl_uintv = {"cl3d_id"};
  std::vector<std::string> cl_floatv = {"cl3d_energy", "cl3d_pt", "cl3d_eta", "cl3d_phi"};
  std::vector<std::string> cl_v = join_vars(cl_uintv, cl_floatv);

  // selection on clusters (within each event)
  std::string condcl = "cl3d_eta > 0";
  dd1 = dd1.Define(vtmp + "_cl", condcl);
  for(auto& v : cl_v)
	dd1 = dd1.Define(vtmp + "_" + v, v + "[" + vtmp + "_cl]");

  //remove events with zero clusters
  auto dfilt2 = dd1.Filter(vtmp + "_cl3d_id.size()!=0");

  // matching
  std::vector<std::string> matchvars = {"deltaR", "matches"};
  std::string cond_deltaR = matchvars[0] + " <= " + deltarThreshold;
  auto dd2 = dfilt2.Define(matchvars[0], calcDeltaR, {vtmp + "_genpart_exeta", vtmp + "_genpart_exphi",
													  vtmp + "_cl3d_eta", vtmp + "_cl3d_phi"})
	.Define(matchvars[1], cond_deltaR);

  // convert root std::vector types to std::vector equivalents (uproot friendly)
  std::vector<std::string> intv = join_vars(gen_intv, tc_intv, tsum_intv);
  for(auto& var : intv) {
	dd2 = dd2.Define("good_" + var,
					 [](const ROOT::VecOps::RVec<int> &v) {
					   return std::vector<int>(v.begin(), v.end());
					 },
					 {vtmp + "_" + var});
  }
  std::vector<std::string> uintv = join_vars(cl_uintv, tc_uintv);
  for(auto& var : uintv) {
    dd2 = dd2.Define("good_" + var,
					 [](const ROOT::VecOps::RVec<unsigned> &v) {
					   return std::vector<unsigned>(v.begin(), v.end());
					 },
					 {vtmp + "_" + var});
  }
  std::vector<std::string> floatv = join_vars(gen_floatv, tc_floatv, cl_floatv, tsum_floatv);
  for(auto& var : floatv) {
	dd2 = dd2.Define("good_" + var,
					 [](const ROOT::VecOps::RVec<float> &v) {
					   return std::vector<float>(v.begin(), v.end());
					 },
					 {vtmp + "_" + var});
  }
  
  // define stored variables (and rename some)
  std::vector<std::string> allvars = join_vars(gen_v, tc_v, cl_v, tsum_v);
  std::vector<std::string> good_allvars = {"event"};
  good_allvars.insert(good_allvars.end(), matchvars.begin(), matchvars.end());
  for(auto& v : allvars)
	good_allvars.push_back("good_" + v);

  // store skimmed file
  dd2.Snapshot(tn, outf, good_allvars);
  ROOT::RDF::RResultPtr<long long unsigned> count = addProgressBar(ROOT::RDF::RNode(dd2));
  count.GetValue();

}
