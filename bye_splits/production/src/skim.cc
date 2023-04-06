#include "include/skim.h"

ROOT::VecOps::RVec<float> calcDeltaR(ROOT::VecOps::RVec<float> geta, ROOT::VecOps::RVec<float> gphi,
                                     ROOT::VecOps::RVec<float> cleta, ROOT::VecOps::RVec<float> clphi)
{
  if (geta.size() == 0) // empty event (filtered before)
    return ROOT::VecOps::RVec<float>();

  assert(geta.size() == 1); // consistency check
  unsigned ncl = cleta.size();
  ROOT::VecOps::RVec<float> deltaRsq(ncl);
  float deta, dphi;

  for (unsigned j = 0; j < ncl; ++j)
  {
    deta = fabs(cleta[j] - geta[0]);
    dphi = fabs(clphi[j] - gphi[0]);
    if (dphi > M_PI)
    {
      dphi -= (2 * M_PI);
    }
    deltaRsq[j] = dphi * dphi + deta * deta;
  }

  return deltaRsq;
}

ROOT::VecOps::RVec<float> calcDeltaRxy(ROOT::VecOps::RVec<float> geta, ROOT::VecOps::RVec<float> gphi,
                                       ROOT::VecOps::RVec<float> tcx, ROOT::VecOps::RVec<float> tcy, ROOT::VecOps::RVec<float> tcz)
{
  if (geta.size() == 0) // empty event (filtered before)
    return ROOT::VecOps::RVec<float>();

  assert(geta.size() == 1); // consistency check

  // Calculate gen (x/z0, y/z0) from (geta, gphi) where z0 is the z-coordinate of the first layer of the HGCAL
  float gen_theta = 2 * atan(exp(-geta[0]));
  float gen_x_over_z = cos(gphi[0])*tan(gen_theta);
  float gen_y_over_z = sin(gphi[0])*tan(gen_theta);

  unsigned ntc = tcx.size();
  ROOT::VecOps::RVec<float> deltaRsq(ntc);
  float dx, dy;

  for (unsigned j = 0; j < ntc; ++j)
  {
    float tc_x_over_z = tcx[j] / tcz[j];
    float tc_y_over_z = tcy[j] / tcz[j];

    dx = fabs(tc_x_over_z - gen_x_over_z);
    dy = fabs(tc_y_over_z - gen_y_over_z);

    deltaRsq[j] = dx * dx + dy * dy;
  }

  return deltaRsq;
}

ROOT::RDF::RResultPtr<long long unsigned> addProgressBar(ROOT::RDF::RNode df)
{
  auto c = df.Count();
  c.OnPartialResult(/*every=*/100,
                    [](long long unsigned e)
                    { std::cout << "Progress: " << e << std::endl; });
  return c;
}

void skim(std::string tn, std::string inf, std::string outf, std::string particle, int nevents)
{

  // read input parameters
  YAML::Node config = YAML::LoadFile("config.yaml");
  std::vector<int> discLayers;
  if (config["selection"]["disconnectedTriggerLayers"])
  {
    discLayers = config["selection"]["disconnectedTriggerLayers"].as<std::vector<int>>();
  }
  std::string reachedEE = "", mipThreshold = "";
  if (config["selection"]["reachedEE"])
    reachedEE = config["selection"]["reachedEE"].as<std::string>();
  if (config["selection"]["mipThreshold"])
    mipThreshold = config["selection"]["mipThreshold"].as<std::string>();

  float tcDeltaRThresh = 0.0, deltarThreshold = 0.0;
  if (config["selection"]["deltarThreshold"])
    deltarThreshold = config["selection"]["deltarThreshold"].as<float>();
  if (config["skim"]["tcDeltaRThresh"])
  {
    tcDeltaRThresh = config["skim"]["tcDeltaRThresh"].as<float>();
  }
  // variables
  std::string vtmp = "tmp_good";

  if (nevents == -1)
  { // RDataFrame.Range() does not work with multithreading
    ROOT::EnableImplicitMT();
    std::cout << "Multithreaded..." << std::endl;
  }
  ROOT::RDataFrame dataframe(tn, inf);

  // gen-related variables
  std::vector<std::string> gen_intv = {"genpart_pid"};
  std::vector<std::string> gen_floatv = {"genpart_exphi", "genpart_exeta", "genpart_energy"};
  // vector<std::string> gen_floatv2 = {"genpart_posx", "genpart_posy", "genpart_posz"};
  // vector<std::string> gen_v = join_vars(gen_intv, gen_floatv, gen_floatv2);
  std::vector<std::string> gen_v = join_vars(gen_intv, gen_floatv);

  // selection on generated particles (within each event)
  std::unordered_map<std::string, std::string>
      pmap = {
          {"photons", "22"},
          {"electrons", "11"},
          {"pions", "211"}};
  std::string condgen = "genpart_gen != -1 && ";
  condgen += "genpart_reachedEE == " + reachedEE;
  condgen += " && genpart_pid == abs(" + pmap[particle] + ")";
  condgen += " && genpart_exeta > 0";

  auto df = dataframe.Define(vtmp + "_gens", condgen);
  for (auto &v : gen_v)
  {
    df = df.Define(vtmp + "_" + v, v + "[" + vtmp + "_gens]");
  }

  // remove events with zero generated particles
  auto dfilt = df.Filter(vtmp + "_genpart_pid.size()!=0");

  // trigger cells-related variables
  std::vector<std::string> tc_uintv = {"tc_multicluster_id"};
  std::vector<std::string> tc_intv = {"tc_layer", "tc_cellu", "tc_cellv", "tc_waferu", "tc_waferv"};
  std::vector<std::string> tc_floatv = {"tc_energy", "tc_mipPt", "tc_pt", "tc_x", "tc_y", "tc_z", "tc_phi", "tc_eta"};

  // selection on trigger cells (within each event)
  std::vector<std::string> tc_v = join_vars(tc_uintv, tc_intv, tc_floatv);

  auto dd1 = dfilt.Define("tc_deltaR", calcDeltaRxy, {vtmp + "_genpart_exeta", vtmp + "_genpart_exphi", "tc_x", "tc_y", "tc_z"});
  std::string condtc = "tc_zside == 1 && tc_mipPt > " + mipThreshold + " && tc_layer <= 28 && tc_deltaR <= " + std::to_string(pow(tcDeltaRThresh, 2)); // Comparing dR^2 to avoid calculating sqrt
  dd1 = dd1.Define(vtmp + "_tcs", condtc);
  for (auto &v : tc_v)
    dd1 = dd1.Define(vtmp + "_" + v, v + "[" + vtmp + "_tcs]");

  // // module sums-related variables
  // vector<std::string> tsum_intv = {"ts_layer", "ts_waferu", "ts_waferv"};
  // vector<std::string> tsum_floatv = {"ts_energy", "ts_mipPt", "ts_pt"};
  // vector<std::string> tsum_v = join_vars(tsum_intv, tsum_floatv);

  // // selection on module trigger sums (within each event)
  // std::string condtsum = "ts_zside == 1 && ts_mipPt > " + mipThreshold;
  // dd1 = dd1.Define(vtmp + "_tsum", condtsum);
  // for(auto& v : tsum_v)
  // 	dd1 = dd1.Define(vtmp + "_" + v, v + "[" + vtmp + "_tsum]");

  // cluster-related variables
  std::vector<std::string> cl_uintv = {"cl3d_id"};
  std::vector<std::string> cl_floatv = {"cl3d_energy", "cl3d_pt", "cl3d_eta", "cl3d_phi"};
  std::vector<std::string> cl_v = join_vars(cl_uintv, cl_floatv);

  // selection on clusters (within each event)
  std::string condcl = "cl3d_eta > 0";
  dd1 = dd1.Define(vtmp + "_cl", condcl);
  for (auto &v : cl_v)
    dd1 = dd1.Define(vtmp + "_" + v, v + "[" + vtmp + "_cl]");

  // remove events with zero clusters
  auto dfilt2 = dd1.Filter(vtmp + "_cl3d_id.size()!=0");

  // matching
  std::vector<std::string> matchvars = {"deltaR", "matches"};
  std::string cond_deltaR = matchvars[0] + " <= " + std::to_string(pow(deltarThreshold, 2)); // Comparing dR^2 to avoid calculating sqrt
  auto dd2 = dfilt2.Define(matchvars[0], calcDeltaR, {vtmp + "_genpart_exeta", vtmp + "_genpart_exphi", vtmp + "_cl3d_eta", vtmp + "_cl3d_phi"})
                 .Define(matchvars[1], cond_deltaR);

  // convert root vector types to vector equivalents (uproot friendly)
  // vector<std::string> intv = join_vars(gen_intv, tc_intv, tsum_intv);
  std::vector<std::string> intv = join_vars(gen_intv, tc_intv);
  for (auto &var : intv)
  {
    dd2 = dd2.Define("good_" + var,
                     [](const ROOT::VecOps::RVec<int> &v)
                     {
                       return std::vector<int>(v.begin(), v.end());
                     },
                     {vtmp + "_" + var});
  }
  std::vector<std::string> uintv = join_vars(cl_uintv, tc_uintv);
  for (auto &var : uintv)
  {
    dd2 = dd2.Define("good_" + var,
                     [](const ROOT::VecOps::RVec<unsigned> &v)
                     {
                       return std::vector<unsigned>(v.begin(), v.end());
                     },
                     {vtmp + "_" + var});
  }
  // vector<std::string> floatv = join_vars(gen_floatv, tc_floatv, cl_floatv, tsum_floatv);
  std::vector<std::string> floatv = join_vars(gen_floatv, tc_floatv, cl_floatv);
  for (auto &var : floatv)
  {
    dd2 = dd2.Define("good_" + var,
                     [](const ROOT::VecOps::RVec<float> &v)
                     {
                       return std::vector<float>(v.begin(), v.end());
                     },
                     {vtmp + "_" + var});
  }
  /*
  vector<std::string> floatv2 = join_vars(gen_floatv2);
  for (auto &var : floatv2)
  {
    dd2 = dd2.Define("good_" + var,
                     [](const ROOT::VecOps::RVec<vector<float>> &v)
                     {
                       vector<vector<float>> vec(v.size());
                       for (unsigned i = 0; i < v.size(); ++i)
                       {
                         vec[i] = vector<float>(v[i].begin(), v[i].end());
                       }
                       return vec;
                     },
                     {vtmp + "_" + var});
  }
  */

  // define stored variables (and rename some)
  // vector<std::string> allvars = join_vars(gen_v, tc_v, cl_v, tsum_v);
  std::vector<std::string> allvars = join_vars(gen_v, tc_v, cl_v);
  std::vector<std::string> good_allvars = {"event"};
  good_allvars.insert(good_allvars.end(), matchvars.begin(), matchvars.end());
  for (auto &v : allvars)
    good_allvars.push_back("good_" + v);

  // store skimmed file
  if (nevents > 0)
  {
    dd2.Range(0, nevents).Snapshot(tn, outf, good_allvars);
  }
  else
  {
    dd2.Snapshot(tn, outf, good_allvars);
  }

  // display event processing progress
  ROOT::RDF::RResultPtr<long long unsigned> count = addProgressBar(ROOT::RDF::RNode(dd2));
  count.GetValue();
}
