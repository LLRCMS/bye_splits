#ifndef _SKIM_H_
#define _SKIM_H_

#include <math.h> // includes M_PI
#include <cmath> 
#include <cassert> // includes assert
#include <vector>
#include <unordered_map>

#include <TROOT.h>
#include <ROOT/RDataFrame.hxx>

#include "yaml-cpp/yaml.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;

void skim(string, string, string, string, int);

template <class U>
auto internal_join_vars(vector<U>& dest, const vector<U>& vec) -> void {
  dest.insert(dest.end(), vec.begin(), vec.end());
}
template <class U, class ... RestArgs>
auto internal_join_vars(vector<U>& dest, const vector<U>& vec, RestArgs... args) -> void
{
  dest.insert(dest.end(), vec.begin(), vec.end());
  internal_join_vars(dest, args...);
}  
// required for compatibility issues between uproot and RDataFrame
template <class ... Ts>
auto join_vars(const vector<string>& val, Ts... args) -> vector<string>
{
  vector<string> tmp;
  internal_join_vars<std::string>(tmp, val, args...);
  return tmp;
}

#endif /* _SKIM_H_ */
