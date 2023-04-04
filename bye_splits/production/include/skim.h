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

void skim(std::string, std::string, std::string, std::string, int);

template <class U>
auto internal_join_vars(std::vector<U>& dest, const std::vector<U>& vec) -> void {
  dest.insert(dest.end(), vec.begin(), vec.end());
}
template <class U, class ... RestArgs>
auto internal_join_vars(std::vector<U>& dest, const std::vector<U>& vec, RestArgs... args) -> void
{
  dest.insert(dest.end(), vec.begin(), vec.end());
  internal_join_vars(dest, args...);
}  
// required for compatibility issues between uproot and RDataFrame
template <class ... Ts>
auto join_vars(const std::vector<std::string>& val, Ts... args) -> std::vector<std::string>
{
  std::vector<std::string> tmp;
  internal_join_vars<std::string>(tmp, val, args...);
  return tmp;
}

#endif /* _SKIM_H_ */
