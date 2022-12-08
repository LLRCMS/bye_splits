#include <iostream>
#include "yaml-cpp/yaml.h"

using namespace std;

int main()
{
    YAML::Node config = YAML::LoadFile("/home/llr/cms/ehle/git/bye_splits/tests/params.yaml");
	if (config["Store"]) {
	  std::cout << "Testing YAML-CPP installation: " << config["Store"].as<std::string>() << "\n";
	}
    return 0;
}
