#include <iostream>
#include "include/skim.h"

#include <stdio.h>  // for printf()
#include <stdlib.h> // for strtol()
#include <errno.h>  // for errno
#include <limits.h> // for INT_MIN and INT_MAX
#include <string.h>  // for strlen

int convert_to_int(char** argv, int idx) {
  char* p;
  errno = 0; // not 'int errno', because the '#include' already defined it
  long arg = strtol(argv[idx], &p, 10);
  if (*p != '\0' || errno != 0) {
	return 1; // In main(), returning non-zero means failure
  }

  if (arg < INT_MIN || arg > INT_MAX) {
	return 1;
  }
  int arg_int = arg;

  // Everything went well, print it as a regular number plus a newline
  return arg_int;
}

void show_help(const po::options_description&, const std::string&);
po::variables_map process_program_options(int argc, char **argv);

void validate(po::variables_map args) {
  std::string particles = args["particles"].as<string>();
  if(!(particles=="photons" || particles=="electrons" || particles=="pions")) {
    throw po::validation_error(po::validation_error::invalid_option_value, "particles");
  }
}

void show_help(const po::options_description& desc,
			   const std::string& topic = "") {
  std::cout << desc << '\n';
  if (topic != "") {
	std::cout << "You asked for help on: " << topic << '\n';
  }
}

po::variables_map process_program_options(int argc, char **argv)
{
  po::options_description desc("Usage");
  desc.add_options()
	("help,h",
	 po::value<string>()->implicit_value("")
	 ->notifier([&desc](const std::string &topic) {show_help(desc, topic);}),
	 "Show help. If given, show help on the specified topic.")
	("nevents", po::value<int>()->default_value(-1),
	 "number of entries to consider, useful for debugging (-1 means all)")
	("particles", po::value<string>()->required(),
	 "type of particle");
  
  if (argc <= 1) {
	show_help(desc); // does not return
	exit( EXIT_SUCCESS );
  }

  po::variables_map args;
  try {
	po::store(po::parse_command_line(argc, argv, desc), args);
  }
  catch (po::error const& e) {
	std::cerr << e.what() << '\n';
	exit( EXIT_FAILURE );
  }
  po::notify(args);
  validate(args);
  return args;
}

//Run with ./produce.exe photons
int main(int argc, char **argv) {
  std::string dir = "/eos/user/b/bfontana/FPGAs/new_algos/";
  std::string tree_name = "FloatingpointMixedbcstcrealsig4DummyHistomaxxydr015GenmatchGenclustersntuple/HGCalTriggerNtuple";

  po::variables_map args = process_program_options(argc, argv);
  if (args.count("help")) {
    return 1;
  }

  string particles = args["particles"].as<string>();
  int nevents = args["nevents"].as<int>();

  std::string infile = particles + "_0PU_bc_stc_hadd.root";
  std::string events_str = nevents > 0 ? std::to_string(nevents) + "events_" : "";
  std::string outfile = "skim_" + events_str + infile;
  skim(tree_name, dir + infile, dir + outfile, particles, nevents);
  return 0;
}
