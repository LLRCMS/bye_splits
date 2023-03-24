#include <iostream>
#include <tuple>
#include "include/skim.h"

#include <stdio.h>  // for printf()
#include <stdlib.h> // for strtol()
#include <errno.h>  // for errno
#include <limits.h> // for INT_MIN and INT_MAX
#include <string.h> // for strlen

int convert_to_int(char **argv, int idx)
{
  char *p;
  errno = 0; // not 'int errno', because the '#include' already defined it
  long arg = strtol(argv[idx], &p, 10);
  if (*p != '\0' || errno != 0)
  {
    return 1; // In main(), returning non-zero means failure
  }

  if (arg < INT_MIN || arg > INT_MAX)
  {
    return 1;
  }
  int arg_int = arg;

  // Everything went well, print it as a regular number plus a newline
  return arg_int;
}

void show_help(const po::options_description &, const std::string &);
po::variables_map process_program_options(int argc, char **argv);

/*
void validate(po::variables_map args)
{
  std::string particles = args["particles"].as<string>();
  if (!(particles == "photons" || particles == "electrons" || particles == "pions"))
  {
    throw po::validation_error(po::validation_error::invalid_option_value, "particles");
  }
}
*/

void show_help(const po::options_description &desc,
               const std::string &topic = "")
{
  std::cout << desc << '\n';
  if (topic != "")
  {
    std::cout << "You asked for help on: " << topic << '\n';
  }
}

po::variables_map process_program_options(int argc, char **argv)
{
  po::options_description desc("Usage");
  desc.add_options()("help,h",
                     po::value<string>()->implicit_value("")->notifier([&desc](const std::string &topic)
                                                                       { show_help(desc, topic); }),
                     "Show help. If given, show help on the specified topic.")("nevents", po::value<int>()->default_value(-1),
                                                                               "number of entries to consider, useful for debugging (-1 means all)")("inpath", po::value<string>()->implicit_value(""),
                                                                                                                                                     "full path to ROOT file");
  /*
  if (argc <= 1)
  {
    show_help(desc); // does not return
    exit(EXIT_SUCCESS);
  }
  */

  po::variables_map args;
  try
  {
    po::store(po::parse_command_line(argc, argv, desc), args);
  }
  catch (po::error const &e)
  {
    std::cerr << e.what() << '\n';
    exit(EXIT_FAILURE);
  }
  po::notify(args);
  // validate(args);
  return args;
}

tuple<string, string, string, string> read_config(YAML::Node config)
{
  string root_dir = "", root_tree = "", input_path = "", out_dir = "", outfile_base_name = "";
  if (config["io"]["dir"])
  {
    root_dir = config["io"]["dir"].as<string>();
  }
  if (config["io"]["tree"])
  {
    root_tree = config["io"]["tree"].as<string>();
  }
  if (config["skim"]["infilePath"])
  {
    input_path = config["skim"]["infilePath"].as<string>();
  }
  if (config["skim"]["dir"])
  {
    out_dir = config["skim"]["dir"].as<string>();
  }
  if (config["skim"]["outfileBaseName"])
  {
    outfile_base_name = config["skim"]["outfileBaseName"].as<string>();
  }
  string tree_name = root_dir + "/" + root_tree;
  return make_tuple(tree_name, input_path, out_dir, outfile_base_name);
}

string get_particles(string inpath)
{
  std::string phot_str = "photon";
  std::string el_str = "electron";
  std::string pi_str = "pion";
  std::string particles;
  if (inpath.find(phot_str) != std::string::npos)
  {
    particles = "photons";
  }
  else if (inpath.find(el_str) != std::string::npos)
  {
    particles = "electrons";
  }
  else if (inpath.find(pi_str) != std::string::npos)
  {
    particles = "pions";
  }
  else
  {
    std::cout << "\nInpath: " << inpath << "\n";
    std::cout << "\nSub string: " << phot_str << "\n";
  }
  return particles;
}

// Run with ./produce.exe --inpath </full/path/to/file.root>
int main(int argc, char **argv)
{
  // read input parameters
  YAML::Node config = YAML::LoadFile("bye_splits/production/prod_params.yaml");

  string tree_name, input_path, out_dir, outfile_base_name;
  tie(tree_name, input_path, out_dir, outfile_base_name) = read_config(config);

  po::variables_map args = process_program_options(argc, argv);
  if (args.count("help"))
  {
    return 1;
  }
  string inpath;
  if (args.count("inpath"))
  {
    inpath = args["inpath"].as<string>();
  }
  else
  {
    inpath = input_path;
  }

  // string inpath = args["inpath"].as<string>();
  int nevents = args["nevents"].as<int>();

  string particles = get_particles(inpath);
  out_dir += particles + "/new_skims/";

  // Grab the file basename from the /full/path
  std::string infile = inpath.substr(inpath.find_last_of("/\\") + 1);
  std::string events_str = nevents > 0 ? std::to_string(nevents) + "events_" : "";
  std::string outfile = outfile_base_name + "_" + events_str + infile;
  std::string outpath = out_dir + outfile;

  skim(tree_name, inpath, outpath, particles, nevents);
  return 0;
}
