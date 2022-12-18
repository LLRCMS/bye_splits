#include <iostream>
#include <vector>
#include <boost/program_options.hpp>
using namespace std;
namespace po = boost::program_options;

//boost custom validator (define option choices)
struct particles {
  particles(const std::string& val): value(val) {}
  std::string value;
};

[[noreturn]]
void show_help(const po::options_description&, const std::string&);
void process_program_options(int argc, char **argv);

void validate(boost::any& v, const vector<string>& values, particles*, int)
{
  // Make sure no previous assignment to 'v' was made.
  po::validators::check_first_occurrence(v);

  // Extract the first string from 'values'. If there is more than
  // one string, it's an error, and exception will be thrown.
  std::string const& s = po::validators::get_single_string(values);

  if (s == "photons" || s == "electrons") {
    v = boost::any(particles(s));
  } else {
    throw std::runtime_error("Error");
	po::validation_error(po::validation_error::invalid_option_value);
  }
}

void show_help(const po::options_description& desc,
			   const std::string& topic = "") {
  std::cout << desc << '\n';
  if (topic != "") {
	std::cout << "You asked for help on: " << topic << '\n';
  }
  exit( EXIT_SUCCESS );
}

void process_program_options(int argc, char **argv)
{
  po::options_description desc("Usage");
  desc.add_options()
	("help,h",
	 po::value<string>()
	 ->implicit_value("")
	 ->notifier([&desc](const std::string &topic) {show_help(desc, topic);}),
	 "Show help. If given, show help on the specified topic.")
	("nentries",
	 po::value<int>()
	 ->implicit_value(-1),
	 "number of entries to consider, useful for debugging (-1 means all)")
	("particles",
	 //po::value<particles>(),
	 po::value<string>(),
	 "type of particle");

    if (argc == 1) {
        show_help(desc); // does not return
    }

    po::variables_map args;

    try {
        po::store(
            po::parse_command_line(argc, argv, desc),
            args
        );
    }
    catch (po::error const& e) {
        std::cerr << e.what() << '\n';
        exit( EXIT_FAILURE );
    }
    po::notify(args);
    return;
}

int main(int argc, char **argv) {
  process_program_options(argc, argv);
  return 0;
}
