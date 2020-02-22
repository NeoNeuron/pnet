// =========================
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Created: 2018-09-02 
//	Description: program for point-neuronal-network simulation;
// =========================

#include <chrono>
#include "network_simulator.h"
using namespace std;

mt19937 rand_gen(1);
size_t NEURON_INTERACTION_TIME = 0;
size_t SPIKE_NUMBER = 0;
size_t POISSON_CALL_TIME = 0;

//	Simulation program for single network system;

int main(int argc, const char* argv[]) {
	auto start = chrono::system_clock::now();
	// Config program options:
  bool verbose;
	po::options_description generic("All Options");
	generic.add_options()
		("help,h", "produce help message")
		("prefix,p", po::value<string>()->default_value("./"), "prefix of output files")
		("config,c", po::value<string>(), "config file, relative to prefix")
    ("verbose,v", po::bool_switch(&verbose), "show output")
		;
	po::options_description config("Configs");
	config.add_options()
		// [network]
		("network.ne", po::value<int>(), "number of Excitatory neurons")
		("network.ni", po::value<int>(), "number of Inhibitory neurons")
		// [neuron]
		("neuron.model", po::value<string>(), "type of neuronal model") 
		("neuron.tref", po::value<double>()->default_value(2.0), "refractory period") 
		// [synapse]
		("synapse.file", po::value<string>(), "file of synaptic strength")
		// [space]
		("space.file", po::value<string>(), "file of spatial location of neurons")
		// [driving]
		("driving.file", po::value<string>()->default_value(""), "file of Poisson settings")
		("driving.seed", po::value<int>(), "seed to generate Poisson point process")
		// [time]
		("time.t", po::value<double>(), "total simulation time")
		("time.dt", po::value<double>(), "simulation time step")
		("time.stp", po::value<double>(), "sampling time step")
		// [output]
		("output.poi", po::value<bool>()->default_value(false), "output flag of Poisson Drive")
		("output.v", po::value<bool>()->default_value(false), "output flag of V")
		("output.i", po::value<bool>()->default_value(false), "output flag of I")
		("output.ge", po::value<bool>()->default_value(false), "output flag of GE")
		("output.gi", po::value<bool>()->default_value(false), "output flag of GI")
		;
  po::options_description cml_options, config_file_options;
  cml_options.add(generic).add(config);
  config_file_options.add(config);
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, cml_options), vm);
	po::notify(vm);
	if (vm.count("help")) {
		cout << generic << '\n';
		cout << config << '\n';
		return 1;
	}
	// existing directory for output files;
	string dir;
	dir = vm["prefix"].as<string>();

	// Loading config.ini:
	ifstream config_file;
	if (vm.count("config")) {
    string cfname = vm["prefix"].as<string>() + vm["config"].as<string>();
    config_file.open(cfname.c_str());
    po::store(po::parse_config_file(config_file, config), vm);
    po::notify(vm);
    if (verbose) {
      cout << ">> Configs loaded.\n";
    }
	}

	//
	// Network initialization
	//
	int Ne = vm["network.ne"].as<int>();
	int Ni = vm["network.ni"].as<int>();
	NeuronPopulation net(vm["neuron.model"].as<string>(), Ne, Ni);
	// Set interneuronal coupling strength;
	net.InitializeSynapticStrength(vm);
	net.InitializeSynapticDelay(vm);
	net.SetRef(vm["neuron.tref"].as<double>());

	// Set driving_mode;
	rand_gen.seed(vm["driving.seed"].as<int>());
	net.InitializePoissonGenerator(vm, 100);

	// Init raster output
	string raster_path = dir + "raster.csv";
	net.InitRasterOutput(raster_path);
	
	// SETUP DYNAMICS:
	double t = 0, dt = vm["time.dt"].as<double>();
	double tmax = vm["time.t"].as<double>();
	double recording_rate = 1.0 / vm["time.stp"].as<double>();
	// Define the shape of data;
	size_t shape[2];
	shape[0] = tmax * recording_rate;
	shape[1] = Ne + Ni;

	// Define file-outputing flags;
	bool v_flag, i_flag, ge_flag, gi_flag;
	v_flag = vm["output.v"].as<bool>();
	i_flag = vm["output.i"].as<bool>();
	ge_flag = vm["output.ge"].as<bool>();
	gi_flag = vm["output.gi"].as<bool>();

	// Create file-write objects;
	FILEWRITE v_file(dir + "V.bin", "trunc");
	FILEWRITE i_file(dir + "I.bin", "trunc");
	FILEWRITE ge_file(dir + "GE.bin", "trunc");
	FILEWRITE gi_file(dir + "GI.bin", "trunc");
	// Initialize size parameter in files:
	if (v_flag) v_file.SetSize(shape);
	if (i_flag) i_file.SetSize(shape);
	if (ge_flag) ge_file.SetSize(shape);
	if (gi_flag) gi_file.SetSize(shape);

	auto finish = chrono::system_clock::now();
	chrono::duration<double> elapsed_seconds = finish-start;
  if (verbose) {
    printf(">> Initialization : \t%3.3f s\n", elapsed_seconds.count());
    fflush(stdout);
  }

	start = chrono::system_clock::now();
  dbg_printf(">>> number of Poisson spikes : %ld", POISSON_CALL_TIME);
	NetworkSimulatorSSC net_sim;
	int progress = 0;
	while (t < tmax) {
		net_sim.UpdatePopulationState(&net, t, dt);
		t += dt;
		// Output temporal data;
		if (abs(recording_rate*t - floor(recording_rate*t)) == 0) {
			if (v_flag) net.OutPotential(v_file);
			if (i_flag) net.OutCurrent(i_file);
			if (ge_flag) net.OutConductance(ge_file, true);
			if (gi_flag) net.OutConductance(gi_file, false);
		}
    if (verbose) {
      if (floor(t / tmax * 100) > progress) {
        progress = floor(t / tmax * 100);
        printf(">> Running ... %d%%\r", progress);
        fflush(stdout);
      }
    }
	}
	finish = chrono::system_clock::now();
  net.CloseRasterOutput();

	// delete files;
	if (!v_flag) v_file.Remove();
	if (!i_flag) i_file.Remove();
	if (!ge_flag) ge_file.Remove();
	if (!gi_flag) gi_file.Remove();
	
  if (verbose) {
    printf(">>> number of Poisson spikes : %ld", POISSON_CALL_TIME);
    printf(">> Done!             \n");
  }
	
	elapsed_seconds = finish-start;
  if (verbose) {
    printf(">> Simulation : \t%3.3f s\n", elapsed_seconds.count());
    printf("Total inter-neuronal interaction : %d\n", (int)NEURON_INTERACTION_TIME);
    printf("Mean firing rate : %5.2f Hz\n", (double)SPIKE_NUMBER/tmax*1000.0/(Ne+Ni));
  }
	return 0;
}
