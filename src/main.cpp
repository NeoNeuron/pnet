//*************************
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Date: 2019-05-02 
//	Description: program for point-neuronal-network simulation;
//*************************

#include <chrono>
#include "network_simulator.h"
using namespace std;

mt19937 rand_gen(1);
uniform_real_distribution<> rand_distribution(0.0, 1.0);
size_t NEURON_INTERACTION_TIME = 0;
size_t SPIKE_NUMBER = 0;

//	Simulation program for single network system;
//	
//	arguments:
//	argv[1] = path of config file;
//	argv[2] = Output directory for neural data;
//
int main(int argc, const char* argv[]) {
	auto start = chrono::system_clock::now();
	// Config program options:
	po::options_description desc("All Options");
	desc.add_options()
		("help,h", "produce help message")
		("config", "detailed message for config file")
		("prefix", po::value<string>()->default_value("./"), "prefix of output files")
		("config-file,c", po::value<string>()->default_value("config.ini"), "config file")
		;
	po::options_description config("Configs");
	config.add_options()
		// [network]
		("network.Ne", po::value<int>(), "number of E neurons")
		("network.Ni", po::value<int>(), "number of I neurons")
		// [neuron]
		("neuron.model", po::value<string>(), "type of neuronal model") 
		("neuron.tref", po::value<double>(), "refractory period") 
		// [synapse]
		("synapse.file", po::value<string>(), "file of synaptic strength")
		// [space]
		("space.mode", po::value<int>()->default_value(-1), "delay mode:\n0: external defined distance-dependent delay\n1: homogeneous delay\n-1: no delay")
		("space.delay", po::value<double>()->default_value(0.0), "synaptic delay time")
		("space.speed", po::value<double>(), "transmitting speed of spikes")	
		("space.file", po::value<string>(), "file of spatial location of neurons")
		// [driving]
		("driving.file", po::value<string>()->default_value(""), "file of Poisson settings")
		("driving.seed", po::value<int>(), "seed to generate Poisson point process")
		("driving.gmode", po::value<bool>()->default_value(true), "true: generate full Poisson sequence as initialization\nfalse: generate Poisson during simulation by parts")
		// [time]
		("time.T", po::value<double>(), "total simulation time")
		("time.dt", po::value<double>(), "simulation time step")
		("time.stp", po::value<double>(), "sampling time step")
		// [output]
		("output.poi", po::value<bool>()->default_value(false), "output flag of Poisson Drive")
		("output.V", po::value<bool>()->default_value(false), "output flag of V")
		("output.I", po::value<bool>()->default_value(false), "output flag of I")
		("output.GE", po::value<bool>()->default_value(false), "output flag of GE")
		("output.GI", po::value<bool>()->default_value(false), "output flag of GI")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
	if (vm.count("help")) {
		cout << desc << '\n';
		return 1;
	}
	if (vm.count("config")) {
		cout << config << '\n';
		return 1;
	}
	// existing directory for output files;
	string dir;
	dir = vm["prefix"].as<string>();

	// Loading config.ini:
	ifstream config_file;
	if (vm.count("config-file")) {
		string cfname = vm["prefix"].as<string>() + vm["config-file"].as<string>();
		config_file.open(cfname.c_str());
	} else {
		cout << "lack of config file\n";
		return -1;
	}
	po::store(po::parse_config_file(config_file, config), vm);
	po::notify(vm);
	cout << ">> Configs loaded.\n";

	//
	// Network initialization
	//
	int Ne = vm["network.Ne"].as<int>();
	int Ni = vm["network.Ni"].as<int>();
	NeuronPopulation net(vm["neuron.model"].as<string>(), Ne, Ni);
	// Set interneuronal coupling strength;
	net.InitializeSynapticStrength(vm);
	net.InitializeSynapticDelay(vm);
	net.SetRef(vm["neuron.tref"].as<double>());

	// Set driving_mode;
	rand_gen.seed(vm["driving.seed"].as<int>());
	net.InitializePoissonGenerator(vm);

	// Init raster output
	string raster_path = dir + "raster.csv";
	net.InitRasterOutput(raster_path);
	
	// SETUP DYNAMICS:
	double t = 0, dt = vm["time.dt"].as<double>();
	double tmax = vm["time.T"].as<double>();
	double recording_rate = 1.0 / vm["time.stp"].as<double>();
	// Define the shape of data;
	size_t shape[2];
	shape[0] = tmax * recording_rate;
	shape[1] = Ne + Ni;

	// Define file-outputing flags;
	bool v_flag, i_flag, ge_flag, gi_flag;
	v_flag = vm["output.V"].as<bool>();
	i_flag = vm["output.I"].as<bool>();
	ge_flag = vm["output.GE"].as<bool>();
	gi_flag = vm["output.GI"].as<bool>();

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
	printf(">> Initialization : \t%3.3f s\n", elapsed_seconds.count());
	fflush(stdout);

	start = chrono::system_clock::now();
	NetworkSimulatorSSC net_sim;
	int progress = 0;
	while (t < tmax) {
		net_sim.UpdateState(&net, t, dt);
		t += dt;
		// Output temporal data;
		if (abs(recording_rate*t - floor(recording_rate*t)) == 0) {
			if (v_flag) net.OutPotential(v_file);
			if (i_flag) net.OutCurrent(i_file);
			if (ge_flag) net.OutConductance(ge_file, true);
			if (gi_flag) net.OutConductance(gi_file, false);
		}
		if (floor(t / tmax * 100) > progress) {
			progress = floor(t / tmax * 100);
			printf(">> Running ... %d%%\r", progress);
			fflush(stdout);
		}
	}
	finish = chrono::system_clock::now();
  net.CloseRasterOutput();

	// delete files;
	if (!v_flag) v_file.Remove();
	if (!i_flag) i_file.Remove();
	if (!ge_flag) ge_file.Remove();
	if (!gi_flag) gi_file.Remove();
	
	printf(">> Done!             \n");
	//net.PrintCycle();
	
	elapsed_seconds = finish-start;
	printf(">> Simulation : \t%3.3f s\n", elapsed_seconds.count());
	
	printf("Total inter-neuronal interaction : %d\n", (int)NEURON_INTERACTION_TIME);
	printf("Mean firing rate : %5.2f Hz\n", (double)SPIKE_NUMBER/tmax*1000.0/(Ne+Ni));

	return 0;
}
