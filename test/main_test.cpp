//***************
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Date: 2019-05-08
//	Description: test program for Class NeuronalNetwork;
//***************
#include "network_simulator.h"
using namespace std;

mt19937 rand_gen(1);
uniform_real_distribution<> rand_distribution(0.0, 1.0);
size_t NEURON_INTERACTION_TIME = 0;

int main(int argc, const char* argv[]) {
	clock_t start, finish;
	start = clock();
	// Config program options:
	po::options_description desc("All Options");
	desc.add_options()
		("help,h", "produce help message")
		("prefix", po::value<string>()->default_value("test/"), "prefix of output files")
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
		("space.mode", po::value<int>()->default_value(-1), "delay mode:\n0: homogeneous delay\n1: distance-dependent delay\n-1: no delay")
		("space.delay", po::value<double>()->default_value(0.0), "synaptic delay time")
		("space.speed", po::value<double>(), "transmitting speed of spikes")	
		("space.file", po::value<string>(), "file of spatial location of neurons")
		// [driving]
		("driving.file", po::value<string>(), "file of Poisson settings")
		("driving.seed", po::value<int>(), "seed to generate Poisson point process")
		("driving.gmode", po::value<bool>()->default_value(true), "true: generate full Poisson sequence as initialization\nfalse: generate Poisson during simulation by parts")
		// [time]
		("time.T", po::value<double>(), "total simulation time")
		("time.dt0", po::value<double>(), "initial time step")
		("time.reps", po::value<int>(), "repeating times")
		// [output]
		("output.poi", po::value<bool>()->default_value(false), "output flag of Poisson Drive")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
	if (vm.count("help")) {
		cout << desc << '\n';
		cout << config << '\n';
		return 1;
	}
	// existing directory for output files;
	string dir;
	dir = vm["prefix"].as<string>();

	// Loading config.ini:
	string config_path = dir + "/config.ini";
	ifstream config_file;
	config_file.open(config_path.c_str());
	po::store(po::parse_config_file(config_file, config), vm);
	po::notify(vm);
	cout << ">> Configs loaded.\n";
	//
	// Network initialization
	//
	int Ne = vm["network.Ne"].as<int>();
	int Ni = vm["network.Ni"].as<int>();
	int neuron_number = Ne + Ni; 
	NeuronPopulation net(vm["neuron.model"].as<string>(), Ne, Ni);
	// Set interneuronal coupling strength;
	net.InitializeSynapticStrength(vm);
	net.InitializeSynapticDelay(vm);
	net.SetRef(vm["neuron.tref"].as<double>());

	// Init raster output
	string raster_path = dir + "raster.csv";

	// SETUP DYNAMICS:
	double t = 0, dt = vm["time.dt0"].as<double>();
	double tmax = vm["time.T"].as<double>();
	int reps = vm["time.reps"].as<int>();
	// Define the shape of data;
	size_t shape[2];
	shape[0] = reps;
	shape[1] = neuron_number;

	// prepare data file;
	FILEWRITE file(dir + "data_network_test.bin", "trunc");
	file.SetSize(shape);
	ofstream ofile(dir + "data_network_raster.csv");
	ofile.close();

	finish = clock();
	printf(">> Initialization : %3.3f s\n", (finish - start)*1.0 / CLOCKS_PER_SEC);
	fflush(stdout);

	NetworkSimulatorSSC net_sim;
	start = clock();
	int spike_num;
	vector<vector<double> > spike_trains;
	vector<double> add_spike_train;
	// Start loop;
	for (int i = 0; i < reps; i++) {
		net.InitRasterOutput(dir + "ras_" + to_string(i) + ".csv");
		// Set driving_mode;
		rand_gen.seed(vm["driving.seed"].as<int>());
		net.InitializePoissonGenerator(vm);

		while (t < tmax) {
			net_sim.UpdateState(&net, t, dt);
			t += dt;
		}
		net.OutPotential(file);
		printf("Total inter-neuronal interaction : %d\n", (int)NEURON_INTERACTION_TIME);
		net.RestoreNeurons();
		t = 0;
		dt /= 2;
		NEURON_INTERACTION_TIME = 0;
		net.CloseRasterOutput();
	}	
	finish = clock();
	printf(">> Total simulation : %3.3f s\n", (finish - start)*1.0 / CLOCKS_PER_SEC);	
	return 0;
}
