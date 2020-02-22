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
size_t SPIKE_NUMBER = 0;
size_t POISSON_CALL_TIME = 0;

int main(int argc, const char* argv[]) {
	clock_t start, finish;
	start = clock();
	// Config program options:
	po::options_description desc("All Options");
	desc.add_options()
		("help,h", "produce help message")
		("prefix", po::value<string>()->default_value("test/"), "prefix of output files")
    ("verbose,v", po::bool_switch(), "show running log.")
		;
	po::options_description config("Configs");
	config.add_options()
		// [network]
		("network.ne", po::value<int>(), "number of Excitatory neurons")
		("network.ni", po::value<int>(), "number of Inhibitory neurons")
		// [neuron]
		("neuron.model", po::value<string>(), "type of neuronal model") 
		("neuron.tref", po::value<double>(), "refractory period") 

		// [synapse]
		("synapse.file", po::value<string>(), "file of synaptic strength")
		// [space]
		("space.file", po::value<string>(), "file of spatial location of neurons")
		// [driving]
		("driving.file", po::value<string>(), "file of Poisson settings")
		("driving.seed", po::value<int>(), "seed to generate Poisson point process")
		// [time]
		("time.t", po::value<double>(), "total simulation time")
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
	int Ne = vm["network.ne"].as<int>();
	int Ni = vm["network.ni"].as<int>();
	int neuron_number = Ne + Ni; 
	NeuronPopulation net(vm["neuron.model"].as<string>(), Ne, Ni);
	// Set interneuronal coupling strength;
	net.InitializeSynapticStrength(vm);
	net.InitializeSynapticDelay(vm);
	net.SetRef(vm["neuron.tref"].as<double>());

	// Init raster output
	string raster_path = dir + "raster.csv";

	// SETUP DYNAMICS:
	double t = 0;
  double dt = vm["time.dt0"].as<double>();
	double tmax = vm["time.t"].as<double>();
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
	vector<vector<double> > spike_trains;
	vector<double> add_spike_train;
	rand_gen.seed(vm["driving.seed"].as<int>());
	net.InitializePoissonGenerator(vm, 100);
	// Start loop;
	for (int i = 0; i < reps; i++) {
		net.InitRasterOutput(dir + "ras_" + to_string(i) + ".csv");
		// Set driving_mode;

		while (t < tmax) {
			net_sim.UpdatePopulationState(&net, t, dt);
			t += dt;
		}
		net.OutPotential(file);
		printf("Total inter-neuronal interaction : %ld\n", NEURON_INTERACTION_TIME);
		printf("Total Poisson Number : %ld\n", POISSON_CALL_TIME);
		t = 0;
		dt /= 2;
		NEURON_INTERACTION_TIME = 0;
		POISSON_CALL_TIME = 0;
    rand_gen.seed(vm["driving.seed"].as<int>());
		net.RestoreNeurons();
		net.CloseRasterOutput();
	}	
	finish = clock();
	printf(">> Total simulation : %3.3f s\n", (finish - start)*1.0 / CLOCKS_PER_SEC);	
	return 0;
}
