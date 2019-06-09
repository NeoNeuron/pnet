//***************
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Date: 2019-05-08
//	Description: test program for Class NeuronalNetwork;
//***************
#include "network.h"
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
		("help", "produce help message")
		("config_file,c", po::value<string>()->default_value("test/config_test.ini"), "config file")
		("prefix", po::value<string>()->default_value("test/"), "prefix of output files")
		;
	po::options_description config("Configs");
	config.add_options()
		// [network]
		("network.size", po::value<int>(), "number of neurons")
		("network.mode", po::value<int>(), "mode of network structure:\n0: external defined connectivity\n1:small-world network\n2: random network")
		("network.file", po::value<string>(), "file of external defined connectivity matrix")
		("network.pee", po::value<double>(), "connecting probability: exc -> exc")
		("network.pie", po::value<double>(), "connecting probability: exc -> inh")
		("network.pei", po::value<double>(), "connecting probability: inh -> exc")
		("network.pii", po::value<double>(), "connecting probability: inh -> inh")
		("network.dens", po::value<int>(), "half of number of connection per neuron in 'small-world' case")
		("network.pr", po::value<double>(), "rewiring probability in 'small-world' case")
		("network.seed", po::value<int>(), "seed to generate network")
		// [neuron]
		("neuron.model", po::value<string>(), "type of neuronal model") 
		("neuron.tref", po::value<double>(), "refractory period") 
		("neuron.mode", po::value<int>(), "distribution mode of neuronal types:\n 0: sequential,\n 1: random,\n 2: external defined;")
		("neuron.p", po::value<double>(), "probability of excitatory neurons")
		("neuron.seed", po::value<int>()->default_value(0), "seed to generate types")
		("neuron.file", po::value<string>(), "file of neuronal types")
		// [synapse]
		("synapse.mode", po::value<int>(), "distribution mode of synaptic strength:\n 0: external defined\n 1: fixed for specific pairs\n 2: fixed with spatial decay")
		("synapse.file", po::value<string>(), "file of synaptic strength")
		("synapse.see", po::value<double>(), "synaptic strength: exc -> exc")
		("synapse.sie", po::value<double>(), "synaptic strength: exc -> inh")
		("synapse.sei", po::value<double>(), "synaptic strength: inh -> exc")
		("synapse.sii", po::value<double>(), "synaptic strength: inh -> inh")
		// [space]
		("space.mode", po::value<int>()->default_value(-1), "delay mode:\n0: homogeneous delay\n1: distance-dependent delay\n-1: no delay")
		("space.delay", po::value<double>()->default_value(0.0), "synaptic delay time")
		("space.speed", po::value<double>(), "transmitting speed of spikes")	
		("space.file", po::value<string>(), "file of spatial location of neurons")
		// [driving]
		("driving.mode", po::value<int>(), "driving mode:\n0: homogeneous Poisson\n1: Poisson with external defined settings")
		("driving.pr", po::value<double>(), "Poisson rate")
		("driving.ps", po::value<double>(), "Poisson strength")
		("driving.file", po::value<string>(), "file of Poisson settings")
		("driving.seed", po::value<int>(), "seed to generate Poisson point process")
		("driving.gmode", po::value<bool>()->default_value(true), "true: generate full Poisson sequence as initialization\nfalse: generate Poisson during simulation by parts")
		// [time]
		("time.T", po::value<double>(), "total simulation time")
		("time.dt0", po::value<double>(), "initial time step")
		("time.stp", po::value<double>(), "sampling time step")
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
	ifstream config_file;
	config_file.open(vm["config_file"].as<string>().c_str());
	po::store(po::parse_config_file(config_file, config), vm);
	po::notify(vm);
	cout << ">> Configs loaded.\n";
	//
	// Network initialization
	//
	int neuron_number = vm["network.size"].as<int>();
	NeuronPopulation net(vm["neuron.model"].as<string>(), neuron_number);
	// initialize the network;
	rand_gen.seed(vm["neuron.seed"].as<int>());
	net.InitializeNeuronalType(vm);
	// load connecting mode;
	rand_gen.seed(vm["network.seed"].as<int>());
	net.InitializeConnectivity(vm);
	rand_gen.seed(vm["network.seed"].as<int>());
	// Set interneuronal coupling strength;
	net.InitializeSynapticStrength(vm);
	net.InitializeSynapticDelay(vm);
	net.SetRef(vm["neuron.tref"].as<double>());

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

	NeuronalNetwork net_sim;
	start = clock();
	int spike_num;
	vector<vector<double> > spike_trains;
	vector<double> add_spike_train;
	// Start loop;
	for (int i = 0; i < reps; i++) {
		net.RestoreNeurons();
		// Set driving_mode;
		rand_gen.seed(vm["driving.seed"].as<int>());
		net.InitializePoissonGenerator(vm);

		while (t < tmax) {
			net_sim.UpdateNetworkState(&net, t, dt);
			t += dt;
		}
		net.OutPotential(file);
		spike_num = net.OutSpikeTrains(spike_trains);
		add_spike_train.clear();
		// record the last spiking event;
		for (int i = 0; i < spike_trains.size(); i ++) {
			add_spike_train.insert(add_spike_train.end(), spike_trains[i].end() - 1, spike_trains[i].end());
		}
		Print1D(dir + "data_network_raster.csv", add_spike_train, "app", 0);
		printf("[-] dt = %.2e s\tmean firing rate = %.2f Hz\n ", dt, spike_num*1000.0/tmax/neuron_number);
		printf("Total inter-neuronal interaction : %d\n", (int)NEURON_INTERACTION_TIME);
		t = 0;
		dt /= 2;
		NEURON_INTERACTION_TIME = 0;
	}	
	finish = clock();
	printf(">> Total simulation : %3.3f s\n", (finish - start)*1.0 / CLOCKS_PER_SEC);	
	return 0;
}
