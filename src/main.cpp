//*************************
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Date: 2019-05-02 
//	Description: program for point-neuronal-network simulation;
//*************************

#include "network.h"
using namespace std;

mt19937 rand_gen(1);
uniform_real_distribution<> rand_distribution(0.0, 1.0);
size_t NEURON_INTERACTION_TIME = 0;

//	Simulation program for single network system;
//	
//	arguments:
//	argv[1] = path of config file;
//	argv[2] = Output directory for neural data;
//
int main(int argc, const char* argv[]) {
	clock_t start, finish;
	start = clock();
	// Config program options:
	po::options_description desc("All Options");
	desc.add_options()
		("help,h", "produce help message")
		("config", "detailed message for config file")
		("config-file,c", po::value<string>(), "config file")
		("prefix", po::value<string>()->default_value("./"), "prefix of output files")
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
		("synapse.mode", po::value<int>(), "distribution mode of synaptic strength:\n 0: external defined\n 1: fixed for specific pairs")
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
		config_file.open(vm["config-file"].as<string>().c_str());
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
	int neuron_number = vm["network.size"].as<int>();
	NeuronalNetwork net(vm["neuron.model"].as<string>(), neuron_number);
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

	// Set driving_mode;
	rand_gen.seed(vm["driving.seed"].as<int>());
	net.InitializePoissonGenerator(vm);

	// SETUP DYNAMICS:
	double t = 0, dt = vm["time.dt"].as<double>();
	double tmax = vm["time.T"].as<double>();
	double recording_rate = 1.0 / vm["time.stp"].as<double>();
	// Define the shape of data;
	size_t shape[2];
	shape[0] = tmax * recording_rate;
	shape[1] = neuron_number;

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

	finish = clock();
	printf(">> Initialization : \t%3.3f s\n", (finish - start)*1.0 / CLOCKS_PER_SEC);
	fflush(stdout);

	start = clock();
	int progress = 0;
	while (t < tmax) {
		net.UpdateNetworkState(t, dt);
		t += dt;
		// Output temporal data;
		if (abs(recording_rate*t - floor(recording_rate*t)) == 0) {
			if (v_flag) net.OutPotential(v_file);
			//if (i_flag) net.OutCurrent(i_file);
			if (ge_flag) net.OutConductance(ge_file, true);
			if (gi_flag) net.OutConductance(gi_file, false);
		}
		if (floor(t / tmax * 100) > progress) {
			progress = floor(t / tmax * 100);
			printf(">> Running ... %d%%\r", progress);
			fflush(stdout);
		}
	}
	finish = clock();

	// delete files;
	if (!v_flag) v_file.Remove();
	if (!i_flag) i_file.Remove();
	if (!ge_flag) ge_file.Remove();
	if (!gi_flag) gi_file.Remove();
	
	printf(">> Done!             \n");
	//net.PrintCycle();
	
	printf(">> Simulation : \t%3.3f s\n", (finish - start)*1.0 / CLOCKS_PER_SEC);
	
	printf("Total inter-neuronal interaction : %d\n", (int)NEURON_INTERACTION_TIME);
	// OUTPUTS:
	start = clock();
	net.SaveNeuronType(dir + "neuron_type.csv");
	net.SaveConMat(dir + "mat.csv");

	vector<vector<double> > spike_trains;
	int spike_num = net.OutSpikeTrains(spike_trains);
	string raster_path = dir + "raster.csv";
	Print2D(raster_path, spike_trains, "trunc");
	printf(">> Mean firing rate: %3.3f Hz\n", spike_num*1000.0/tmax/neuron_number);
	finish = clock();

	// Timing:
	printf(">> Saving outputs : \t%3.3f s\n", (finish - start)*1.0 / CLOCKS_PER_SEC);
	return 0;
}
