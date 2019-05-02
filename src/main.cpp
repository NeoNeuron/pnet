//*************************
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Date: 2018-09-02 
//	Description: test program for multi-network simulation;
//*************************

#include "network.h"
using namespace std;

mt19937 rand_gen(1);
uniform_real_distribution<> rand_distribution(0.0, 1.0);

//	Simulation program for single network system;
//	
//	arguments:
//	argv[1] = path of config file;
//	argv[2] = Output directory for neural data;
//
int main(int argc, const char* argv[]) {
	if (argc != 3) throw runtime_error("wrong number of args");
	clock_t start, finish;
	start = clock();
	// 	Setup directory for output files;
	//	it must be existing dir;
	string dir;
	dir = argv[2];

	// Loading config.ini:
	string net_config_path = argv[1];
  map<string, string> m_map_config;
  ReadConfig(net_config_path,m_map_config);
  cout << ">> [Config.ini]:\n#####\n";
	PrintConfig(m_map_config);
	cout << "#####\n";
	// load neuron number;
	int neuron_number = atoi(m_map_config["NeuronNumber"].c_str());
	NeuronalNetwork net(m_map_config["NeuronType"], neuron_number);
	// initialize the network;
	rand_gen.seed(atoi(m_map_config["TypeSeed"].c_str()));
	net.InitializeNeuronalType(m_map_config);
	// load connecting mode;
	rand_gen.seed(atoi(m_map_config["NetSeed"].c_str()));
	net.InitializeConnectivity(m_map_config);
	// Set interneuronal coupling strength;
	net.InitializeSynapticStrength(m_map_config);
	net.InitializeSynapticDelay(m_map_config);
	net.SetRef(atof(m_map_config["RefractoryTime"].c_str()));

	// Set driving_mode;
	rand_gen.seed(atoi(m_map_config["pSeed"].c_str()));
	net.InitializePoissonGenerator(m_map_config);

	// SETUP DYNAMICS:
	double t = 0, dt = atof(m_map_config["TimingStep"].c_str());
	double tmax = atof(m_map_config["MaximumTime"].c_str());
	double recording_rate = 1.0 / atof(m_map_config["SamplingTimingStep"].c_str());
	// Define the shape of data;
	size_t shape[2];
	shape[0] = tmax * recording_rate;
	shape[1] = neuron_number;

	// Define file-outputing flags;
	bool v_flag, i_flag, ge_flag, gi_flag;
	istringstream(m_map_config["SaveV"]) >> boolalpha >> v_flag;
	istringstream(m_map_config["SaveI"]) >> boolalpha >> i_flag;
	istringstream(m_map_config["SaveGE"]) >> boolalpha >> ge_flag;
	istringstream(m_map_config["SaveGI"]) >> boolalpha >> gi_flag;

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
	printf(">> Initialization : %3.3f s\n", (finish - start)*1.0 / CLOCKS_PER_SEC);
	fflush(stdout);

	start = clock();
	int progress = 0;
	while (t < tmax) {
		net.UpdateNetworkState(t, dt);
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
	finish = clock();

	// delete files;
	if (!v_flag) v_file.Remove();
	if (!i_flag) i_file.Remove();
	if (!ge_flag) ge_file.Remove();
	if (!gi_flag) gi_file.Remove();
	
	cout << endl;
	net.PrintCycle();
	
	// OUTPUTS:
	net.SaveNeuronType(dir + "neuron_type.csv");
	net.SaveConMat(dir + "mat.csv");

	vector<vector<double> > spike_trains;
	int spike_num = net.OutSpikeTrains(spike_trains);
	string raster_path = dir + "raster.csv";
	Print2D(raster_path, spike_trains, "trunc");
	cout << ">> Mean firing rate: " << (double)spike_num*1000.0/tmax/neuron_number << endl;

	// Timing:
	printf(">> Simulation : %3.3f s\n", (finish - start)*1.0 / CLOCKS_PER_SEC);
	return 0;
}
