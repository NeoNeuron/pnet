// =========================
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Created: 2018-09-02 
//	Description: program for point-neuronal-network simulation;
// =========================

#include <chrono>
#include "network_simulator.h"
using namespace std;

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
    ("network.ne", po::value<int>()->default_value(1), "number of Exc. neurons")
    ("network.ni", po::value<int>()->default_value(0), "number of Inh. neurons")
    ("network.simulator", po::value<string>()->default_value("SSC"),
      "One of Simple, SSC, SSC_Sparse.")
    // [neuron]
    ("neuron.model", po::value<string>()->default_value("LIF_GH"),
      "One of LIF_I, LIF_G, LIF_GH.") 
    ("neuron.tref", po::value<double>()->default_value(2.0), "(ms) refractory period") 
    // [synapse]
    ("synapse.file", po::value<string>(), "file of synaptic strength")
    // [space]
    ("space.file", po::value<string>(), "file of spatial location of neurons")
    // [driving]
    ("driving.file", po::value<string>()->default_value(""), "file of Poisson settings")
    ("driving.seed", po::value<int>()->default_value(NULL), "seed to generate Poisson point process")
    // [time]
    ("time.t", po::value<double>()->default_value(1000), "(ms) total simulation time")
    ("time.dt", po::value<double>()->default_value(0.03125), "(ms) simulation time step")
    ("time.stp", po::value<double>()->default_value(0.5), "(ms) sampling time step")
    // [output]
    ("output.poi", po::value<bool>()->default_value(false), "output flag of Poisson Drive")
    ("output.v",  po::value<bool>()->default_value(false), "output flag of V to V.bin")
    ("output.i",  po::value<bool>()->default_value(false), "output flag of I to I.bin")
    ("output.ge", po::value<bool>()->default_value(false), "output flag of GE to GE.bin")
    ("output.gi", po::value<bool>()->default_value(false), "output flag of GI to GI.bin")
    ;
  po::options_description cml_options;
  cml_options.add(generic).add(config);
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
    if (verbose) cout << ">> Configs loaded.\n";
  }

  //
  // Network initialization
  //
  int Ne = vm["network.ne"].as<int>();
  int Ni = vm["network.ni"].as<int>();
  NeuronPopulationBase* p_net = NULL;
  if (vm["neuron.model"].as<std::string>() == "LIF_G") {
    p_net = new NeuronPopulationNoContinuousCurrent<LIF_G>(Ne, Ni);
  } else if (vm["neuron.model"].as<std::string>() == "LIF_GH") {
    p_net = new NeuronPopulationNoContinuousCurrent<LIF_GH>(Ne, Ni);
  } else if (vm["neuron.model"].as<std::string>() == "LIF_I") {
    p_net = new NeuronPopulationNoContinuousCurrent<LIF_I>(Ne, Ni);
  } else {
    throw runtime_error("Invalid neuron type");
  }

  // Set interneuronal coupling strength;
  p_net->InitializeSynapticStrength(vm);
  p_net->InitializeSynapticDelay(vm);
  p_net->SetRef(vm["neuron.tref"].as<double>());

  // Set driving_mode;
  p_net->InitializePoissonGenerator(vm, 10);
  // p_net->RestoreNeurons();

  // Init raster output
  string raster_path = dir + "raster.csv";
  p_net->InitRasterOutput(raster_path);

  
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
  NetworkSimulatorBase* net_sim = NULL;
  if (vm["network.simulator"].as<string>() == "Simple") {
    net_sim = new NetworkSimulatorSimple();
  } else if (vm["network.simulator"].as<string>() == "SSC") {
    net_sim = new NetworkSimulatorSSC();
  } else if (vm["network.simulator"].as<string>() == "SSC_Sparse") {
    net_sim = new NetworkSimulatorSSC_Sparse();
  } else {
    throw runtime_error("Invalid simulator type");
  }

  int progress = 0;
  while (t < tmax) {
    net_sim->UpdatePopulationState(p_net, t, dt);
    t += dt;
    // Output temporal data;
    if (abs(recording_rate*t - floor(recording_rate*t)) == 0) {
      if (v_flag) p_net->OutPotential(v_file);
      if (i_flag) p_net->OutCurrent(i_file);
      if (ge_flag) p_net->OutConductance(ge_file, true);
      if (gi_flag) p_net->OutConductance(gi_file, false);
    }
    if (verbose) {
      if (floor(t / tmax * 100) > progress) {
        progress = floor(t / tmax * 100);
        printf(">> Running ... %d%%\r", progress);
        fflush(stdout);
      }
    }
  }
  cout << endl;
  finish = chrono::system_clock::now();
  p_net->CloseRasterOutput();

  // delete files;
  if (!v_flag) v_file.Remove();
  if (!i_flag) i_file.Remove();
  if (!ge_flag) ge_file.Remove();
  if (!gi_flag) gi_file.Remove();
  
  if (verbose) {
    printf(">> Done!\n");
    printf(">>> number of Poisson spikes : %ld\n", POISSON_CALL_TIME);
  }
  
  elapsed_seconds = finish-start;
  if (verbose) {
    printf(">> Simulation : \t%3.3f s\n", elapsed_seconds.count());
    printf("Total inter-neuronal interaction : %d\n", (int)NEURON_INTERACTION_TIME);
    printf("Mean firing rate : %5.2f Hz\n", (double)SPIKE_NUMBER/tmax*1000.0/(Ne+Ni));
  }
  return 0;
}
