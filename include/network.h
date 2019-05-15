//***************
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Date: 2018-05-31
//	Description: define Struct SpikeElement and Class NeuronalNetwork;
//***************
#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "io.h"
#include "neuron.h"
#include "poisson_generator.h"
#include "common_header.h"
#include <boost/program_options.hpp>
#include <eigen3/Eigen/Dense>
namespace po = boost::program_options;

using namespace std;

struct SpikeElement {
	int index;	// The sequence order of spikes within single time interval;
	double t;	// exact spiking time;
	bool type;	// The type of neuron that fired;
};

bool compSpikeElement(const SpikeElement &x, const SpikeElement &y);

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DymVals; 

inline double* GetPtr(DymVals &mat, int id) {
	return mat.data() + id * mat.cols();
}

class NeuronalNetwork {
private:
	// Neuron Simulators:
	vector<NeuronSim> neurons_;
	int dym_n_;
	
	// PoissonGenerators:
	vector<PoissonGenerator> pgs_;
	bool pg_mode;

	// Network Parameters:
	int neuron_number_;	// number of the neurons in the group;
	DymVals dym_vals_;	// dynamic variables of neurons;
	vector<bool> types_; // vector to store types of neurons;

	// Network Structure:
	vector<vector<bool> > con_mat_; // built-in matrix for neuronal connectivity;
	bool is_con_;
	vector<vector<double> > s_mat_; // matrix of inter-neuronal interacting strength;
	vector<vector<double> > delay_mat_;

	// Network Inputs:
	vector<queue<Spike> > ext_inputs_; // temp storage of external Poisson input;


	// Functions:
	//
  // Set interaction delay between neurons;
	void SetDelay(vector<vector<double> > &coordinates, double speed);

	// Sort spikes within single time interval, and return the time of first spike;
	double SortSpikes(DymVals &dym_val_new, vector<int>& update_list, vector<int>& fired_list, double t, double dt, vector<SpikeElement> &T);

public:
	//	Neuronal network initialization:
	NeuronalNetwork(string neuron_type, int neuron_number) {
		// Network Parameters:
		neuron_number_ = neuron_number;
		neurons_.resize(neuron_number_, NeuronSim(neuron_type));
		if (neuron_type == "LIF_GH") {
			dym_n_ = 6;
		} else {
			dym_n_ = 4;
		}
		dym_vals_.resize(neuron_number_, dym_n_);
		for (int i = 0; i < neuron_number_; i++) {
			neurons_[i].SetDefaultDymVal(GetPtr(dym_vals_, i));
		}
		pgs_.resize(neuron_number_);
		types_.resize(neuron_number_, false);
		// Network structure:
		con_mat_.resize(neuron_number_, vector<bool>(neuron_number_, false));
		is_con_ = false;
		s_mat_.resize(neuron_number_, vector<double>(neuron_number_, 0.0));
		delay_mat_.resize(neuron_number_, vector<double>(neuron_number_, 0.0));
		ext_inputs_.resize(neuron_number_);
	}
	
	~NeuronalNetwork() {  }
	// Initialize network connectivity matrix:
	// Three options:
	// 0. given pre-defined network connectivity matrix;
	// 1. small-world network, defined by connectivity density and rewiring;
	// 2. randomly connected network;
	void InitializeConnectivity(po::variables_map &vm);
	
	// INPUTS:
	// Set interneuronal coupling strength;
	void InitializeSynapticStrength(po::variables_map &vm);
	// Initialize the delay of synaptic interaction;
	void InitializeSynapticDelay(po::variables_map &vm);
	
	// Set time period of refractory:
	void SetRef(double t_ref);

	// 	Initialize neuronal types in the network;
	//	p: the probability of the presence of excitatory neuron;
	//	seed: seed for random number generator;
	void InitializeNeuronalType(po::variables_map &vm);

	//	Set driving type: true for external Poisson driven, false for internal ones;
	void SetDrivingType(bool driving_type);

	//	Initialize internal homogeneous feedforward Poisson rate;
	void InitializePoissonGenerator(po::variables_map &vm);

	// 	Input new spikes for neurons all together;
	void InNewSpikes(vector<vector<Spike> > &data);

	// DYNAMICS:

	//	Restore neuronal state for all neurons, including neuronal potential, conductances, refractory periods and external network drive;
	void RestoreNeurons();

	//	Update network state:
	void UpdateNetworkState(double t, double dt);

	// OUTPUTS:

	// Print cycle:
	void PrintCycle();

	//	Output potential to *.csv file;
	void OutPotential(FILEWRITE& file);

  //	Output synaptic conductance to *.csv files:
  //		BOOL function: function of synaptic conductance, true for excitation, false for inhibition;
  void OutConductance(FILEWRITE& file, bool function);

	//	Output current to *.csv file;
	void OutCurrent(FILEWRITE& file);

	// Save neuronal type vector;
	void SaveNeuronType(string neuron_type_file);

	// Save connectivity matrix
	void SaveConMat(string connecting_matrix_file);

	// Output spike trains of the network to spike_trains;
	// return: the total number of spikes in the network during the simulation;
	int OutSpikeTrains(vector<vector<double> >& spike_trains);

  //  Output spikes before t, including their functions;
	void GetNewSpikes(double t, vector<vector<Spike> >& data);

	int GetNeuronNumber();

	void GetConductance(int i, bool function);
};

#endif // _NETWORK_H_
