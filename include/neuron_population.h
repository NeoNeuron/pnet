//***************
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Date: 2019-06-09
//	Description: define Struct SpikeElement and Class NeuronPopulation;
//***************
#ifndef _NEURON_POPULATION_H_
#define _NEURON_POPULATION_H_

#include "io.h"
#include "neuron.h"
#include "poisson_generator.h"
#include "common_header.h"
#include <boost/program_options.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
namespace po = boost::program_options;

using namespace std;

struct SpikeElement {
	int index;	// The sequence order of spikes within single time interval;
	double t;		// exact spiking time;
	bool type;	// The type of neuron that fired;

	bool operator < (const SpikeElement &b) const
  { return t < b.t; }
  bool operator > (const SpikeElement &b) const
  { return t > b.t; }
  bool operator == (const SpikeElement &b) const
  { return t == b.t && type == b.type; }
};

inline bool compSpikeElement(const SpikeElement &x, const SpikeElement &y) { return x < y; }
//inline bool compSpikeElement(const SpikeElement &x, const SpikeElement &y) { return x.t < y.t; }

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DymVals; 
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> ConMat; 

inline double* GetPtr(DymVals &mat, int id) {
	return mat.data() + id * mat.cols();
}

// class to containing neuronal data
class NeuronPopulation {
	public:
		// Neuron Simulators:
		NeuronSimulatorBase *neuron_sim_ = NULL;
		int dym_n_;
	
		// Network Parameters:
		int neuron_number_;		// number of the neurons in the group;
		int Ne_;		// number of excitatory neurons;
		DymVals dym_vals_;		// dynamic variables of neurons;

		// PoissonGenerators:
		vector<PoissonGenerator> pgs_;
		bool pg_mode;

		// Network Structure:
		bool is_con_;
		ConMat s_mat_;									// matrix of inter-neuronal interacting strength;
		vector<vector<double> > delay_mat_;

		// Network Inputs:
		vector<queue<Spike> > ext_inputs_; // temp storage of external Poisson input;

		// Network data:
		vector<vector<Spike> > synaptic_drivens_;
		ofstream raster_file_;

	//public:
		// Neuronal network initialization:
		NeuronPopulation(string neuron_type, int Ne, int Ni) {
			// Network Parameters:
			if (neuron_type == "LIF_G") {
				neuron_sim_ = new Sim_LIF_G();
			} else if (neuron_type == "LIF_GH") {
				neuron_sim_ = new Sim_LIF_GH();
			} else if (neuron_type == "LIF_I") {
				neuron_sim_ = new Sim_LIF_I();
			} else {
				throw runtime_error("ERROR: wrong neuron type");
			}
			dym_n_ = neuron_sim_->GetDymNum();
			Ne_ = Ne;
			neuron_number_ = Ne + Ni;
			dym_vals_.resize(neuron_number_, dym_n_);
			for (int i = 0; i < neuron_number_; i++) {
				neuron_sim_->GetDefaultDymVal(GetPtr(dym_vals_, i));
			}
			pgs_.resize(neuron_number_);
			// Network structure:
			is_con_ = false;
			s_mat_.resize(neuron_number_, neuron_number_);
			delay_mat_.resize(neuron_number_, vector<double>(neuron_number_, 0.0));
			ext_inputs_.resize(neuron_number_);
			synaptic_drivens_.resize(neuron_number_);
		}
		
		// INPUTS:
		// Set interneuronal coupling strength;
		void InitializeSynapticStrength(po::variables_map &vm);
		// Initialize the delay of synaptic interaction;
		void InitializeSynapticDelay(po::variables_map &vm);
		
		// Set interaction delay between neurons;
		void SetDelay(vector<vector<double> > &coordinates, double speed);

		// Set time period of refractory:
		void SetRef(double t_ref);

		//	Set driving type: true for external Poisson driven, false for internal ones;
		void SetDrivingType(bool driving_type);

		//	Initialize internal homogeneous feedforward Poisson rate;
		void InitializePoissonGenerator(po::variables_map &vm);

		// Initialize interface for raster data;
		void InitRasterOutput(string ras_path);

		// 	Input new spikes for neurons all together;
		//void InNewSpikes(vector<vector<Spike> > &data);

		// DYNAMICS:
		// Inject Poisson sequence from ext_inputs_ to synaptic_drivens_, autosort after generatation if synaptic delay is nonzero;
		// tmax: maximum time of Poisson sequence;
		// return: none;
		void InjectPoisson(double tmax);

		//	Inject synaptic inputs, either feedforward or interneuronal ones, autosort after insertion;
		void InjectSpike(Spike x, int id);

		//	NewSpike: record new spikes for id-th neurons which fire at t = t + dt;
		void NewSpike(int id, double t, double spike_time);
		void NewSpike(int id, double t, vector<double>& spike_times);

		// Clean used synaptic inputs:
		void CleanUsedInputs(double tmax);


		//	Restore neuronal state for all neurons, including neuronal potential, conductances, refractory periods and external network drive;
		void RestoreNeurons();

		// OUTPUTS:

		//	Output potential to *.csv file;
		void OutPotential(FILEWRITE& file);

		//	Output synaptic conductance to *.csv files:
		//		BOOL function: function of synaptic conductance, true for excitation, false for inhibition;
		void OutConductance(FILEWRITE& file, bool function);

		//	Output current to *.csv file;
		void OutCurrent(FILEWRITE& file);

		// Save connectivity matrix
		void SaveConMat(string connecting_matrix_file);

		int GetNeuronNumber();

		double GetConductance(int i, bool function);

};
#endif // _NEURON_POPULATION_H_
