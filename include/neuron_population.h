// ===============
//  Copyright: Kyle Chen
//  Author: Kyle Chen
//  Created: 2019-06-09
//  Description: define template class NeuronPopulation;
// ===============
#ifndef _NEURON_POPULATION_H_
#define _NEURON_POPULATION_H_

#include "io.h"
#include "neuron.h"
#include "poisson_generator.h"
#include "common_header.h"
namespace po = boost::program_options;

struct SpikeElement {
  int index;  // The sequence order of spikes within single time interval;
  double t;   // exact spiking time;
  bool type;  // The type of neuron that fired;
  SpikeElement() : index(-1), t(dNaN), type(false) {  }
  SpikeElement(int index_val, double t_val, bool type_val)
   : index(index_val), t(t_val), type(type_val) {  }

  bool operator < (const SpikeElement &b) const
  { return t < b.t; }
  bool operator > (const SpikeElement &b) const
  { return t > b.t; }
  bool operator == (const SpikeElement &b) const
  { return t == b.t && type == b.type; }
};

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> TyDymVals; 
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> TySparseAdjMat; 
typedef std::vector<TyPoissonInput> TyPoissonInputVec;

inline double* GetPtr(TyDymVals &mat, int id) {
	return mat.data() + id * mat.cols();
}

// class to containing neuronal data
class NeuronPopulation {
	public:
		// Neuron Simulators:
		NeuronBase *neuron_sim_ = NULL;
		int dym_n_;
	
		// Network Parameters:
		int neuron_number_;		// number of the neurons in the group;
		int Ne_;		// number of excitatory neurons;
		TyDymVals dym_vals_;		// dynamic variables of neurons;

		// Poisson-based network inputs:
    TyPoissonInputVec inputs_vec_;

		// Network Structure:
		bool is_con_;
		TySparseAdjMat s_mat_;									// matrix of inter-neuronal interacting strength;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> TyDelayMatrix;
    TyDelayMatrix delay_mat_;

		// Data output interface:
		ofstream raster_file_;

		NeuronPopulation(std::string neuron_type, int Ne, int Ni) {
			// Network Parameters:
			if (neuron_type == "LIF_G") {
				neuron_sim_ = new LIF_G();
			} else if (neuron_type == "LIF_GH") {
				neuron_sim_ = new LIF_GH();
			} else if (neuron_type == "LIF_I") {
				neuron_sim_ = new LIF_I();
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
			// Network structure:
			is_con_ = false;
			s_mat_.resize(neuron_number_, neuron_number_);
			delay_mat_ = TyDelayMatrix::Zero(neuron_number_, neuron_number_);
		}
		~NeuronPopulation() { delete neuron_sim_; }
		
		// INPUTS:
		// Set interneuronal coupling strength;
		void InitializeSynapticStrength(po::variables_map &vm);
		// Initialize the delay of synaptic interaction;
		void InitializeSynapticDelay(po::variables_map &vm);
    // TODO: improve the accuracy for large delay period in convergence test.
		
		// Set time period of refractory:
		void SetRef(double t_ref);

		//	Set driving type: true for external Poisson driven, false for internal ones;
		void SetDrivingType(bool driving_type);

		//	Initialize internal homogeneous feedforward Poisson rate;
		void InitializePoissonGenerator(po::variables_map &vm, double buffer_size);

		// Initialize interface for raster data;
		void InitRasterOutput(std::string ras_path);
		void CloseRasterOutput() {
			raster_file_.close();	
		}
		// DYNAMICS:
    // Update single neuron state locally, return GLOBAL spike time in SpikeElement.
    inline void UpdateNeuronStateLocal(int index, double t, double dt, std::vector<SpikeElement>& new_spikes) {
      double t_spike = neuron_sim_->UpdateDymState(GetPtr(dym_vals_,index), dt);
      if (t_spike >= 0) {
        if (index < Ne_) {
          new_spikes.emplace_back(index, t+t_spike, true);
        } else {
          new_spikes.emplace_back(index, t+t_spike, false);
        }
      }
    }

    // Delta interaction:
    inline void DeltaInteraction(Spike spike, int index) {
      if (spike.type) 
        *(GetPtr(dym_vals_, index) + neuron_sim_->GetIDGEInject()) += spike.s;
      else 
        *(GetPtr(dym_vals_, index) + neuron_sim_->GetIDGIInject()) += spike.s;
    }

		//	Inject synaptic inputs, either feedforward or interneuronal ones, autosort after insertion;
		inline void InjectSpike(Spike x, int id) {
			inputs_vec_[id].Inject(x);
		}

		//  export new spikes of id's neurons at t = spike_time to file;
		void NewSpike(int id, double spike_time);
		void NewSpike(std::vector<SpikeElement>& spikes);

		// Clean used synaptic inputs:
    void CleanUsedInputs(double tmax) {
      for (int i = 0; i < neuron_number_; i ++) {
        inputs_vec_[i].CleanAndRefillPoisson(tmax);
      }
    }

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
		void SaveConMat(std::string connecting_matrix_file);

		int GetNeuronNumber();

		double GetConductance(int i, bool function);

};
#endif // _NEURON_POPULATION_H_
