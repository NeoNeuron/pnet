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
#include "cnpy.h"
namespace po = boost::program_options;

struct SpikeTimeId {
  int index;  // The sequence order of spikes within single time interval;
  double t;   // exact spiking time;
  SpikeTimeId() : index(-1), t(dNaN) {  }
  SpikeTimeId(double time, int index_val)
   : index(index_val), t(time) {  }

  bool operator < (const SpikeTimeId &b) const
  { return t < b.t; }
  bool operator > (const SpikeTimeId &b) const
  { return t > b.t; }
  bool operator == (const SpikeTimeId &b) const
  { return t == b.t && index == b.index; }
};

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> TyDymVals; 
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> TySparseAdjMat; 
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> TyDelayMatrix;
typedef std::vector<TyPoissonInput> TyPoissonInputVec;

inline double* GetPtr(TyDymVals &mat, int id) {
	return mat.data() + id * mat.cols();
}

class NeuronPopulationBase {
  public:
    // Dynamics
    virtual void UpdateNeuronStateLocal(int index, double t, double dt, std::vector<SpikeTimeId>& new_spikes) = 0;

    // Initialization
    virtual void InitializeSynapticStrength(po::variables_map& vm) = 0;
    virtual void InitializeSynapticDelay(po::variables_map& vm) = 0;
    virtual void SetRef(double t_ref) = 0;
    virtual void InitializePoissonGenerator(po::variables_map& vm, double buffer_size) = 0;
    virtual void InitRasterOutput(std::string ras_path) = 0;
    virtual void CloseRasterOutput() = 0;

    // Network Manipulation
    virtual void SynapticInteraction(SpikeTimeId& spike) = 0;
    virtual void DeltaInteraction(Spike spike, int index) = 0;
    virtual void InjectSpike(Spike& x, int id) = 0;
    virtual void NewSpike(SpikeTimeId& spike) = 0;
    virtual void CleanUsedInputs(double tmax) = 0;
    virtual void CleanUsedInputs(int index, double tmax) = 0;
    virtual void RestoreNeurons() = 0;
    
    // Outputs:
    virtual void OutPotential(FILEWRITE& file) = 0;
    virtual void OutConductance(FILEWRITE& file, bool type) = 0;
    virtual void OutCurrent(FILEWRITE& file) = 0;
    
    // Get function
    virtual const NeuronBase* GetNeuronModel() const = 0;
    virtual const TySparseAdjMat* GetAdjMat() const = 0;
    virtual const double GetDelay(int i, int j) const = 0;
    virtual double* GetDymPtr(int index) = 0;
    virtual Spike GetSingleSynapticInput(int index, size_t pointer) const = 0;
    virtual const int GetNeuronNumber() const = 0;
    virtual const int GetNe() const = 0;
    virtual const int GetDymN() const = 0;
    virtual double GetConductance(int i, bool type) const = 0;
    virtual const bool GetIsCon() const = 0;

};

// class to containing neuronal data
template<class Neuron>
class NeuronPopulationNoContinuousCurrent:
  public Neuron, public NeuronPopulationBase
{
	public:
		// Neuron Simulators:
		Neuron neuron_sim_;
		int dym_n_;
	
		// Network Parameters:
		int neuron_number_;		// number of the neurons in the group;
		int Ne_;		          // number of excitatory neurons;
		TyDymVals dym_vals_;	// dynamic variables of neurons;

		// Poisson-based network inputs:
    TyPoissonInputVec inputs_vec_;

		// Network Structure:
		bool is_con_;
		TySparseAdjMat s_mat_;      // matrix of inter-neuronal interacting strength;
    TyDelayMatrix delay_mat_;   // matrix of synaptic delay;

		// Data output interface:
		ofstream raster_file_;

		NeuronPopulationNoContinuousCurrent(int Ne, int Ni) {
			dym_n_ = neuron_sim_.GetDymNum();
			Ne_ = Ne;
			neuron_number_ = Ne + Ni;
			dym_vals_.resize(neuron_number_, dym_n_);
			for (int i = 0; i < neuron_number_; i++) {
				neuron_sim_.GetDefaultDymVal(GetDymPtr(i));
			}
			// Network structure:
			is_con_ = false;
			s_mat_.resize(neuron_number_, neuron_number_);
			delay_mat_ = TyDelayMatrix::Zero(neuron_number_, neuron_number_);
		}
		
		// Inputs:
		// Set interneuronal coupling strength;
    void InitializeSynapticStrength(po::variables_map &vm) override {
      typedef Eigen::Triplet<double> T;
      vector<T> T_list;
      string sfname = vm["prefix"].as<string>() + vm["synapse.file"].as<string>();
      auto s_arr = cnpy::npy_load(sfname.c_str());
      double* s_vals = s_arr.data<double>();
      int counter = 0;
      for (int i = 0; i < neuron_number_; i ++) {
        for (int j = 0; j < neuron_number_; j ++) {
          if (s_vals[counter] > 0) T_list.push_back(T(i,j,s_vals[counter]));
          counter += 1;
        }
      }
      s_mat_.setFromTriplets(T_list.begin(), T_list.end());
      s_mat_.makeCompressed();
      if (vm["verbose"].as<bool>()) {
        printf("(number of connections in sparse-mat %d)\n", (int)s_mat_.nonZeros());
      }
      if (s_mat_.nonZeros()) is_con_ = true;
    }

		// Initialize the delay of synaptic interaction;
    // TODO: improve the accuracy for large delay period in convergence test.
    void InitializeSynapticDelay(po::variables_map &vm) override {
      if (vm.count("space.file")) {
        string sfname = vm["prefix"].as<string>() + vm["space.file"].as<string>();
        auto s_arr = cnpy::npy_load(sfname.c_str());
        double* s_vals = s_arr.data<double>();
        memcpy(delay_mat_.data(), s_vals, sizeof(double)*neuron_number_*neuron_number_);
      } else {
        delay_mat_ = TyDelayMatrix::Zero(neuron_number_, neuron_number_);
      }
    }
		
		// Set time period of refractory:
    void SetRef(double t_ref) override {
      neuron_sim_.SetRefTime(t_ref);
    }

		//	Initialize internal homogeneous feedforward Poisson rate;
    void InitializePoissonGenerator(po::variables_map &vm, double buffer_size) override {
      vector<vector<double> > poisson_settings;
      //	poisson_setting: 
      //		[:,0] Exc. Poisson rate;
      //		[:,1] Inh. Poisson rate;
      //		[:,2] Exc. Poisson strength;
      //		[:,3] Inh. Poisson strength;
      // import the data file of feedforward driving rate:
      Read2D(vm["prefix"].as<string>() + vm["driving.file"].as<string>(), poisson_settings);
      if (poisson_settings.size() != neuron_number_) {
        throw runtime_error("Error inputing length! (Not equal to the number of neurons in the net)");
      }
      for (int i = 0; i < neuron_number_; i++) {
        inputs_vec_.emplace_back(poisson_settings[i].data(), poisson_settings[i].data()+2, buffer_size);
      }
      // generate the first episode of Poisson sequence,
      // default episode length used here.
      for (int i = 0; i < neuron_number_; i++) {
        inputs_vec_[i].InitInput();
        inputs_vec_[i].CleanAndRefillPoisson(0.0); // fill the first portion of spikes;
        dbg_printf("number of Poisson spikes in %d's neuron: %ld", i, POISSON_CALL_TIME);
      }
    }

		// Initialize interface for raster data;
    void InitRasterOutput(string ras_path) override {
      raster_file_.open(ras_path.c_str());	
    }
		void CloseRasterOutput() override {
			raster_file_.close();	
		}
		// Dynamics:
    // Update single neuron state locally, return GLOBAL spike time in SpikeTimeId.
    void UpdateNeuronStateLocal(int index, double t, double dt, std::vector<SpikeTimeId>& new_spikes) override {
      if (dt < 0) {
        printf(">> Warning negative dt : id = (%d), t = (%f) dt = (%.2f).\n", index, t, dt);
      }
      double t_spike = neuron_sim_.UpdateDymState(GetDymPtr(index), dt);
      if (t_spike >= 0)
        new_spikes.emplace_back(t+t_spike, index);
    }

    void SynapticInteraction(SpikeTimeId& spike) override {
      Spike spike_buffer;
      spike_buffer.type = spike.index < Ne_ ? true : false;
      for (TySparseAdjMat::InnerIterator iit(s_mat_, spike.index); iit; ++iit) {
        spike_buffer.t = spike.t + delay_mat_(iit.index(), spike.index);
        spike_buffer.s = iit.value();
        InjectSpike(spike_buffer, iit.index());
        NEURON_INTERACTION_TIME ++;
      }
    }

    // Delta interaction:
    void DeltaInteraction(Spike spike, int index) override {
      if (spike.type) 
        *(GetDymPtr(index) + neuron_sim_.GetIDGEInject()) += spike.s;
      else 
        *(GetDymPtr(index) + neuron_sim_.GetIDGIInject()) += spike.s;
    }

		//	Inject synaptic inputs, either feedforward or interneuronal ones, autosort after insertion;
		void InjectSpike(Spike& x, int id) override {
			inputs_vec_[id].Inject(x);
		}

		//  export new spikes of id's neurons at t = spike_time to file;
    void NewSpike(SpikeTimeId& spike) override {
      raster_file_ << (int)spike.index << ',' << setprecision(18) << (double)spike.t << '\n';
      SPIKE_NUMBER ++;
      if (spike.t==dNaN) {
        printf("Invalid spike time with neuron ID = %d, SPIKE_NUMBER = %ld\n", spike.index, SPIKE_NUMBER);
      }
    }

		// Clean used synaptic inputs:
    void CleanUsedInputs(double tmax) override {
      for (int i = 0; i < neuron_number_; i ++) {
        inputs_vec_[i].CleanAndRefillPoisson(tmax);
      }
    }
    void CleanUsedInputs(int index, double tmax) override {
      inputs_vec_[index].CleanAndRefillPoisson(tmax);
    }

		//	Restore neuronal state for all neurons, including neuronal potential, conductances, refractory periods and external network drive;
    void RestoreNeurons() override {
      for (int i = 0; i < neuron_number_; i++) {
        neuron_sim_.GetDefaultDymVal(GetDymPtr(i));
        inputs_vec_[i].InitInput();
        inputs_vec_[i].CleanAndRefillPoisson(0.0); // fill the first portion of spikes;
      }
    }

		// Outputs:

		//	Output potential to *.csv file;
    void OutPotential(FILEWRITE& file) override {
      vector<double> potential(neuron_number_);
      int id = neuron_sim_.GetIDV();
      for (int i = 0; i < neuron_number_; i++) {
        potential[i] = dym_vals_(i, id);
      }
      file.Write(potential);
    }

		//	Output synaptic conductance to *.csv files:
		//	type: true for Exc. conductance, false for Inh. ones;
    void OutConductance(FILEWRITE& file, bool type) override {
      vector<double> conductance(neuron_number_);
      int id;
      if (type) {
        id = neuron_sim_.GetIDGE();
        for (int i = 0; i < neuron_number_; i++) {
          conductance[i] = dym_vals_(i, id);
        }
      } else {
        id = neuron_sim_.GetIDGI();
        for (int i = 0; i < neuron_number_; i++) {
          conductance[i] = dym_vals_(i, id);
        }
      }
      file.Write(conductance);
    }

		//	Output current to *.csv file;
    void OutCurrent(FILEWRITE& file) override {
      vector<double> current(neuron_number_);
      for (int i = 0; i < neuron_number_; i++) {
        current[i] = neuron_sim_.GetCurrent(GetDymPtr(i));
      }
      file.Write(current);
    }

    const NeuronBase* GetNeuronModel() const override {
      return &neuron_sim_;
    }

    const TySparseAdjMat* GetAdjMat() const override {
      return &s_mat_; 
    }

    const double GetDelay(int i, int j) const override {
      return delay_mat_(i, j);
    }

    double* GetDymPtr(int index) override {
      return dym_vals_.data() + index * dym_vals_.cols();
    }

    // TODO: improve the efficiency for accessing synaptic inputs;
    Spike GetSingleSynapticInput(int index, size_t pointer) const override {
      return inputs_vec_[index].At(pointer);
    }

    const int GetNeuronNumber() const override {
      return neuron_number_;
    }

    const int GetNe() const override {
      return Ne_;
    }

    const int GetDymN() const override {
      return dym_n_;
    }

    double GetConductance(int i, bool type) const override {
      int id;
      if (type) id = neuron_sim_.GetIDGE();
      else id = neuron_sim_.GetIDGI();
      return dym_vals_(i, id);
    }

    const bool GetIsCon() const override {
      return is_con_;
    }

};
#endif // _NEURON_POPULATION_H_
