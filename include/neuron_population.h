//***************
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Date: 2019-11-03
//	Description: define template class NeuronPopulation;
//***************
#ifndef _NEURON_POPULATION_H_
#define _NEURON_POPULATION_H_

#include "io.h"
#include "neuron.h"
#include "poisson_generator.h"
#include "common_header.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>
namespace po = boost::program_options;

using namespace std;

struct SpikeElement {
	int index;	// The sequence order of spikes within single time interval;
	double t;		// exact spiking time;
	bool type;	// The type of neuron that fired;
	SpikeElement() : index(-1), t(dnan), type(false) {	}
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
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> TyConMat; 


inline double L2(vector<double> &x, vector<double> &y) {	
	return sqrt((x[0] - y[0])*(x[0] - y[0]) + (x[1] - y[1])*(x[1] - y[1]));
}

class NeuronPopulationBase {
	public:
		// INPUTS:
		virtual void InitializeSynapticStrength(po::variables_map &vm) = 0;
		virtual void InitializeSynapticDelay(po::variables_map &vm) = 0;
		virtual void SetDelay(vector<vector<double> > &coordinates, double speed) = 0;
		virtual void SetRef(double t_ref) = 0;
		virtual void InitializePoissonGenerator(po::variables_map &vm) = 0;
		virtual void InitRasterOutput(string ras_path) = 0;
		virtual void CloseRasterOutput() = 0;
		// DYNAMICS:
		//virtual double UpdateNeuronalState(double *dym_val, TyNeuronalInput &synaptic_driven, double t, double dt, vector<double>& new_spikes) const = 0;
		virtual void UpdateNeuronalState(bool dym_flag, int i, double t, double dt, vector<double>& new_spikes) = 0;
		virtual void UpdateConductance(bool dym_flag, int i, double t, double dt) = 0;
		virtual void SynapticInteraction(SpikeElement & x, vector<int> & post_ids) = 0;
		//	Inject synaptic inputs, either feedforward or interneuronal ones, autosort after insertion;
		virtual void InjectSpike(Spike x, int id) = 0;
		virtual void InjectPoisson(double tmax) = 0;  // tmax: maximum time of Poisson sequence;
		virtual void GeneratePoisosn(double time) = 0; // Generate new Poisson sequence up to time

		virtual void BackupDymVal() = 0;
		virtual void UpdateDymVal() = 0;
		// Clean used synaptic inputs:
		virtual void CleanUsedInputs(double tmax) = 0;

		// Get Parameter interface:
		virtual double * GetDymPtr(int id) = 0;
		virtual double * GetTmpPtr(int id) = 0;
		virtual const double GetDymVal(int id, int dym_id) const = 0;
		virtual const int GetDymN() const = 0;
		virtual const int GetNeuronNumber() const = 0;
		virtual const int GetNe() const = 0;
		virtual const int GetNi() const = 0;
		virtual const bool GetPoissonMode() const = 0;
		virtual const bool GetIsCon() const = 0;

		// OUTPUTS : Output data to *.csv files:
		virtual void NewSpike(int id, double t, double spike_time) = 0;
		virtual void OutPotential(FILEWRITE& file) = 0;
		virtual void OutConductance(FILEWRITE& file, bool function) = 0;
		virtual void OutCurrent(FILEWRITE& file) = 0;

		//	Restore neuronal state for all neurons, including neuronal potential, conductances, refractory periods and external network drive;
		virtual void RestoreNeurons() = 0;
		virtual ~NeuronPopulationBase() {  }
};

// class to containing neuronal data
template <class NeuronSimulator>
class NeuronPopulationNoExtCurrent : public NeuronSimulator, public NeuronPopulationBase {
	private:
		using NeuronSimulator::GetDymNum;
		using NeuronSimulator::GetIDV;
		using NeuronSimulator::GetIDGE;
		using NeuronSimulator::GetIDGI;
		using NeuronSimulator::UpdateNeuronalState;
		using NeuronSimulator::UpdateConductance;
		using NeuronSimulator::GetDefaultDymVal;
		using NeuronSimulator::GetCurrent;
	public:
		int dym_n_;
		// Network Parameters:
		int neuron_number_;		// number of the neurons in the group;
		int Ne_;		// number of excitatory neurons;
		TyDymVals dym_vals_;		// dynamic variables of neurons;
		TyDymVals dym_vals_tmp_;// buffer for dynamic variables of neurons;

		// PoissonGenerators:
		vector<PoissonGenerator> pge_;
		vector<PoissonGenerator> pgi_;
		bool pg_mode;

		// Network Structure:
		bool is_con_;
		TyConMat s_mat_;									// matrix of inter-neuronal interacting strength;
		vector<vector<double> > delay_mat_;

		// Network Inputs:
		vector<PoissonSeq> ext_inputs_; // temp storage of external Poisson input;
		TyNeuronalInputVec synaptic_drivens_;

		// Data output interface:
		ofstream raster_file_;

		// Neuronal network initialization:
		NeuronPopulationNoExtCurrent(int Ne, int Ni) {
			// Network Parameters:
			dym_n_ = GetDymNum();
			Ne_ = Ne;
			neuron_number_ = Ne + Ni;
			dym_vals_.resize(neuron_number_, dym_n_);
			dym_vals_tmp_.resize(neuron_number_, dym_n_);
			for (int i = 0; i < neuron_number_; i++) {
				GetDefaultDymVal(GetDymPtr(i));
			}
			pge_.resize(neuron_number_);
			pgi_.resize(neuron_number_);
			// Network structure:
			is_con_ = false;
			s_mat_.resize(neuron_number_, neuron_number_);
			delay_mat_.resize(neuron_number_, vector<double>(neuron_number_, 0.0));
			ext_inputs_.resize(neuron_number_);
			synaptic_drivens_.resize(neuron_number_);
		}
		
		// INPUTS:
		// Set interneuronal coupling strength;
		void InitializeSynapticStrength(po::variables_map &vm) override {
			typedef Eigen::Triplet<double> T;
			vector<T> T_list;
			string sfname = vm["prefix"].as<string>() + vm["synapse.file"].as<string>();
			auto s_vals = xt::load_npy<double>(sfname.c_str());
			for (size_t i = 0; i < neuron_number_; i ++) {
				for (size_t j = 0; j < neuron_number_; j ++) {
					if (s_vals(i,j) > 0) T_list.push_back(T(i,j,s_vals(i,j)));
				}
			}
			s_mat_.setFromTriplets(T_list.begin(), T_list.end());
			s_mat_.makeCompressed();
			printf("(number of connections in sparse-mat %d)\n", (int)s_mat_.nonZeros());
			if (s_mat_.nonZeros()) is_con_ = true;
		}
		// Initialize the delay of synaptic interaction;
		void InitializeSynapticDelay(po::variables_map &vm) override {
			int space_mode = vm["space.mode"].as<int>();
			if (space_mode == 0) {
				vector<vector<double> > coordinates;
				Read2D(vm["prefix"].as<string>() + vm["space.file"].as<string>(), coordinates);
				SetDelay(coordinates, vm["space.speed"].as<double>());
			} else if (space_mode == 1) {
				delay_mat_.clear();
				delay_mat_.resize(neuron_number_, vector<double>(neuron_number_, vm["space.delay"].as<double>()));
			} else if (space_mode == -1) {
				delay_mat_.clear();
				delay_mat_.resize(neuron_number_, vector<double>(neuron_number_, 0.0));
			}
		}	
		// Set interaction delay between neurons;
		void SetDelay(vector<vector<double> > &coordinates, double speed) override {
			double meta_dis;
			for (int i = 0; i < neuron_number_; i ++) {
				for (int j = 0; j < i; j ++) {
					meta_dis = L2(coordinates[i], coordinates[j]) / speed;
					delay_mat_[i][j] = meta_dis;
					delay_mat_[j][i] = meta_dis;
				}
			}
		}

		// Set time period of refractory:
		void SetRef(double t_ref) override {
			NeuronSimulator::SetRef(t_ref);
		}

		//	Initialize internal homogeneous feedforward Poisson rate;
		void InitializePoissonGenerator(po::variables_map &vm) override {
			vector<vector<double> > poisson_settings;
			//	poisson_setting: 
			//		[:,0] excitatory Poisson rate;
			//		[:,1] excitatory Poisson strength;
			//		[:,2] inhibitory Poisson rate;
			//		[:,3] inhibitory Poisson strength;
			// import the data file of feedforward driving rate:
			Read2D(vm["prefix"].as<string>() + vm["driving.file"].as<string>(), poisson_settings);
			if (poisson_settings.size() != neuron_number_) {
				cout << "Error inputing length! (Not equal to the number of neurons in the net)";
				return;
			}
			bool poisson_output = vm["output.poi"].as<bool>();
			for (int i = 0; i < neuron_number_; i++) {
				pge_[i].SetRate(poisson_settings[i][0]);
				pge_[i].SetStrength(poisson_settings[i][1]);
				pgi_[i].SetRate(poisson_settings[i][2]);
				pgi_[i].SetStrength(poisson_settings[i][3]);
				if (poisson_output) {
					pge_[i].SetOuput( vm["prefix"].as<string>() + "pge" + to_string(i) + ".csv" );
					pgi_[i].SetOuput( vm["prefix"].as<string>() + "pgi" + to_string(i) + ".csv" );
				}
			}
			//	Set driving type: true for external Poisson driven, false for internal ones;
			pg_mode = vm["driving.gmode"].as<bool>();

			if ( pg_mode ) {
				for (int i = 0; i < neuron_number_; i ++) {
					pge_[i].GenerateNewPoisson( true, vm["time.T"].as<double>(), ext_inputs_[i] );
					pgi_[i].GenerateNewPoisson( false, vm["time.T"].as<double>(), ext_inputs_[i] );
				}
			}
		}
		// Initialize interface for raster data;
		void InitRasterOutput(string ras_path) override {
			raster_file_.open(ras_path.c_str());	
		}
		void CloseRasterOutput() override {
			raster_file_.close();	
		}

		// DYNAMICS:
		void UpdateNeuronalState(bool dym_flag, int i, double t, double dt, vector<double>& new_spikes) override {
			if (dym_flag) {
				UpdateNeuronalState(GetDymPtr(i), synaptic_drivens_[i], t, dt, new_spikes);
			} else {
				UpdateNeuronalState(GetTmpPtr(i), synaptic_drivens_[i], t, dt, new_spikes);
			}
		}

		void UpdateConductance(bool dym_flag, int i, double t, double dt) override {
			if (dym_flag) {
				UpdateConductance(GetDymPtr(i), synaptic_drivens_[i], t, dt);
			} else {
				UpdateConductance(GetTmpPtr(i), synaptic_drivens_[i], t, dt);
			}
		}
		void SynapticInteraction(SpikeElement & x, vector<int> & post_ids) override {
			post_ids.clear();
			Spike new_spike;
			new_spike.type = x.type;
			int IND = x.index;
			for (TyConMat::InnerIterator it(s_mat_, IND); it; ++it) {
				new_spike.s = it.value();
				new_spike.t = x.t + delay_mat_[it.index()][IND];
				InjectSpike(new_spike, it.index());
				NEURON_INTERACTION_TIME ++;
				post_ids.push_back(it.index());
			}
		}

		//	Inject synaptic inputs, either feedforward or interneuronal ones, autosort after insertion;
		inline void InjectSpike(Spike x, int id) override {
			synaptic_drivens_[id].Inject(x);
		}

		// Inject Poisson sequence from ext_inputs_ to synaptic_drivens_, autosort after generatation if synaptic delay is nonzero;
		// tmax: maximum time of Poisson sequence;
		// return: none;
		void InjectPoisson(double tmax) override {
			for (int i = 0; i < neuron_number_; i ++) {
				if ( !ext_inputs_[i].empty() ) {
					while ( ext_inputs_[i].top().t < tmax ) {
						InjectSpike(ext_inputs_[i].top(), i);
						ext_inputs_[i].pop();
						if ( ext_inputs_[i].empty() ) break;
					}
				}
			}
		}

		void GeneratePoisosn(double time) override {
			for (int i = 0; i < neuron_number_; i ++) {
				pge_[i].GenerateNewPoisson( true, time, ext_inputs_[i]);
				pgi_[i].GenerateNewPoisson( false, time, ext_inputs_[i]);
			}
		}

		//	NewSpike: record new spikes for id-th neurons which fire at t = t + dt;
		inline void NewSpike(int id, double t, double spike_time) override {
			raster_file_ << (int)id << ',' << setprecision(18) << (double)(t+spike_time) << '\n';
			SPIKE_NUMBER ++;
			if (t+spike_time==dnan) {
				printf("Invalid spike time with neuron ID = %d, t = %d, SPIKE_NUMBER = %d\n", id, t, SPIKE_NUMBER);
			}
		}
		//void NewSpike(int id, double t, vector<double>& spike_times);

		// copy from dym_vals_ to dym_vals_tmp_
		inline void BackupDymVal() override {
			memcpy(dym_vals_tmp_.data(), dym_vals_.data(), sizeof(double)*neuron_number_*dym_n_);
		}
		// copy from dym_vals_tmp_ to dym_vals_
		inline void UpdateDymVal() override {
			memcpy(dym_vals_.data(), dym_vals_tmp_.data(), sizeof(double)*neuron_number_*dym_n_);
		}

		// Clean used synaptic inputs:
		void CleanUsedInputs(double tmax) override {
			for (int i = 0; i < neuron_number_; i ++) {
				synaptic_drivens_[i].Move(tmax);
				// clean old synaptic driven;
				synaptic_drivens_[i].Clear();
			}
		}

		//	Restore neuronal state for all neurons, including neuronal potential, conductances, refractory periods and external network drive;
		void RestoreNeurons() override {
			for (int i = 0; i < neuron_number_; i++) {
				GetDefaultDymVal(GetDymPtr(i));
				pge_[i].Reset();
				pgi_[i].Reset();
			}
			ext_inputs_.clear();
			ext_inputs_.resize(neuron_number_);
			synaptic_drivens_.clear();
			synaptic_drivens_.resize(neuron_number_);
		}

		// OUTPUTS:
		//	Output potential to *.csv file;
		void OutPotential(FILEWRITE& file) override {
			vector<double> potential(neuron_number_);
			int id = GetIDV();
			for (int i = 0; i < neuron_number_; i++) {
				potential[i] = dym_vals_(i, id);
			}
			file.Write(potential);
		}

		//		BOOL function: function of synaptic conductance, true for excitation, false for inhibition;
		void OutConductance(FILEWRITE& file, bool function) override {
			vector<double> conductance(neuron_number_);
			int id;
			if (function) {
				id = GetIDGE();
				for (int i = 0; i < neuron_number_; i++) {
					conductance[i] = dym_vals_(i, id);
				}
			} else {
				id = GetIDGI();
				for (int i = 0; i < neuron_number_; i++) {
					conductance[i] = dym_vals_(i, id);
				}
			}
			file.Write(conductance);
		}

		void OutCurrent(FILEWRITE& file) override {
			vector<double> current(neuron_number_);
			for (int i = 0; i < neuron_number_; i++) {
				current[i] = GetCurrent(GetDymPtr(i));
			}
			file.Write(current);
		}

		inline double* GetDymPtr(int id) override {
			return dym_vals_.data() + id * dym_vals_.cols();
		}
		inline double* GetTmpPtr(int id) override {
			return dym_vals_tmp_.data() + id * dym_vals_tmp_.cols();
		}
		inline const double GetDymVal(int id, int dym_id) const override {
			return dym_vals_(id, dym_id);
		}
		inline const int GetDymN() const override {
			return dym_n_;
		}
		inline const int GetNeuronNumber() const override {
			return neuron_number_;
		}
		inline const int GetNe() const override{
			return Ne_;
		}
		inline const int GetNi() const override {
			return neuron_number_ - Ne_;
		}
		inline const bool GetPoissonMode() const override {
			return pg_mode;
		}
		inline const bool GetIsCon() const override {
			return is_con_;
		}

};

typedef NeuronPopulationNoExtCurrent<Sim_LIF_G>  Ty_Neuron_Pop_LIF_G_NO_EXT;
typedef NeuronPopulationNoExtCurrent<Sim_LIF_GH> Ty_Neuron_Pop_LIF_GH_NO_EXT;
typedef NeuronPopulationNoExtCurrent<Sim_LIF_I>  Ty_Neuron_Pop_LIF_I_NO_EXT;

// class to containing neuronal data
//template <class NeuronSimulator>
class NeuronPopulationSine : public Sim_LIF_GH_Sine, public Ty_Neuron_Pop_LIF_GH_NO_EXT {
	using Sim_LIF_GH_Sine::GetDymNum;
	using Sim_LIF_GH_Sine::GetDefaultDymVal;
	using Sim_LIF_GH_Sine::GetIDV;
	using Sim_LIF_GH_Sine::GetIDGE;
	using Sim_LIF_GH_Sine::GetIDGI;
	using Sim_LIF_GH_Sine::GetCurrent;
	public:
	using Sim_LIF_GH_Sine::UpdateNeuronalState;
		typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> TySinePara; 
		TySinePara sin_para;

		void SetSineAmplitude(double amp_val) {
			for (size_t i = 0; i < neuron_number_; i ++) {
				sin_para(i,0) = amp_val;
			}
		}
		void SetSineFrequency(double freq_val) {
			for (size_t i = 0; i < neuron_number_; i ++) {
				sin_para(i,1) = freq_val;
			}
		}
		void SetSinePhase(double phase_val) {
			for (size_t i = 0; i < neuron_number_; i ++) {
				sin_para(i,2) = phase_val;
			}
		}

		void SetSineAmplitude(int id, double amp_val) {
			sin_para(id,0) = amp_val;
		}
		void SetSineFrequency(int id, double freq_val) {
			sin_para(id,1) = freq_val;
		}
		void SetSinePhase(int id, double phase_val) {
			sin_para(id,2) = phase_val;
		}

		//void SetSineAmplitude(vector<double> & amp_val) {
		//	//sin_para.col(0) += amp_val;
		//	for (size_t i = 0; i < neuron_number_; i ++) {
		//		sin_para(i,0) = amp_val[i];
		//	}
		//}
		//void SetSineFrequency(vector<double> & freq_val) {
		//	for (size_t i = 0; i < neuron_number_; i ++) {
		//		sin_para(i,1) = freq_val[i];
		//	}
		//}
		//void SetSinePhase(vector<double> & phase_val) {
		//	for (size_t i = 0; i < neuron_number_; i ++) {
		//		sin_para(i,2) = phase_val[i];
		//	}
		//}

		// Neuronal network initialization:
		NeuronPopulationSine(int Ne, int Ni) : Ty_Neuron_Pop_LIF_GH_NO_EXT(Ne, Ni) {
			// Sine Network Parameters:
			sin_para.resize(Ne+Ni, 4);
			for (int i=0; i<Ne+Ni; i++) {
				sin_para(i, 0) = 0;
				sin_para(i, 1) = 0;
				sin_para(i, 2) = 0;
				//2*M_PI / n_neurons() * j;
			}
		}

		// DYNAMICS:
		void UpdateNeuronalState(bool dym_flag, int i, double t, double dt, vector<double>& new_spikes) override {
			double *d = &sin_para(i, 0);
			if (dym_flag) {
				UpdateNeuronalState(GetDymPtr(i), synaptic_drivens_[i], d, t, dt, new_spikes);
			} else {
				UpdateNeuronalState(GetTmpPtr(i), synaptic_drivens_[i], d, t, dt, new_spikes);
			}
		}

		//	Restore neuronal state for all neurons, including neuronal potential, conductances, refractory periods and external network drive;
		void RestoreNeurons() override {
			Ty_Neuron_Pop_LIF_GH_NO_EXT::RestoreNeurons();
			for (int i=0; i<neuron_number_; i++) {
				sin_para(i, 0) = 0;
				sin_para(i, 1) = 0;
				sin_para(i, 2) = 0;
				//2*M_PI / n_neurons() * j;
			}
		}

		void OutCurrent(FILEWRITE& file) override {  }

};
#endif // _NEURON_POPULATION_H_
