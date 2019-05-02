//******************************
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Description: Define base class of Neuron model and Neuron simulator, structure Spike;
//	Date: 2018-12-13
//******************************
#ifndef _NEURON_H_
#define _NEURON_H_

#include "common_header.h"
#include "math_helper.h"

using namespace std;

struct Spike {
	bool type; // type of spike: true for excitation(AMPA), false for inhibition(GABA);
	double t; // Exact spiking time;
	double s; // strength of spikes;
};

bool compSpike(const Spike &x, const Spike &y);

// Neuron Base:
class NeuronBase {
	public:
		virtual int GetDymNum() const = 0; 
		virtual int GetVID() const = 0;
		virtual int GetGEID() const = 0; 
		virtual int GetGIID() const = 0; 
		virtual int GetTRID() const = 0; 
		virtual double GetRestingPotential() const = 0;
		virtual double GetRefTime() const = 0; 
		virtual double GetCurrent(double* dym_val) const = 0;
		virtual void SetRefTime(double t_ref) = 0;
		virtual void SetConstCurrent(double i_val) = 0;
		virtual void ManuallyFire(double* dym_val) const = 0;
		virtual double DymCore(double *dym_val, double dt) const = 0;
		virtual void UpdateSource(double *dym_val, double dt) const = 0;
		virtual void SetDefaultDymVal(double *dym_val) const = 0;
		virtual ~NeuronBase() {  }
};


// Class Neuron: Based on integrate and fire neuron model;
class LIF_G_Model {
	public:
		// PARAMETERS:
		double tau_e_ = 2.0;	// (ms) time const for excitatory conductance;
		double tau_i_ = 5.0;	// (ms) time const for inhibitory conductance;
		double g_m_ = 5e-2;		// (1/ms) normalized membrane conductance;
		double tau_ = 2.0;		// (ms) refractory Period;
		double resting_potential_ = 0.0;
		double threshold_potential_ = 1.0;
		double excitatory_reversal_potential_ = 14.0 / 3.0;
		double inhibitory_reversal_potential_ = -2.0 / 3.0;
		
		// Temporal Setting:
		// Constant drive:
		double const_I = 0.0;

		// excitatory and inhibitory conductance; evolve precisely with the given expression;
		const int dym_n_ = 4;
		const int v_idx_ = 0;
		const int gE_idx_ = 1;
		const int gI_idx_ = 2;
		const int tr_idx_ = 3;
		// index of remaining refractory period time. if negative, remaining refractory period equals to zero;

		// DYNAMICS:

		//	Purely update conductance after single time step dt;
		//	dym_val: dynamical variables;
		//	dt: time step;
		//	return: none;
		void UpdateG(double *dym_val, double dt) const {
			dym_val[gE_idx_] *= exp( -dt / tau_e_ );
			dym_val[gI_idx_] *= exp( -dt / tau_i_ );
		}

		// ODE govern the dynamic of IF neuron;
		// dym_val: dynamical variables;
		// return: dV/dt, the derivative of V;
		double GetDv(double *dym_val) const {
			return - g_m_ * (dym_val[v_idx_] - resting_potential_)
				- dym_val[gE_idx_] * (dym_val[v_idx_] - excitatory_reversal_potential_)
				- dym_val[gI_idx_] * (dym_val[v_idx_] - inhibitory_reversal_potential_)
				+ const_I;
		}
		
		//	Update the conductance and membrane potential for t = [t_n, t_n + dt];
		//	Description: 4th-order Runge Kutta integration scheme is applied;
		//	*voltage: pointer of voltage, updated after excecution;
		//	dt: size of time step, unit ms;
		//	return: derivative of membrane potential at t = t(n);
		double DymInplaceRK4(double *dym_val, double dt) const {
			double exp_E = exp(-0.5 * dt / tau_e_);
			double exp_I = exp(-0.5 * dt / tau_i_);
			// k1 = GetDv(t_n, v_n);
			// k2 = GetDv(t_n+1/2, v_n + k1*dt / 2);
			// k3 = GetDv(t_n+1/2, v_n + k2*dt / 2);
			// k4 = GetDv(t_n+1, v_n + k3*dt);
			// v_n+1 = v_n + dt/6*(k1 + 2*k2 + 2*k3 + k4);
			double v_n = dym_val[v_idx_];
			double k1, k2, k3, k4;
			k1 = GetDv(dym_val);
			// Update G:
			dym_val[gE_idx_] *= exp_E;
			dym_val[gI_idx_] *= exp_I;
			dym_val[v_idx_] = v_n + 0.5*k1*dt;
			k2 = GetDv(dym_val);
			dym_val[v_idx_] = v_n + 0.5*k2*dt;
			k3 = GetDv(dym_val);
			// Update G:
			dym_val[gE_idx_] *= exp_E;
			dym_val[gI_idx_] *= exp_I;
			dym_val[v_idx_] = v_n + k3*dt;
			k4 = GetDv(dym_val);
			// Get v_n+1;
			dym_val[v_idx_] = v_n + dt / 6 *(k1 + 2 * k2 + 2 * k3 + k4);
			return k1;
		}
};

class LIF_I_Model {
	public:
		// PARAMETERS:
		double tau_e_ = 2.0;	// (ms) time const for exc synaptic current;
		double tau_i_ = 2.0;	// (ms) time const for inh synaptic current;
		double g_m_ = 5e-2;		// (1/ms) normalized membrane conductance;
		double tau_ = 2.0;		// (ms) refractory Period;
		double resting_potential_ = 0.0;	// scaled resting membrane potential;
		double threshold_potential_ = 1.0;// scaled threshold membrane potential;
		
		// Temporal Setting:
		// Constant drive:
		double const_I = 0.0;

		// excitatory and inhibitory conductance; evolve precisely with the given expression;
		const int dym_n_ = 4;
		const int v_idx_ = 0;
		const int gE_idx_ = 1;
		const int gI_idx_ = 2;
		const int tr_idx_ = 3;
		// index of remaining refractory period time. if negative, remaining refractory period equals to zero;

		// DYNAMICS:

		//	Purely update current after single time step dt;
		//	dym_val: dynamical variables;
		//	dt: time step;
		//	return: none;
		void UpdateG(double *dym_val, double dt) const {
			dym_val[gE_idx_] *= exp( -dt / tau_e_ );
			dym_val[gI_idx_] *= exp( -dt / tau_i_ );
		}

		// ODE govern the dynamic of IF neuron;
		// dym_val: dynamical variables;
		// return: dV/dt, the derivative of V;
		double GetDv(double *dym_val) const {
			return - g_m_ * (dym_val[v_idx_] - resting_potential_) 
				+ dym_val[gE_idx_] 
				- dym_val[gI_idx_]
				+ const_I;
		}
		
		//	Update the conductance and membrane potential for t = [t_n, t_n + dt];
		//	Description: 4th-order Runge Kutta integration scheme is applied;
		//	*voltage: pointer of voltage, updated after excecution;
		//	dt: size of time step, unit ms;
		//	return: derivative of membrane potential at t = t(n);
		double DymInplaceRK4(double *dym_val, double dt) const {
			double exp_e = exp(-0.5 * dt / tau_e_);
			double exp_i = exp(-0.5 * dt / tau_i_);
			// k1 = GetDv(t_n, v_n);
			// k2 = GetDv(t_n+1/2, v_n + k1*dt / 2);
			// k3 = GetDv(t_n+1/2, v_n + k2*dt / 2);
			// k4 = GetDv(t_n+1, v_n + k3*dt);
			// v_n+1 = v_n + dt/6*(k1 + 2*k2 + 2*k3 + k4);
			double v_n = dym_val[v_idx_];
			double k1, k2, k3, k4;
			k1 = GetDv(dym_val);
			// Update current:
			dym_val[gE_idx_] *= exp_e;
			dym_val[gI_idx_] *= exp_i;
			dym_val[v_idx_] = v_n + 0.5*k1*dt;
			k2 = GetDv(dym_val);
			dym_val[v_idx_] = v_n + 0.5*k2*dt;
			k3 = GetDv(dym_val);
			// Update current:
			dym_val[gE_idx_] *= exp_e;
			dym_val[gI_idx_] *= exp_i;
			dym_val[v_idx_] = v_n + k3*dt;
			k4 = GetDv(dym_val);
			// Get v_n+1;
			dym_val[v_idx_] = v_n + dt / 6 *(k1 + 2 * k2 + 2 * k3 + k4);
			return k1;
		}
};


// class Neuron_LIF:
// Implement basic operations for sub-timestep dynamics;
template <class NeuronModel>
class Neuron_LIF: public NeuronModel, public NeuronBase {
	using NeuronModel::tau_;		// (ms) refractory Period;
	using NeuronModel::const_I; // Constant current drive;
	using NeuronModel::resting_potential_;
	using NeuronModel::threshold_potential_;
	using NeuronModel::dym_n_;
	using NeuronModel::v_idx_;
	using NeuronModel::gE_idx_;
	using NeuronModel::gI_idx_;
	using NeuronModel::tr_idx_;
	using NeuronModel::UpdateG;
	using NeuronModel::GetDv;
	using NeuronModel::DymInplaceRK4;
	public:
		int GetDymNum() const override { return dym_n_; }
		int GetVID()		const override { return v_idx_; }
		int GetGEID() 	const override { return gE_idx_; }
		int GetGIID() 	const override { return gI_idx_; }
		int GetTRID() 	const override { return tr_idx_; }
		double GetRestingPotential() const override { return resting_potential_; }
		double GetRefTime() const override { return tau_; }
		double GetCurrent(double * dym_val) const override { return GetDv(dym_val); }
		void SetRefTime(double t_ref) override { tau_ = t_ref; }
		void SetConstCurrent(double i_val) override { const_I = i_val; }
		void ManuallyFire(double* dym_val) const override {
			dym_val[v_idx_] = resting_potential_;
			dym_val[tr_idx_] = tau_;
		}
		//	Core operation for updating neuronal state within single timing step dt;
		//	Description: operation to update neuronal state in primary level, including updating conductances, membrane potential and checking spiking events; 
		//	dym_val: array of dynamic variables;
		//  dt: size of time step, unit ms;
		//	return: -1 for no spiking events; otherwise, return relative spiking time respect to the begining of the time step;
		//	Remark: if the input current (strength of synaptic input) is too large, or the neuron are at bursting state, the function might fail;
		double DymCore(double *dym_val, double dt) const override {
			double vn = dym_val[v_idx_];
			// Update conductance;
			double dvn, dv_new;
			double t_spike = -1; // spike time within dt;
			if (dym_val[tr_idx_] <= 0) { // neuron is not in the refractory period;
				dvn = DymInplaceRK4(dym_val, dt);
				// Check whether fire or not;
				if (dym_val[v_idx_] > threshold_potential_) {
					dv_new = GetDv(dym_val);
					t_spike = cubic_hermite_root(dt, vn, dym_val[v_idx_], dvn, dv_new, threshold_potential_);
					dym_val[v_idx_] = resting_potential_;
					// update remaining fractory period
					dym_val[tr_idx_] = tau_ + t_spike - dt;
					// if the refractory period is short enough, the neuron will be reactivate;
					if (dym_val[tr_idx_] < 0) {
						// restore the source (driving current or conductance);
						UpdateG( dym_val, dym_val[tr_idx_] );
						DymInplaceRK4( dym_val, -dym_val[tr_idx_] );
					}
				}	
			} else { // neuron is about to exit the refractory period;
				if (dym_val[tr_idx_] < dt) {
					UpdateG(dym_val, dym_val[tr_idx_]);
					dvn = DymInplaceRK4(dym_val, dt - dym_val[tr_idx_]);
					// Check whether fire or not;
					if (dym_val[v_idx_] >= threshold_potential_) {
						dv_new = GetDv(dym_val);
						t_spike = cubic_hermite_root(dt - dym_val[tr_idx_], vn, dym_val[v_idx_], dvn, dv_new, threshold_potential_);
						dym_val[v_idx_] = resting_potential_;
						// update remaining fractory period
						t_spike += dym_val[tr_idx_];
						dym_val[tr_idx_] = tau_ + t_spike;
					}
				} else { // neuron is in the refractory period;
					UpdateG(dym_val, dt);
				}
				dym_val[tr_idx_] -= dt;
			}
			return t_spike;
		}
		//	Update conductance of fired neuron within single time step dt; it has the same hierachy level as the PrimelyUpdateState(double*, bool, Spike, double, bool);
		//	Description: operation to update neuronal state in primary level, ONE synaptic input most which arrives at the begining of time step;
		//	dym_val: array of dynamic variables;
		//  dt: size of time step, unit millisecond;
		//	return: none;
		void UpdateSource(double *dym_val, double dt) const override { UpdateG(dym_val, dt); }

		void SetDefaultDymVal(double* dym_val) const override {
			dym_val[v_idx_] = 0.0;
			dym_val[gE_idx_] = 0.0;
			dym_val[gI_idx_] = 0.0;
			dym_val[tr_idx_] = -1;
		}

};

typedef Neuron_LIF<LIF_G_Model> LIF_G;
typedef Neuron_LIF<LIF_I_Model> LIF_I;

// Class Neuron: Based on integrate and fire neuron model;
class NeuronSim {
	private:
		// Neuron;
		NeuronBase* p_neuron_ = NULL; 

		// DATA:
		size_t cycle_;	// number of cycle that neuron processed;
		vector<double> spike_train_; // Exact time nodes that neuron fires.
		// Synaptic input received by neuron, including feedforward and interneuronal spikes;
		vector<Spike> synaptic_driven_;  

		// FUNCTIONS:

		// Input external Poisson sequence within each time step, autosort after generatation if synaptic delay is nonzero;
		// tmax: maximum time of Poisson sequence;
		// x: container of external inputing spikes;
		// return: none;
		void InputExternalPoisson(double tmax, queue<Spike> & x);

		public:
		// Initialization of parameters in Neuron;
		NeuronSim(string neuron_type) {
			if (neuron_type == "LIF_I") {
				p_neuron_ = new LIF_I();
			} else if (neuron_type == "LIF_G") {
				p_neuron_ = new LIF_G();
			} else throw runtime_error("ERROR: wrong neuron type");
			cycle_ = 0;
		}

		void SetDefaultDymVal(double *&dym_val);

		// INPUTS:
		// Set refractory period:
		void SetRef(double t_ref) { p_neuron_ -> SetRefTime(t_ref); }

		// Set Constant Current:
		void SetConstDrive(double i_val) { p_neuron_ -> SetConstCurrent(i_val); }

		//	Input synaptic inputs, either feedforward or interneuronal ones, autosort after insertion;
		void InSpike(Spike x);

		// Reset neuron into the condition at zero time point;
		void Reset(double *dym_val);

		// DYNAMICS:

		// 	Update neuronal state:
		//	Description: update neuron within single time step, including its membrane potential, conductances and counter of refractory period;
		//	dym_val: dynamic variables;
		//	double t: time point of the begining of the time step;
		//	double dt: size of time step;
		//	queue<Spike> extPoisson: external Poisson sequence;
		//	vector<double> new_spikes: new spikes generated during dt;
		//	Return: membrane potential at t = t + dt;
		double UpdateNeuronalState(double *dym_val, double t, double dt, queue<Spike> & extPoisson, vector<double>& new_spikes);

		// Clean used synaptic inputs:
		// clean used synaptic inputs and update dym_val with dym_val_new;
		// return the new v;
		double CleanUsedInputs(double *dym_val, double * dym_val_new, double tmax);

		// Purely update conductances for fired neurons;
		void UpdateSource(double *dym_val, double t, double dt);

		//	Fire: update neuronal state for neurons which fire at t = t + dt;
		void Fire(double t, double spike_time);
		void Fire(double t, vector<double>& spike_times);

		// OUTPUTS:

		// Print cycle_:
		size_t GetCycle() {
			cout << cycle_;
			return cycle_;
		}
		// Get last spike: return the time point of latest spiking events;
		double GetLastSpike() { return spike_train_.back(); }

		// Get potential: return the current value of membrane potential;
		double GetPotential(double *dym_val) { return dym_val[p_neuron_ -> GetVID()]; }

		// True return excitatory conductance, false return inhibitory conductance;
		double GetConductance(double *dym_val, bool x) {
			if (x) return dym_val[ p_neuron_ -> GetGEID() ];
			else return dym_val[ p_neuron_ -> GetGIID() ];
		}
		
		double GetCurrent(double *dym_val) { return p_neuron_ -> GetCurrent(dym_val); }

		//	Output spike train
		void OutSpikeTrain(vector<double> & spikes);

		//  Output Spikes after t;
		//  the interacting strength of Spikes are set as default(0.0);
		void GetNewSpikes(double t, vector<Spike> &x);
};

#endif 	// _NEURON_H_
