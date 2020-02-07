//***************
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Date: 2019-06-09
//	Description: define Struct SpikeElement and Class Network;
//***************
#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "common_header.h"
#include "neuron.h"
#include "neuron_population.h"
using namespace std;

class NetworkSimulatorBase {
	public:
	virtual void UpdateState(NeuronPopulation *neuron_pop, double t, double dt) const = 0;
	virtual ~NetworkSimulatorBase() {  }
};

// TODO: numerical convergence need to be checked;
class NetworkSimulatorSimple : public NetworkSimulatorBase {
	public:
		void UpdateState(NeuronPopulation *neuron_pop, double t, double dt) const override {
			if ( !neuron_pop->pg_mode ) {
				for (int i = 0; i < neuron_pop->neuron_number_; i ++) {
					neuron_pop->pge_[i].GenerateNewPoisson(true, t + dt, neuron_pop->ext_inputs_[i]);
					neuron_pop->pgi_[i].GenerateNewPoisson(false, t + dt, neuron_pop->ext_inputs_[i]);
				}
			}
			// inject poisson
			neuron_pop->InjectPoisson(t + dt);

			if (neuron_pop->is_con_) {
				vector<SpikeElement> T;
				vector<int> update_list, fired_list;
				for (int i = 0; i < neuron_pop->neuron_number_; i++) update_list.push_back(i);
				//SortSpikes(neuron_pop, dym_vals_new, update_list, fired_list, t, dt, T);
				
				vector<double> tmp_spikes;
				for (int i = 0; i < neuron_pop->neuron_number_; i++) {
					neuron_pop->neuron_sim_->UpdateNeuronalState(GetPtr(neuron_pop->dym_vals_, i), neuron_pop->synaptic_drivens_[i], t, dt, tmp_spikes);
					for (auto it = tmp_spikes.begin(); it != tmp_spikes.end(); it++) {
						if (i < neuron_pop->Ne_) {
							T.emplace_back(i, *it, true);
						} else {
							T.emplace_back(i, *it, false);
						}
					}
				}

				for (auto iter = T.begin(); iter != T.end(); iter++ ) {
					int IND = iter->index;
					Spike ADD_mutual;
					ADD_mutual.type = iter->type;
					// erase used spiking events;
					neuron_pop->NewSpike(IND, t, iter->t);
					for (TyConMat::InnerIterator it(neuron_pop->s_mat_, IND); it; ++it) {
						ADD_mutual.s = it.value();
						// Force the interneuronal interaction to the end of the time step
						ADD_mutual.t = t + dt + neuron_pop->delay_mat_[it.index()][IND];
						neuron_pop->InjectSpike(ADD_mutual, it.index());
						NEURON_INTERACTION_TIME ++;
					}
				}
				neuron_pop->CleanUsedInputs(t + dt);
			} else {
				vector<vector<double> > new_spikes(neuron_pop->neuron_number_);
				#pragma omp parallel for
				for (int i = 0; i < neuron_pop->neuron_number_; i++) {
					neuron_pop->neuron_sim_->UpdateNeuronalState(GetPtr(neuron_pop->dym_vals_, i), neuron_pop->synaptic_drivens_[i], t, dt, new_spikes[i]);
					if ( !new_spikes[i].empty() ) neuron_pop->NewSpike(i, t, new_spikes[i]);
				}
				neuron_pop->CleanUsedInputs(t + dt);
			}
		}
};

class NetworkSimulatorSSC : public NetworkSimulatorBase {
	private:
		bool CheckExist(int index, vector<int> &list) const {
			for (int i = 0; i < list.size(); i ++) {
				if (list[i] == index) return true;
			}
			return false;
		}
		// Sort spikes within single time interval, and return the time of first spike;
		double SortSpikes(NeuronPopulation *neuron_pop, TyDymVals &dym_vals_new, vector<int> &update_list, vector<int> &fired_list, double t, double dt, vector<SpikeElement> &T) const {
			vector<double> tmp_spikes;
			// start scanning;
			double id;
			for (int i = 0; i < update_list.size(); i++) {
				id = update_list[i];
				// Check whether id's neuron is in the fired list;
				if (CheckExist(id, fired_list)) {
					memcpy(GetPtr(dym_vals_new, id)+1, GetPtr(neuron_pop->dym_vals_, id)+1, sizeof(double)*(neuron_pop->dym_n_-2));
					neuron_pop->neuron_sim_->UpdateConductance(GetPtr(dym_vals_new,id), neuron_pop->synaptic_drivens_[id], t, dt);
				} else {
					memcpy(GetPtr(dym_vals_new, id), GetPtr(neuron_pop->dym_vals_, id), sizeof(double)*neuron_pop->dym_n_);
					neuron_pop->neuron_sim_->UpdateNeuronalState(GetPtr(dym_vals_new, id), neuron_pop->synaptic_drivens_[id], t, dt, tmp_spikes);
					if (!tmp_spikes.empty()) {
						if (id < neuron_pop->Ne_) {
							T.emplace_back(id, tmp_spikes.front(), true);
						} else {
							T.emplace_back(id, tmp_spikes.front(), false);
						}
					}
				}
			}
			if (T.empty()) {
				return -1;
			} else if (T.size() == 1) {
				return (T.front()).t;
			} else {
				sort(T.begin(), T.end(), std::greater<SpikeElement>() );
				return (T.front()).t;
			}
		}

	public:
		//	Update network state:
		void UpdateState(NeuronPopulation *neuron_pop, double t, double dt) const override {
			if ( !neuron_pop->pg_mode ) {
				for (int i = 0; i < neuron_pop->neuron_number_; i ++) {
					neuron_pop->pge_[i].GenerateNewPoisson(true, t + dt, neuron_pop->ext_inputs_[i]);
					neuron_pop->pgi_[i].GenerateNewPoisson(false, t + dt, neuron_pop->ext_inputs_[i]);
				}
			}
			// inject poisson
			neuron_pop->InjectPoisson(t + dt);

			if (neuron_pop->is_con_) {
				TyDymVals dym_vals_new(neuron_pop->neuron_number_, neuron_pop->dym_n_);
				memcpy(dym_vals_new.data(), neuron_pop->dym_vals_.data(), sizeof(double)*neuron_pop->neuron_number_*neuron_pop->dym_n_);
				vector<SpikeElement> T;
				double newt;
				// Creating updating pool;
				vector<int> update_list, fired_list;
				for (int i = 0; i < neuron_pop->neuron_number_; i++) update_list.push_back(i);
				newt = SortSpikes(neuron_pop, dym_vals_new, update_list, fired_list, t, dt, T);
				while (newt > 0) {
					update_list.clear();
					int IND = (T.front()).index;
					fired_list.push_back(IND);
					Spike ADD_mutual;
					ADD_mutual.type = (T.front()).type;
					// erase used spiking events;
					T.erase(T.begin());
					neuron_pop->NewSpike(IND, t, newt);
					for (TyConMat::InnerIterator it(neuron_pop->s_mat_, IND); it; ++it) {
						ADD_mutual.s = it.value();
						ADD_mutual.t = t + newt + neuron_pop->delay_mat_[it.index()][IND];
						neuron_pop->InjectSpike(ADD_mutual, it.index());
						NEURON_INTERACTION_TIME ++;
						update_list.push_back(it.index());
						// Check whether this neuron appears in the firing list T;
						for (int k = 0; k < T.size(); k ++) {
							if (it.index() == T[k].index) {
								T.erase(T.begin() + k);
								break;
							}
						}
					}
					newt = SortSpikes(neuron_pop, dym_vals_new, update_list, fired_list, t, dt, T);
				}
				memcpy(neuron_pop->dym_vals_.data(), dym_vals_new.data(), sizeof(double)*neuron_pop->neuron_number_*neuron_pop->dym_n_);
				neuron_pop->CleanUsedInputs(t + dt);
			} else {
				vector<vector<double> > new_spikes(neuron_pop->neuron_number_);
				#pragma omp parallel for
				for (int i = 0; i < neuron_pop->neuron_number_; i++) {
					neuron_pop->neuron_sim_->UpdateNeuronalState(GetPtr(neuron_pop->dym_vals_, i), neuron_pop->synaptic_drivens_[i], t, dt, new_spikes[i]);
					if ( !new_spikes[i].empty() ) neuron_pop->NewSpike(i, t, new_spikes[i]);
				}
				neuron_pop->CleanUsedInputs(t + dt);
			}
		}
};

#endif // _NETWORK_H_
