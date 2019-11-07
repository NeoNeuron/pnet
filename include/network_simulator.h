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
	virtual void UpdateState(NeuronPopulationBase *neuron_pop, double t, double dt) const = 0;
	virtual ~NetworkSimulatorBase() {  }
};

// TODO: numerical convergence need to be checked;
class NetworkSimulatorSimple : public NetworkSimulatorBase {
	public:
		void UpdateState(NeuronPopulationBase *neuron_pop, double t, double dt) const override {
			if ( !neuron_pop->GetPoissonMode() ) {
				neuron_pop->GeneratePoisosn(t+dt);
			}
			// inject poisson
			neuron_pop->InjectPoisson(t + dt);

			if (neuron_pop->GetIsCon()) {
				vector<SpikeElement> T;
				vector<int> update_list, fired_list;
				for (int i = 0; i < neuron_pop->GetNeuronNumber(); i++) update_list.push_back(i);
				//SortSpikes(neuron_pop, dym_vals_new, update_list, fired_list, t, dt, T);
				
				vector<double> tmp_spikes;
				for (int i = 0; i < neuron_pop->GetNeuronNumber(); i++) {
					neuron_pop->UpdateNeuronalState(true, i, t, dt, tmp_spikes);
					for (auto it = tmp_spikes.begin(); it != tmp_spikes.end(); it++) {
						if (i < neuron_pop->GetNe()) {
							T.emplace_back(i, *it, true);
						} else {
							T.emplace_back(i, *it, false);
						}
					}
				}

				vector<int> post_ids;
				for (auto iter = T.begin(); iter != T.end(); iter++ ) {
					int IND = iter->index;
					neuron_pop->NewSpike(IND, t, iter->t);
					iter->t += t;
					neuron_pop->SynapticInteraction(*iter, post_ids);
				}
				neuron_pop->CleanUsedInputs(t + dt);
			} else {
				vector<vector<double> > new_spikes(neuron_pop->GetNeuronNumber());
				#pragma omp parallel for
				for (size_t i = 0; i < neuron_pop->GetNeuronNumber(); i++) {
					neuron_pop->UpdateNeuronalState(true, i, t, dt, new_spikes[i]);
					if ( !new_spikes[i].empty() ) {
						for (size_t j = 0; j < new_spikes[i].size(); j ++) {
							neuron_pop->NewSpike(i, t, new_spikes[i][j]);
						}
					}
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
		double SortSpikes(NeuronPopulationBase *neuron_pop, vector<int> &update_list, vector<int> &fired_list, double t, double dt, vector<SpikeElement> &T) const {
			vector<double> tmp_spikes;
			// start scanning;
			double id;
			int dym_n = neuron_pop->GetDymN();
			for (int i = 0; i < update_list.size(); i++) {
				id = update_list[i];
				// Check whether id's neuron is in the fired list;
				if (CheckExist(id, fired_list)) {
					memcpy(neuron_pop->GetTmpPtr(id)+1, neuron_pop->GetDymPtr(id)+1, sizeof(double)*(dym_n-2));
					neuron_pop->UpdateConductance(false, id, t, dt);
				} else {
					memcpy(neuron_pop->GetTmpPtr(id), neuron_pop->GetDymPtr(id), sizeof(double)*dym_n);
					neuron_pop->UpdateNeuronalState(false, id, t, dt, tmp_spikes);
					if (!tmp_spikes.empty()) {
						if (id < neuron_pop->GetNe()) {
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
		void UpdateState(NeuronPopulationBase *neuron_pop, double t, double dt) const override {
			if ( !neuron_pop->GetPoissonMode() ) {
				neuron_pop->GeneratePoisosn(t+dt);
			}
			// inject poisson
			neuron_pop->InjectPoisson(t + dt);

			if (neuron_pop->GetIsCon()) {
				neuron_pop->BackupDymVal();
				vector<SpikeElement> T;
				double newt;
				// Creating updating pool;
				vector<int> update_list, fired_list;
				for (int i = 0; i < neuron_pop->GetNeuronNumber(); i++) update_list.push_back(i);
				newt = SortSpikes(neuron_pop, update_list, fired_list, t, dt, T);
				while (newt > 0) {
					update_list.clear();
					int IND = (T.front()).index;
					fired_list.push_back(IND);
					vector<int> post_ids;
					neuron_pop->NewSpike(IND, t, newt);
					T.front().t += t;
					neuron_pop->SynapticInteraction(T.front(), post_ids);
					// erase used spiking events;
					T.erase(T.begin());
						
					// Check whether this neuron appears in the firing list T;
					for (size_t i = 0; i < post_ids.size(); i++) {
						for (int k = 0; k < T.size(); k ++) {
							if (post_ids[i] == T[k].index) {
								T.erase(T.begin() + k);
								break;
							}
						}
					}
					update_list.insert(update_list.end(), post_ids.begin(), post_ids.end());
					newt = SortSpikes(neuron_pop, update_list, fired_list, t, dt, T);
				}
				neuron_pop->UpdateDymVal();
				neuron_pop->CleanUsedInputs(t + dt);
			} else {
				vector<vector<double> > new_spikes(neuron_pop->GetNeuronNumber());
				#pragma omp parallel for
				for (size_t i = 0; i < neuron_pop->GetNeuronNumber(); i++) {
					neuron_pop->UpdateNeuronalState(true, i, t, dt, new_spikes[i]);
					if ( !new_spikes[i].empty() ) {
						for (size_t j = 0; j < new_spikes[i].size(); j ++) {
							neuron_pop->NewSpike(i, t, new_spikes[i][j]);
						}
					}
				}
				neuron_pop->CleanUsedInputs(t + dt);
			}
		}
};

#endif // _NETWORK_H_
