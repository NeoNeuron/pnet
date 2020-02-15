// ========================================
// Copyright: Kyle Chen
// Author: Kyle Chen
// Created: 2019-06-09
// Description: define Struct SpikeElement and Class Network;
// ========================================
#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "common_header.h"
#include "neuron.h"
#include "neuron_population.h"

class NetworkSimulatorBase {
	public:
	virtual void UpdatePopulationState(NeuronPopulation *neuron_pop, double t, double dt) = 0;
	virtual ~NetworkSimulatorBase() {  }
};

// TODO: numerical convergence need to be checked;
class NetworkSimulatorSimple : public NetworkSimulatorBase {
	public:
  // TODO: Test the feasibility of pragma omp parallel implementation
				// #pragma omp parallel for
		// 	Update neuronal state:
		//	Description: update neuron within single time step, including its membrane potential, conductances and counter of refractory period;
		//	neuron_pop: pointer of neuron population;
		//	double t: time point of the begining of the time step;
		//	double dt: size of time step;
		//	new_spikes: new spikes generated during dt;
		void UpdateWithoutInteraction(NeuronPopulation* neuron_pop, std::vector<int>& update_list, double t, double dt, std::vector<SpikeElement>& new_spikes) {
			new_spikes.clear();
			double tmax = t + dt;
			double t_spike;
      for (size_t i = 0; i < update_list.size(); i ++) {
        int idx = update_list[i];
        if (std::isnan(neuron_pop->synaptic_drivens_[idx].At().t) || neuron_pop->synaptic_drivens_[idx].At() >= tmax) {
          t_spike = neuron_pop->neuron_sim_->UpdateDymState(GetPtr(neuron_pop->dym_vals_,idx), dt);
          //cycle_ ++;
          if (t_spike >= 0) {
            if (idx < neuron_pop->Ne_) {
              new_spikes.emplace_back(idx, t_spike, true);
            } else {
              new_spikes.emplace_back(idx, t_spike, false);
            }
          }
        } else {
          if (neuron_pop->synaptic_drivens_[idx].At() != t) {
            t_spike = neuron_pop->neuron_sim_->UpdateDymState(GetPtr(neuron_pop->dym_vals_,idx), neuron_pop->synaptic_drivens_[idx].At().t - t);
            //cycle_ ++;
            if (t_spike >= 0) {
              if (idx < neuron_pop->Ne_) {
                new_spikes.emplace_back(idx, t_spike, true);
              } else {
                new_spikes.emplace_back(idx, t_spike, false);
              }
            }
          }
          size_t iter = 0;
          while (true) {
            // Update conductance due to the synaptic inputs;
            if (neuron_pop->synaptic_drivens_[idx].At(iter).type) 
              neuron_pop->dym_vals_(idx, neuron_pop->neuron_sim_->GetIDGEInject()) += neuron_pop->synaptic_drivens_[idx].At(iter).s;
            else 
              neuron_pop->dym_vals_(idx, neuron_pop->neuron_sim_->GetIDGIInject()) += neuron_pop->synaptic_drivens_[idx].At(iter).s;

            if (std::isnan(neuron_pop->synaptic_drivens_[idx].At(iter + 1).t) || neuron_pop->synaptic_drivens_[idx].At(iter + 1) >= tmax) {
              t_spike = neuron_pop->neuron_sim_->UpdateDymState(GetPtr(neuron_pop->dym_vals_,idx), tmax - neuron_pop->synaptic_drivens_[idx].At(iter).t);
              //cycle_ ++;
              if (t_spike >= 0) {
                if (idx < neuron_pop->Ne_) {
                  new_spikes.emplace_back(idx, t_spike, true);
                } else {
                  new_spikes.emplace_back(idx, t_spike, false);
                }
              }
              break;
            } else {
              t_spike = neuron_pop->neuron_sim_->UpdateDymState(GetPtr(neuron_pop->dym_vals_,idx), neuron_pop->synaptic_drivens_[idx].At(iter + 1).t - neuron_pop->synaptic_drivens_[idx].At(iter).t);
              //cycle_ ++;
              if (t_spike >= 0) {
                if (idx < neuron_pop->Ne_) {
                  new_spikes.emplace_back(idx, t_spike, true);
                } else {
                  new_spikes.emplace_back(idx, t_spike, false);
                }
              }
            }
            iter ++;
          }
        }
      }
    }

		void UpdatePopulationState(NeuronPopulation *neuron_pop, double t, double dt) override {
			if ( !neuron_pop->pg_mode ) {
				for (int i = 0; i < neuron_pop->neuron_number_; i ++) {
					neuron_pop->pge_[i].GenerateNewPoisson(true, t + dt, neuron_pop->ext_inputs_[i]);
					neuron_pop->pgi_[i].GenerateNewPoisson(false, t + dt, neuron_pop->ext_inputs_[i]);
				}
			}
			neuron_pop->InjectPoisson(t + dt);
      // update neurons without interactions
      std::vector<SpikeElement> tmp_spikes;
      std::vector<int> update_list;
      for (int i=0; i<neuron_pop->neuron_number_; i++)
        update_list.push_back(i);
      UpdateWithoutInteraction(neuron_pop, update_list, t, dt, tmp_spikes);
			if (neuron_pop->is_con_) {
				for (auto iter = tmp_spikes.begin(); iter != tmp_spikes.end(); iter++ ) {
					int IND = iter->index;
					Spike ADD_mutual;
					ADD_mutual.type = iter->type;
					for (TyConMat::InnerIterator it(neuron_pop->s_mat_, IND); it; ++it) {
						ADD_mutual.s = it.value();
						// Force the interneuronal interaction to the end of the time step
						ADD_mutual.t = t + dt + neuron_pop->delay_mat_[it.index()][IND];
						neuron_pop->InjectSpike(ADD_mutual, it.index());
						NEURON_INTERACTION_TIME ++;
					}
				}
			}
      // erase used spiking events;
      neuron_pop->NewSpike(t, tmp_spikes);
      neuron_pop->CleanUsedInputs(t + dt);
		}
};

class NetworkSimulatorSSC : public NetworkSimulatorSimple {
	private:
		// Sort spikes within single time interval, and return the time of first spike;
		double SortSpikes(NeuronPopulation *neuron_pop, TyDymVals &dym_vals_bk, std::vector<int> &update_list, std::vector<bool> &fired_list, double t, double dt, std::vector<SpikeElement> &T) {
      // restore states of neurons in update_list
      double t_ref_bk;
      for (auto iter = update_list.begin(); iter != update_list.end(); iter++) {
        if (!fired_list[*iter]) {
          memcpy(GetPtr(neuron_pop->dym_vals_, *iter), GetPtr(dym_vals_bk, *iter), sizeof(double)*neuron_pop->dym_n_);
        } else {
          // set artifitial refractory time to skip the update of membrane potential
          t_ref_bk = neuron_pop->dym_vals_(*iter, neuron_pop->neuron_sim_->GetIDTR());
          memcpy(GetPtr(neuron_pop->dym_vals_, *iter), GetPtr(dym_vals_bk, *iter), sizeof(double)*neuron_pop->dym_n_);
          neuron_pop->dym_vals_(*iter, neuron_pop->neuron_sim_->GetIDV()) = neuron_pop->neuron_sim_->GetRestingPotential();
          neuron_pop->dym_vals_(*iter, neuron_pop->neuron_sim_->GetIDTR()) = t_ref_bk + dt;
        }
      }
      std::vector<SpikeElement> tmp_spikes;
      UpdateWithoutInteraction(neuron_pop, update_list, t, dt, tmp_spikes);
      T.insert(T.end(), tmp_spikes.begin(), tmp_spikes.end());
			// start scanning;
			if (T.empty()) {
				return -1;
      } else {
        std::sort(T.begin(), T.end());
				return T.front().t;
			}
		}

	public:
		//	Update network state:
		void UpdatePopulationState(NeuronPopulation *neuron_pop, double t, double dt) override {
			if ( !neuron_pop->pg_mode ) {
				for (int i = 0; i < neuron_pop->neuron_number_; i ++) {
					neuron_pop->pge_[i].GenerateNewPoisson(true, t + dt, neuron_pop->ext_inputs_[i]);
					neuron_pop->pgi_[i].GenerateNewPoisson(false, t + dt, neuron_pop->ext_inputs_[i]);
				}
			}
			// inject poisson
			neuron_pop->InjectPoisson(t + dt);
      // backup neuron states
      TyDymVals dym_vals_bk(neuron_pop->neuron_number_, neuron_pop->dym_n_);
      memcpy(dym_vals_bk.data(), neuron_pop->dym_vals_.data(), sizeof(double)*neuron_pop->neuron_number_*neuron_pop->dym_n_);
      std::vector<int> update_list;
      std::vector<SpikeElement> new_spikes;
      for (int i=0; i<neuron_pop->neuron_number_; i++)
        update_list.push_back(i);
      UpdateWithoutInteraction(neuron_pop, update_list, t, dt, new_spikes);
      if (!new_spikes.empty()) {
        if (neuron_pop->is_con_) {
          std::sort(new_spikes.begin(), new_spikes.end());
          std::vector<bool> fired_list(neuron_pop->neuron_number_, false);
          std::vector<SpikeElement> T = new_spikes;
          double newt = T.front().t;
          while (newt > 0) {
            update_list.clear();
            int IND = T.front().index;
            fired_list[IND] = true;
            Spike ADD_mutual;
            ADD_mutual.type = T.front().type;
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
              for (size_t k = 0; k < T.size(); k ++) {
                if (it.index() == T[k].index) {
                  T.erase(T.begin() + k);
                  break;
                }
              }
            }
            newt = SortSpikes(neuron_pop, dym_vals_bk, update_list, fired_list, t, dt, T);
          }
        } else {
          neuron_pop->NewSpike(t, new_spikes);
        }
      }
      neuron_pop->CleanUsedInputs(t + dt);
    }
};

#endif // _NETWORK_H_
