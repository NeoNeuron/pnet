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
    //	update_list: list of indices of neurons to be updated
		//	t_vec: vector of starting time point of each individual neurons
		//	dt_vec: vector of time step of each neurons
		//	new_spikes: new spikes generated during dt, recorded in GLOBAL spike time.
		void UpdateWithoutInteraction(NeuronPopulation* neuron_pop, std::vector<int>& update_list, std::vector<double>& t_vec, std::vector<double>& dt_vec, std::vector<SpikeElement>& new_spikes) {
			new_spikes.clear();
			double tmax, t, dt;
      int idx;
      for (size_t i = 0; i < update_list.size(); i ++) {
        idx = update_list[i];
        t = t_vec[i];
        dt = dt_vec[i];
        tmax = t + dt;
        if (std::isnan(neuron_pop->inputs_vec_[idx].At().t) || neuron_pop->inputs_vec_[idx].At() >= tmax) {
          neuron_pop->UpdateNeuronStateLocal(idx, t, dt, new_spikes);
        } else {
          if (neuron_pop->inputs_vec_[idx].At() != t) {
            neuron_pop->UpdateNeuronStateLocal(idx, t, neuron_pop->inputs_vec_[idx].At().t - t, new_spikes);
          }
          size_t iter = 0;
          while (true) {
            // Update conductance due to the synaptic inputs;
            neuron_pop->DeltaInteraction(neuron_pop->inputs_vec_[idx].At(iter), idx);

            if (std::isnan(neuron_pop->inputs_vec_[idx].At(iter + 1).t) || neuron_pop->inputs_vec_[idx].At(iter + 1) >= tmax) {
              neuron_pop->UpdateNeuronStateLocal(idx, neuron_pop->inputs_vec_[idx].At(iter).t, tmax - neuron_pop->inputs_vec_[idx].At(iter).t, new_spikes);
              break;
            } else {
              neuron_pop->UpdateNeuronStateLocal(idx, neuron_pop->inputs_vec_[idx].At(iter).t, neuron_pop->inputs_vec_[idx].At(iter+1).t - neuron_pop->inputs_vec_[idx].At(iter).t, new_spikes);
            }
            iter ++;
          }
        }
      }
    }

		void UpdatePopulationState(NeuronPopulation *neuron_pop, double t, double dt) override {
      // update neurons without interactions
      std::vector<SpikeElement> tmp_spikes;
      std::vector<int> update_list;
      for (int i=0; i<neuron_pop->neuron_number_; i++)
        update_list.push_back(i);
      std::vector<double> t_vec(neuron_pop->neuron_number_, t);
      std::vector<double> dt_vec(neuron_pop->neuron_number_, dt);
      UpdateWithoutInteraction(neuron_pop, update_list, t_vec, dt_vec, tmp_spikes);
      // synaptic interactions
			if (neuron_pop->is_con_) {
				for (auto iter = tmp_spikes.begin(); iter != tmp_spikes.end(); iter++ ) {
					int IND = iter->index;
					Spike ADD_mutual;
					ADD_mutual.type = iter->type;
					for (TySparseAdjMat::InnerIterator it(neuron_pop->s_mat_, IND); it; ++it) {
						ADD_mutual.s = it.value();
						// Force the interneuronal interaction to the end of the time step
            // TODO incorrect;
						//ADD_mutual.t = dt + neuron_pop->delay_mat_(it.index(), IND);
            neuron_pop->DeltaInteraction(ADD_mutual, it.index());
//						neuron_pop->InjectSpike(ADD_mutual, it.index());
						NEURON_INTERACTION_TIME ++;
					}
				}
			}
      // erase used spiking events;
      neuron_pop->NewSpike(tmp_spikes);
      neuron_pop->CleanUsedInputs(t + dt);
		}
};

class NetworkSimulatorSSC : public NetworkSimulatorSimple {
	private:
		// Sort spikes within single time interval, and return the time of first spike;
		void SortSpikes(NeuronPopulation *neuron_pop, TyDymVals &dym_vals_bk, std::vector<int> &update_list, std::vector<bool> &fired_list,
        std::vector<double>& t_vec, std::vector<double>& dt_vec, std::vector<SpikeElement> &T) {
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
          neuron_pop->dym_vals_(*iter, neuron_pop->neuron_sim_->GetIDTR()) = t_ref_bk + dt_vec[*iter];
        }
      }
      std::vector<SpikeElement> tmp_spikes;
      UpdateWithoutInteraction(neuron_pop, update_list, t_vec, dt_vec, tmp_spikes);
      T.insert(T.end(), tmp_spikes.begin(), tmp_spikes.end());
      std::sort(T.begin(), T.end());
		}

	public:
		//	Update network state:
		void UpdatePopulationState(NeuronPopulation *neuron_pop, double t, double dt) override {
      // backup neuron states
      TyDymVals dym_vals_bk(neuron_pop->neuron_number_, neuron_pop->dym_n_);
      memcpy(dym_vals_bk.data(), neuron_pop->dym_vals_.data(), sizeof(double)*neuron_pop->neuron_number_*neuron_pop->dym_n_);
      std::vector<int> affected_list;   // Attention: affected_list should not be sorted!
      std::vector<double> affected_list_strength;
      std::vector<SpikeElement> new_spikes;
      for (int i=0; i<neuron_pop->neuron_number_; i++)
        affected_list.push_back(i);
      std::vector<double> t_vec(neuron_pop->neuron_number_, t);
      std::vector<double> dt_vec(neuron_pop->neuron_number_, dt);
      UpdateWithoutInteraction(neuron_pop, affected_list, t_vec, dt_vec, new_spikes);
      if (!new_spikes.empty()) {
        if (neuron_pop->is_con_) {
          std::sort(new_spikes.begin(), new_spikes.end());
          std::vector<bool> fired_list(neuron_pop->neuron_number_, false);
          std::vector<SpikeElement> T = new_spikes;
          double newt;
          int IND;
          while (!T.empty()) {
            affected_list.clear();
            //affected_list_strength.clear();
            IND = T.front().index;
            newt = T.front().t;
            fired_list[IND] = true;
            // perform synaptic interaction
            Spike ADD_mutual;
            ADD_mutual.type = T.front().type;
            // erase used spiking events;
            T.erase(T.begin());
            neuron_pop->NewSpike(IND, newt);
            for (TySparseAdjMat::InnerIterator it(neuron_pop->s_mat_, IND); it; ++it) {
              ADD_mutual.s = it.value();
              ADD_mutual.t = newt + neuron_pop->delay_mat_(it.index(), IND);
              neuron_pop->InjectSpike(ADD_mutual, it.index());
              //neuron_pop->DeltaInteraction(ADD_mutual, it.index());
              NEURON_INTERACTION_TIME ++;
              affected_list.push_back(it.index());
              // Check whether this neuron appears in the firing list T;
              for (size_t k = 0; k < T.size(); k ++) {
                if (it.index() == T[k].index) {
                  T.erase(T.begin() + k);
                  break;
                }
              }
              UpdateWithoutInteraction(neuron_pop, affected_list, t_vec, dt_vec, new_spikes);
            }
            SortSpikes(neuron_pop, dym_vals_bk, affected_list, fired_list, t_vec, dt_vec, T);
          }
        } else {
          neuron_pop->NewSpike(new_spikes);
        }
      }
      neuron_pop->CleanUsedInputs(t + dt);
    }
};

#endif // _NETWORK_H_
