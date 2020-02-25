// ========================================
// Copyright: Kyle Chen
// Author: Kyle Chen
// Created: 2019-06-09
// Description: define class network simulator;
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
		void UpdateWithoutInteraction(NeuronPopulation* neuron_pop, std::vector<int>& update_list, std::vector<double>& t_vec, std::vector<double>& dt_vec, std::vector<SpikeTimeId>& new_spikes) {
			double tmax, t;
      for (const int& idx : update_list) {
        t = t_vec[idx];
        tmax = t + dt_vec[idx];
        if (neuron_pop->inputs_vec_[idx].At().t< t) {
          std::cout << "ERROR: invalid situation in Poisson input container\n";
        }
        size_t iter = 0;
        while (neuron_pop->inputs_vec_[idx].At(iter) < tmax) {
          neuron_pop->UpdateNeuronStateLocal(idx, t, neuron_pop->inputs_vec_[idx].At(iter).t - t, new_spikes);
          // Update conductance due to the synaptic inputs;
          neuron_pop->DeltaInteraction(neuron_pop->inputs_vec_[idx].At(iter), idx);
          t = neuron_pop->inputs_vec_[idx].At(iter).t;
          iter ++;
        }
        neuron_pop->UpdateNeuronStateLocal(idx, t, tmax - t, new_spikes);
      }
    }

		void UpdatePopulationState(NeuronPopulation *neuron_pop, double t, double dt) override {
      // update neurons without interactions
      std::vector<SpikeTimeId> tmp_spikes;
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
					Spike spike_buffer;
          if (IND < neuron_pop->Ne_) {
            spike_buffer.type = true;
          } else {
            spike_buffer.type = false;
          }
					for (TySparseAdjMat::InnerIterator it(neuron_pop->s_mat_, IND); it; ++it) {
						spike_buffer.s = it.value();
						spike_buffer.t = iter->t + neuron_pop->delay_mat_(it.index(), IND);
						// Force the interneuronal interaction to the end of the time step
            if (spike_buffer.t < t + dt)
              spike_buffer.t = t + dt;
            neuron_pop->InjectSpike(spike_buffer, it.index());
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
	public:
		//	Update network state:
		void UpdatePopulationState(NeuronPopulation *neuron_pop, double t, double dt) override {
      // backup neuron states
      TyDymVals dym_vals_bk(neuron_pop->neuron_number_, neuron_pop->dym_n_);
      memcpy(dym_vals_bk.data(), neuron_pop->dym_vals_.data(), sizeof(double)*neuron_pop->neuron_number_*neuron_pop->dym_n_);

      std::vector<int> affected_list;   // Attention: affected_list should not be sorted!
      std::vector<double> affected_list_strength;
      std::vector<SpikeTimeId> new_spikes;
      for (int i = 0; i<neuron_pop->neuron_number_; i++)
        affected_list.push_back(i);
      std::vector<double> t_vec(neuron_pop->neuron_number_, t);
      std::vector<double> dt_vec(neuron_pop->neuron_number_, dt);
      UpdateWithoutInteraction(neuron_pop, affected_list, t_vec, dt_vec, new_spikes);
      if (new_spikes.empty()) {
        // do nothing
      } else {
        if (!neuron_pop->is_con_) {
          neuron_pop->NewSpike(new_spikes);
        } else {
          std::vector<bool> affected_list_bool(neuron_pop->neuron_number_, false);
          std::vector<SpikeTimeId> missing_spikes;
          while (!new_spikes.empty()) {
            std::sort(new_spikes.begin(), new_spikes.end());
            affected_list.clear();
            affected_list_strength.clear();
            affected_list_bool = std::vector<bool>(neuron_pop->neuron_number_, false);

            int IND = new_spikes.front().index;
            double newt = new_spikes.front().t;

            // erase used spiking events;
            new_spikes.erase(new_spikes.begin());

            // get updating information
            affected_list.push_back(IND);
            affected_list_bool[IND] = true;
            dt_vec[IND] = newt - t_vec[IND];
            for (TySparseAdjMat::InnerIterator iit(neuron_pop->s_mat_, IND); iit; ++iit) {
              affected_list.push_back(iit.index());
              affected_list_bool[iit.index()] = true;
              affected_list_strength.push_back(iit.value());
              dt_vec[iit.index()] = newt-t_vec[iit.index()];
              if (dt_vec[iit.index()] < 0) {
                printf("ERROR: negative dt  = %f for affected neuron [%d] at global t = %f ms, local t = %f ms\n",
                    dt_vec[iit.index()], iit.index(), t, t_vec[iit.index()]);
                cout << "Current 'first' spike " << newt << endl;
                throw runtime_error("");
              }
            }
            // Check whether this neuron appears in the firing list new_spikes;
            auto T_iter = new_spikes.begin();
            while (T_iter != new_spikes.end()) {
              if (affected_list_bool[T_iter->index]) {
                T_iter = new_spikes.erase(T_iter);
              } else {
                T_iter ++;
              }
            }
            // roll back affected_list
            for (auto idx : affected_list) {
              memcpy(GetPtr(neuron_pop->dym_vals_, idx), 
                  GetPtr(dym_vals_bk, idx), 
                  sizeof(double)*neuron_pop->dym_n_);
            }
            // update affected_list to newt
            UpdateWithoutInteraction(neuron_pop, affected_list, t_vec, dt_vec, missing_spikes);
            // Check missing spike due to the inaccurate estimation.
            bool new_spike_toggle = false;
            for (auto iter = missing_spikes.begin(); iter != missing_spikes.end(); iter++) {
              if (iter->index == IND) {
                new_spike_toggle=true;
              } else {
                fprintf(stderr, 
                    "Missing spike before 'first' spike: [%d] neuron at t = %5.2f\n",
                    iter->index, iter->t);
                new_spikes.emplace_back(iter->index, newt);
              }
            }
            if (!new_spike_toggle)
              neuron_pop->neuron_sim_->ForceFire(GetPtr(neuron_pop->dym_vals_, IND));

            missing_spikes.clear();

            Spike spike_buffer;
            neuron_pop->NewSpike(IND, newt);
            if (IND < neuron_pop->Ne_) {
              spike_buffer.type = true;
            } else {
              spike_buffer.type = false;
            }

            // perform synaptic interaction
            //  (note: synaptic interaction should better perform 
            //         before updating backup states to prevent failure)
            for (size_t i = 1; i < affected_list.size(); i++) {
              spike_buffer.s = affected_list_strength[i-1];
              spike_buffer.t = newt + neuron_pop->delay_mat_(affected_list[i], IND);
              neuron_pop->InjectSpike(spike_buffer, affected_list[i]);
              NEURON_INTERACTION_TIME ++;
            }

            // update backup states of affected neurons
            for (auto idx : affected_list) {
              memcpy(GetPtr(dym_vals_bk, idx), 
                  GetPtr(neuron_pop->dym_vals_, idx), 
                  sizeof(double)*neuron_pop->dym_n_);
              t_vec[idx] = newt;
              dt_vec[idx] = t + dt - newt;
              neuron_pop->CleanUsedInputs(idx, newt);
            }
            
            // update affected list to the end of the time step
            UpdateWithoutInteraction(neuron_pop, affected_list, t_vec, dt_vec, new_spikes);
          }
        }
      }
      neuron_pop->CleanUsedInputs(t + dt);
    }
};

#endif // _NETWORK_H_
