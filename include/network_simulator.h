// ========================================
// Copyright: Kyle Chen
// Author: Kyle Chen
// Created: 2019-06-09
// Description: define class network simulator;
// ========================================
#ifndef _NETWORK_SIMULATOR_H_
#define _NETWORK_SIMULATOR_H_

#include "common_header.h"
#include "neuron.h"
#include "neuron_population.h"

class NetworkSimulatorBase {
	public:
	virtual void UpdatePopulationState(NeuronPopulationBase* neuron_pop, double t, double dt) = 0;
	virtual ~NetworkSimulatorBase() {  }
};

// TODO: numerical convergence need to be checked;
class NetworkSimulatorSimple : public NetworkSimulatorBase {
	public:
  // TODO: Test the feasibility of pragma omp parallel implementation
				// #pragma omp parallel for
		//	Update neuron within single time step, including its membrane potential, conductances and counter of refractory period;
		//	neuron_pop: pointer of neuron population;
    //	update_list: list of indices of neurons to be updated
		//	t_vec: vector of starting time point of each individual neurons
		//	dt_vec: vector of time step of each neurons
		//	new_spikes: new spikes generated during dt, recorded in GLOBAL spike time.
    //	comment: this new iteration procedure is more accurate than old one;
		void UpdateWithoutInteraction(NeuronPopulationBase* neuron_pop, 
        std::vector<int>& update_list, std::vector<double>& t_vec, 
        std::vector<double>& dt_vec, std::vector<SpikeTimeId>& new_spikes) 
    {
			double tmax, t;
      for (const int& idx : update_list) {
        t = t_vec[idx];
        tmax = t + dt_vec[idx];
        if (neuron_pop->GetSingleSynapticInput(idx, 0).t< t) {
          std::cout << "ERROR: time pointer behind Poisson pointer.\n";
        }
        size_t iter = 0;
        while (neuron_pop->GetSingleSynapticInput(idx, iter) < tmax) {
          neuron_pop->UpdateNeuronStateLocal(idx, t, neuron_pop->GetSingleSynapticInput(idx, iter).t - t, new_spikes);
          // Update conductance due to the synaptic inputs;
          neuron_pop->DeltaInteraction(neuron_pop->GetSingleSynapticInput(idx, iter), idx);
          t = neuron_pop->GetSingleSynapticInput(idx, iter).t;
          iter ++;
        }
        neuron_pop->UpdateNeuronStateLocal(idx, t, tmax - t, new_spikes);
      }
    }

		void UpdatePopulationState(NeuronPopulationBase* neuron_pop, double t, double dt) override {
      int N = neuron_pop->GetNeuronNumber();
      // update neurons without interactions
      std::vector<SpikeTimeId> tmp_spikes;
      std::vector<int> update_list;
      for (int i=0; i<N; i++)
        update_list.push_back(i);
      std::vector<double> t_vec(N, t);
      std::vector<double> dt_vec(N, dt);
      UpdateWithoutInteraction(neuron_pop, update_list, t_vec, dt_vec, tmp_spikes);
      // synaptic interactions
			if (neuron_pop->GetIsCon()) {
				for (auto iter = tmp_spikes.begin(); iter != tmp_spikes.end(); iter++ ) {
					int index = iter->index;
					Spike spike_buffer;
          spike_buffer.type = index < neuron_pop->GetNe() ? true : false;
          // TODO: integrate into SynapticInteraction module
					for (TySparseAdjMat::InnerIterator it(*(neuron_pop->GetAdjMat()), index); it; ++it) {
						spike_buffer.s = it.value();
						spike_buffer.t = iter->t + neuron_pop->GetDelay(it.index(), index);
						// Force the interneuronal interaction to the end of the time step
            if (spike_buffer.t < t + dt)
              spike_buffer.t = t + dt;
            neuron_pop->InjectSpike(spike_buffer, it.index());
            NEURON_INTERACTION_TIME ++;
					}
				}
			}
      // Output spikes, which are not sorted.
      for (auto spike : tmp_spikes)
        neuron_pop->NewSpike(spike);
      // erase used spiking events;
      neuron_pop->CleanUsedInputs(t + dt);
		}
};

class NetworkSimulatorSSC : public NetworkSimulatorSimple {
	public:
		//	Update network state:
    void UpdatePopulationState(NeuronPopulationBase* neuron_pop, double t, double dt) override {
      int N = neuron_pop->GetNeuronNumber();
      int N_dym = neuron_pop->GetDymN();
      // backup neuron states
      TyDymVals dym_vals_bk(N, N_dym);
      memcpy(dym_vals_bk.data(), neuron_pop->GetDymPtr(0), sizeof(double)*N*N_dym);

      std::vector<int> affected_list;   // Attention: affected_list should not be sorted!
      std::vector<SpikeTimeId> new_spikes;
      for (int i = 0; i<N; i++)
        affected_list.push_back(i);
      std::vector<double> t_vec(N, t);
      std::vector<double> dt_vec(N, dt);
      UpdateWithoutInteraction(neuron_pop, affected_list, t_vec, dt_vec, new_spikes);
      if (new_spikes.empty()) {
        // do nothing
      } else {
        if (!neuron_pop->GetIsCon()) {
          // Output spikes, which are not sorted.
          for (auto spike : new_spikes)
            neuron_pop->NewSpike(spike);
        } else {
          std::vector<bool> affected_list_bool(N, false);
          std::vector<bool> fired_list(N, false);
          std::vector<SpikeTimeId> missing_spikes;
          while (!new_spikes.empty()) {
            std::sort(new_spikes.begin(), new_spikes.end());
            affected_list.clear();
            affected_list_bool = std::vector<bool>(N, false);

            SpikeTimeId heading_spike = new_spikes.front();
            // erase used spiking events;
            new_spikes.erase(new_spikes.begin());
            fired_list[heading_spike.index] = true;

            // get affected_list, not including spiking one;
            for (TySparseAdjMat::InnerIterator iit(*(neuron_pop->GetAdjMat()), heading_spike.index); iit; ++iit) {
              affected_list.push_back(iit.index());
              affected_list_bool[iit.index()] = true;
            }
            // Check whether this neuron appears in the firing list new_spikes;
            auto iter = new_spikes.begin();
            while (iter != new_spikes.end()) {
              if (affected_list_bool[iter->index]) {
                iter = new_spikes.erase(iter);
              } else {
                iter ++;
              }
            }
            // roll back affected_list
            for (auto idx : affected_list) {
              memcpy(neuron_pop->GetDymPtr(idx), GetPtr(dym_vals_bk, idx), 
                  sizeof(double)*N_dym);
            }

            neuron_pop->NewSpike(heading_spike);
            // perform synaptic interaction
            //  (note: synaptic interaction should better perform 
            //         before updating backup states to prevent failure)
            neuron_pop->SynapticInteraction(heading_spike);

            // update affected_list to the end of time step
            UpdateWithoutInteraction(neuron_pop, affected_list, t_vec, dt_vec, missing_spikes);
            // Check missing spike due to the inaccurate estimation.
            for (auto iter = missing_spikes.begin(); iter != missing_spikes.end(); iter++) {
              if (!fired_list[iter->index]) {
                if (iter->t < heading_spike.t) {
                  fprintf(stderr, 
                      "Missing spike before 'first' spike: [%d] neuron at t = %5.2f\n",
                      iter->index, iter->t);
                  new_spikes.emplace_back(heading_spike.t,iter->index);
                } else {
                  new_spikes.push_back(*iter);
                }
              }
            }
            missing_spikes.clear();
          }
        }
      }
      neuron_pop->CleanUsedInputs(t + dt);
    }
};

class NetworkSimulatorSSC_Sparse : public NetworkSimulatorSimple {
	public:
		//	Update network state:
    void UpdatePopulationState(NeuronPopulationBase* neuron_pop, double t, double dt) override {
      int N = neuron_pop->GetNeuronNumber();
      int N_dym = neuron_pop->GetDymN();
      // backup neuron states
      TyDymVals dym_vals_bk(N, N_dym);
      memcpy(dym_vals_bk.data(), neuron_pop->GetDymPtr(0), sizeof(double)*N*N_dym);

      std::vector<int> affected_list;   // Attention: affected_list should not be sorted!
      std::vector<SpikeTimeId> new_spikes;
      for (int i = 0; i<N; i++)
        affected_list.push_back(i);
      std::vector<double> t_vec(N, t);
      std::vector<double> dt_vec(N, dt);
      UpdateWithoutInteraction(neuron_pop, affected_list, t_vec, dt_vec, new_spikes);
      if (new_spikes.empty()) {
        // do nothing
      } else {
        if (!neuron_pop->GetIsCon()) {
          for (auto spike : new_spikes)
            // Output spikes, which are not sorted.
            neuron_pop->NewSpike(spike);
        } else {
          std::vector<bool> affected_list_bool(N, false);
          std::vector<SpikeTimeId> missing_spikes;
          while (!new_spikes.empty()) {
            std::sort(new_spikes.begin(), new_spikes.end());
            affected_list.clear();
            affected_list_bool = std::vector<bool>(N, false);

            SpikeTimeId heading_spike = new_spikes.front();
            // erase used spiking events;
            new_spikes.erase(new_spikes.begin());

            // get updating information
            affected_list.push_back(heading_spike.index);
            affected_list_bool[heading_spike.index] = true;
            dt_vec[heading_spike.index] = heading_spike.t - t_vec[heading_spike.index];
            for (TySparseAdjMat::InnerIterator iit(*(neuron_pop->GetAdjMat()), heading_spike.index); iit; ++iit) {
              affected_list.push_back(iit.index());
              affected_list_bool[iit.index()] = true;
              dt_vec[iit.index()] = heading_spike.t-t_vec[iit.index()];
              if (dt_vec[iit.index()] < 0) {
                printf("ERROR: negative dt  = %f for affected neuron [%d] at global t = %f ms, local t = %f ms\n",
                    dt_vec[iit.index()], iit.index(), t, t_vec[iit.index()]);
                cout << "Current 'first' spike " << heading_spike.t << endl;
                throw runtime_error("");
              }
            }
            // Check whether this neuron appears in the firing list new_spikes;
            auto iter = new_spikes.begin();
            while (iter != new_spikes.end()) {
              if (affected_list_bool[iter->index]) {
                iter = new_spikes.erase(iter);
              } else {
                iter ++;
              }
            }
            // roll back affected_list
            for (auto idx : affected_list) {
              memcpy(neuron_pop->GetDymPtr(idx), GetPtr(dym_vals_bk, idx), 
                  sizeof(double)*N_dym);
            }
            // update affected_list to heading_spike.t
            UpdateWithoutInteraction(neuron_pop, affected_list, t_vec, dt_vec, missing_spikes);
            // Check missing spike due to the inaccurate estimation.
            bool new_spike_toggle = false;
            for (auto iter = missing_spikes.begin(); iter != missing_spikes.end(); iter++) {
              if (iter->index == heading_spike.index) {
                new_spike_toggle=true;
              } else {
                fprintf(stderr, 
                    "Missing spike before 'first' spike: [%d] neuron at t = %5.2f\n",
                    iter->index, iter->t);
                new_spikes.emplace_back(heading_spike.t,iter->index);
              }
            }
            if (!new_spike_toggle)
              neuron_pop->GetNeuronModel()->ForceFire(neuron_pop->GetDymPtr(heading_spike.index));

            missing_spikes.clear();

            neuron_pop->NewSpike(heading_spike);
            // perform synaptic interaction
            //  (note: synaptic interaction should perform before
            //         backup update to avoid wrong sequence order failure)
            neuron_pop->SynapticInteraction(heading_spike);

            // update backup states of affected neurons
            for (auto idx : affected_list) {
              memcpy(GetPtr(dym_vals_bk, idx), neuron_pop->GetDymPtr(idx),
                  sizeof(double)*N_dym);
              t_vec[idx] = heading_spike.t;
              dt_vec[idx] = t + dt - heading_spike.t;
              neuron_pop->CleanUsedInputs(idx, heading_spike.t);
            }

            // update affected list to the end of the time step
            UpdateWithoutInteraction(neuron_pop, affected_list, t_vec, dt_vec, new_spikes);
          }
        }
      }
      neuron_pop->CleanUsedInputs(t + dt);
    }
};

#endif // _NETWORK_SIMULATOR_H_
