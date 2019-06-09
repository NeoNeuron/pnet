//******************************
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Description: Define class Network;
//	Date: 2019-06-09
//******************************
#include "network.h"

bool CheckExist(int index, vector<int> &list) {
	for (int i = 0; i < list.size(); i ++) {
		if (list[i] == index) return true;
	}
	return false;
}

double NeuronalNetwork::SortSpikes(NeuronPopulation *neuron_pop, DymVals &dym_vals_new, vector<int> &update_list, vector<int> &fired_list, double t, double dt, vector<SpikeElement> &T) {
	vector<double> tmp_spikes;
	SpikeElement ADD;	
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
				ADD.index = id;
				ADD.t = tmp_spikes.front();
				ADD.type = neuron_pop->types_[id];
				T.push_back(ADD);
			}
		}
	}
	if (T.empty()) {
		return -1;
	} else if (T.size() == 1) {
		return (T.front()).t;
	} else {
		sort(T.begin(), T.end(), compSpikeElement);
		return (T.front()).t;
	}
}

void NeuronalNetwork::UpdateNetworkState(NeuronPopulation *neuron_pop, double t, double dt) {
	if ( !neuron_pop->pg_mode ) {
		for (int i = 0; i < neuron_pop->neuron_number_; i ++) {
			neuron_pop->pgs_[i].GenerateNewPoisson(t + dt, neuron_pop->ext_inputs_[i]);
		}
	}
	// inject poisson
	neuron_pop->InjectPoisson(t + dt);

	if (neuron_pop->is_con_) {
		DymVals dym_vals_new(neuron_pop->neuron_number_, neuron_pop->dym_n_);
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
			for (ConMat::InnerIterator it(neuron_pop->s_mat_, IND); it; ++it) {
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
		vector<double> new_spikes;
		for (int i = 0; i < neuron_pop->neuron_number_; i++) {
			neuron_pop->neuron_sim_->UpdateNeuronalState(GetPtr(neuron_pop->dym_vals_, i), neuron_pop->synaptic_drivens_[i], t, dt, new_spikes);
			if ( !new_spikes.empty() ) neuron_pop->NewSpike(i, t, new_spikes);
		}
		neuron_pop->CleanUsedInputs(t + dt);
	}
}
