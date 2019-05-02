//******************************
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Description: Define class Neuron, structure Spike and NeuronState;
//	Date: 2018-05-30
//******************************
#include "neuron.h"
#include "math_helper.h"
#include "fmath.hpp"
#define exp(x) fmath::expd(x)
using namespace std;

bool compSpike(const Spike &x, const Spike &y) { return x.t < y.t; }

void NeuronSim::InputExternalPoisson(double tmax, queue<Spike>& x) {
	if ( !x.empty() ) {
		while ( x.front().t < tmax ) {
			synaptic_driven_.push_back( x.front() );
			x.pop();
			if ( x.empty() ) break;
		}
	}
	sort(synaptic_driven_.begin(), synaptic_driven_.end(), compSpike);
}

void NeuronSim::SetDefaultDymVal(double *&dym_val) {
	if ( !dym_val ) {
		dym_val = new double[ p_neuron_ -> GetDymNum() ];
	}
	p_neuron_ -> SetDefaultDymVal(dym_val);
}

void NeuronSim::Reset(double *dym_val) {
	synaptic_driven_.clear();
	spike_train_.clear();
	cycle_ = 0;
	// reset dynamic variables;
	p_neuron_ -> SetDefaultDymVal(dym_val);
}

void NeuronSim::OutSpikeTrain(vector<double> & spikes) {
	spikes.clear();
	spikes = spike_train_;
}

void NeuronSim::GetNewSpikes(double t, vector<Spike>& x) {
	Spike add_spike;
	add_spike.s = 0.0;
	// NOTE: the type of new spike always set to true, to be determined in network class
	add_spike.type = true;
	x.clear();
	for (vector<double>::reverse_iterator iter = spike_train_.rbegin(); iter != spike_train_.rend(); iter++) {
		if (*iter >= t) {
			add_spike.t = *iter;
			x.push_back(add_spike);
		} else break;
	}
}

double NeuronSim::UpdateNeuronalState(double *dym_val, double t, double dt, queue<Spike>& extPoisson, vector<double>& new_spikes) {
	new_spikes.clear();
	double tmax = t + dt;
	InputExternalPoisson(tmax, extPoisson);
	double t_spike;
	vector<Spike>::iterator s_begin = synaptic_driven_.begin();
	if (s_begin == synaptic_driven_.end() || tmax <= s_begin->t) {
		t_spike = p_neuron_ -> DymCore(dym_val, dt);
		cycle_ ++;
		if (t_spike >= 0) new_spikes.push_back(t_spike);
	} else {
		if (t != s_begin->t) {
			t_spike = p_neuron_ -> DymCore(dym_val, s_begin->t - t);
			cycle_ ++;
			if (t_spike >= 0) new_spikes.push_back(t_spike);
		}
		for (vector<Spike>::iterator iter = s_begin; iter != synaptic_driven_.end(); iter++) {
			// Update conductance due to the synaptic inputs;
			if (iter -> type) dym_val[ p_neuron_ -> GetGEID() ] += iter -> s;
			else dym_val[ p_neuron_ -> GetGIID() ] += iter -> s;
			if (iter + 1 == synaptic_driven_.end() || (iter + 1)->t >= tmax) {
				t_spike = p_neuron_ -> DymCore(dym_val, tmax - iter->t);
				cycle_ ++;
				if (t_spike >= 0) new_spikes.push_back(t_spike);
				break;
			} else {
				t_spike = p_neuron_ -> DymCore(dym_val, (iter + 1)->t - iter->t);
				cycle_ ++;
				if (t_spike >= 0) new_spikes.push_back(t_spike);
			}
		}
	}
	return dym_val[p_neuron_ -> GetVID()];
}

double NeuronSim::CleanUsedInputs(double *dym_val, double *dym_val_new, double tmax) {
	// Update dym_val with dym_val_new;
	for (int i = 0; i < p_neuron_ -> GetDymNum(); i ++) dym_val[i] = dym_val_new[i];
	// clean old synaptic driven;
	int slen = synaptic_driven_.size();
	if (slen != 0) {
		int i = 0;
		for (; i < slen; i ++) {
			if (synaptic_driven_[i].t >= tmax) break;
		}
		synaptic_driven_.erase(synaptic_driven_.begin(), synaptic_driven_.begin() + i);
	}
	return dym_val[p_neuron_ -> GetVID()];
}

void NeuronSim::UpdateSource(double *dym_val, double t, double dt) {
	double tmax = t + dt;
	if (synaptic_driven_.empty() || tmax <= synaptic_driven_.begin()->t) {
		p_neuron_ -> UpdateSource(dym_val, dt);
	} else {
		if (t != synaptic_driven_.begin()->t) {
			p_neuron_ -> UpdateSource(dym_val, synaptic_driven_.begin()->t - t);
		}
		for (vector<Spike>::iterator iter = synaptic_driven_.begin(); iter != synaptic_driven_.end(); iter++) {
			if (iter -> s) dym_val[ p_neuron_ -> GetGEID() ] += iter -> s;
			else dym_val[ p_neuron_ -> GetGIID() ] += iter -> s;
			if (iter + 1 == synaptic_driven_.end() || (iter + 1)->t >= tmax) {
				p_neuron_ -> UpdateSource(dym_val, tmax - iter->t);
				break;
			} else {
				p_neuron_ -> UpdateSource(dym_val, (iter + 1)->t - iter->t);
			}
		}
	}
}

void NeuronSim::Fire(double t, double spike_time) {
		spike_train_.push_back(t + spike_time);
}

void NeuronSim::Fire(double t, vector<double>& spike_times) {
	for (vector<double>::iterator it = spike_times.begin(); it != spike_times.end(); it ++) {
		spike_train_.push_back(t + *it);
	}
}	

void NeuronSim::InSpike(Spike x) {
	if (synaptic_driven_.empty()) {
		synaptic_driven_.push_back(x);
	} else if (synaptic_driven_.back().t < x.t) {
		synaptic_driven_.push_back(x);
	} else {
		synaptic_driven_.push_back(x);
		sort(synaptic_driven_.begin(),synaptic_driven_.end(),compSpike);
	}
}

