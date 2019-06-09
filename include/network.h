//***************
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Date: 2019-06-09
//	Description: define Class Network;
//***************
#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "common_header.h"
#include "neuron.h"
#include "neuron_population.h"
using namespace std;

class NeuronalNetwork {
private:
	// Sort spikes within single time interval, and return the time of first spike;
	double SortSpikes(NeuronPopulation *neuron_pop, DymVals &dym_val_new, vector<int>& update_list, vector<int>& fired_list, double t, double dt, vector<SpikeElement> &T);

public:
	//	Neuronal network initialization:
	NeuronalNetwork() {  }
	~NeuronalNetwork() {  }
	//	Update network state:
	void UpdateNetworkState(NeuronPopulation *neuron_pop, double t, double dt);

};

#endif // _NETWORK_H_
