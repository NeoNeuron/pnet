//******************************
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Description: Define class Neuron, structure Spike and NeuronState;
//	Date: 2017-02-21 16:06:30
//******************************
#include "network.h"

using namespace std;

void Scan(vector<bool> & mat, bool target_value, vector<int> &output_indices) {
	output_indices.clear();
	for (int s = 0; s < mat.size(); s++) {
		if (mat[s] == target_value) output_indices.push_back(s);
	}
}

bool compSpikeElement(const SpikeElement &x, const SpikeElement &y) { return x.t < y.t; }

inline double L2(vector<double> &x, vector<double> &y) {	
	return sqrt((x[0] - y[0])*(x[0] - y[0]) + (x[1] - y[1])*(x[1] - y[1]));
}

void NeuronalNetwork::InitializeConnectivity(po::variables_map &vm) {
	int connecting_mode = vm["network.mode"].as<int>();
	if (connecting_mode == 0) { // External connectivity matrix;
		vector<vector<int> > connecting_matrix;
		Read2D(vm["network.file"].as<string>(), connecting_matrix);
		if (connecting_matrix.size() != neuron_number_ || connecting_matrix[0].size() != neuron_number_) {
			throw runtime_error("wrong size of connectivity matrix");
		} else {
			for (size_t i = 0; i < neuron_number_; i ++) {
				for (size_t j = 0; j < neuron_number_; j ++) {
					if (connecting_matrix[i][j]) {
						con_mat_[i][j] = true;
						if (!is_con_) is_con_ = true;
					}
				}
			}
		}
	} else if (connecting_mode == 1) {
		// Based on directed network
		int con_density = vm["network.dens"].as<int>();
		for (int i = 0; i < neuron_number_; i++)  {
			for (int j = 0; j < neuron_number_; j++) {
				if (i != j) {
					if (abs(i - j) <= con_density or neuron_number_ - abs(i - j) <= con_density) {
					con_mat_[i][j] = true;
					if (!is_con_) is_con_ = true;
					}
				}
			}
		}
		double rewiring_probability = vm["network.pr"].as<double>();
		// Generate networks;
		cout << 2 * neuron_number_ * con_density << " connections total with ";
		double x; // random variable;
		int ind, empty_connection, count = 0;
		vector<int> ones, zeros;
		for (int i = 0; i < neuron_number_; i++) {
			Scan(con_mat_[i], true, ones);
			for (int j = 0; j < ones.size(); j++) {
				x = rand_distribution(rand_gen);
				if (x <= rewiring_probability) {
					Scan(con_mat_[i], false, zeros);
					for (vector<int>::iterator it = zeros.begin(); it != zeros.end(); it++) {
						if (*it == i) {
							zeros.erase(it);
							break;
						}
					}
					empty_connection = zeros.size();
					ind = rand_gen() % empty_connection;
					con_mat_[i][zeros[ind]] = true;
					con_mat_[i][ones[j]] = false;
					count += 1;
				}
			}
		}
		cout << count << " rewirings." << endl;
	} else if (connecting_mode == 2) {
		double p_ee = vm["network.pee"].as<double>();
		double p_ie = vm["network.pie"].as<double>();
		double p_ei = vm["network.pei"].as<double>();
		double p_ii = vm["network.pii"].as<double>();
		double x;
		int count = 0;
		for (size_t i = 0; i < neuron_number_; i ++) {
			for (size_t j = 0; j < neuron_number_; j ++) {
				if (i != j) { // avoid self-connection
					x = rand_distribution(rand_gen);
					if (types_[i]) {
						if (types_[j]) {
							if (x <= p_ee) {
								con_mat_[i][j] = true;
								count ++;
							}
						} else {
							if (x <= p_ei) {
								con_mat_[i][j] = true;
								count ++;
							}
						}
					} else {
						if (types_[j]) {
							if (x <= p_ie) {
								con_mat_[i][j] = true;
								count ++;
							}
						} else {
							if (x <= p_ii) {
								con_mat_[i][j] = true;
								count ++;
							}
						}
					}
					if (con_mat_[i][j]) {
						if (!is_con_) is_con_ = true;
					}
				}
			}
		}
		printf(">> Total connections : %d\n", count);
	}
}

void NeuronalNetwork::InitializeSynapticStrength(po::variables_map &vm) {
	int synaptic_mode = vm["synapse.mode"].as<int>();
	typedef Eigen::Triplet<double> T;
	vector<T> T_list;
	if (synaptic_mode == 0) {
		vector<vector<double> > s_vals;
		Read2D(vm["synapse.file"].as<string>(), s_vals);
		for (size_t i = 0; i < neuron_number_; i ++) {
			for (size_t j = 0; j < neuron_number_; j ++) {
				if (s_vals[i][j] > 0) T_list.push_back(T(i,j,s_vals[i][j]));
			}
		}
	} else if (synaptic_mode == 1) {
		double s_ee = vm["synapse.see"].as<double>();
		double s_ie = vm["synapse.sie"].as<double>();
		double s_ei = vm["synapse.sei"].as<double>();
		double s_ii = vm["synapse.sii"].as<double>();
		for (size_t i = 0; i < neuron_number_; i ++) {
			for (size_t j = 0; j < neuron_number_; j ++) {
				if (con_mat_[i][j]) {
					if (types_[i]) {
						if (types_[j]) T_list.push_back(T(i,j,s_ee));
						else T_list.push_back(T(i,j,s_ei));
					} else {
						if (types_[j]) T_list.push_back(T(i,j,s_ie));
						else T_list.push_back(T(i,j,s_ii));
					}
				}
			}
		}
	}
	s_mat_.setFromTriplets(T_list.begin(), T_list.end());
	s_mat_.makeCompressed();
	printf("(number of connections in sparse-mat %d)\n", (int)s_mat_.nonZeros());
}

void NeuronalNetwork::InitializeSynapticDelay(po::variables_map &vm) {
	int space_mode = vm["space.mode"].as<int>();
	if (space_mode == 0) {
		delay_mat_.clear();
		delay_mat_.resize(neuron_number_, vector<double>(neuron_number_, vm["space.delay"].as<double>()));
	} else if (space_mode == 1) {
		vector<vector<double> > coordinates;
		Read2D(vm["space.file"].as<string>(), coordinates);
		SetDelay(coordinates, vm["space.speed"].as<double>());
	} else if (space_mode == -1) {
		delay_mat_.clear();
		delay_mat_.resize(neuron_number_, vector<double>(neuron_number_, 0.0));
	}
}

bool CheckExist(int index, vector<int> &list) {
	for (int i = 0; i < list.size(); i ++) {
		if (list[i] == index) return true;
	}
	return false;
}

void NeuronalNetwork::SetDelay(vector<vector<double> > &coordinates, double speed) {
	double meta_dis;
	for (int i = 0; i < neuron_number_; i ++) {
		for (int j = 0; j < i; j ++) {
			meta_dis = L2(coordinates[i], coordinates[j]) / speed;
			delay_mat_[i][j] = meta_dis;
			delay_mat_[j][i] = meta_dis;
		}
	}
}

double NeuronalNetwork::SortSpikes(DymVals &dym_vals_new, vector<int> &update_list, vector<int> &fired_list, double t, double dt, vector<SpikeElement> &T) {
	vector<double> tmp_spikes;
	SpikeElement ADD;	
	// start scanning;
	double id;
	for (int i = 0; i < update_list.size(); i++) {
		id = update_list[i];
		// Check whether id's neuron is in the fired list;
		if (CheckExist(id, fired_list)) {
			memcpy(GetPtr(dym_vals_new, id)+1, GetPtr(dym_vals_, id)+1, sizeof(double)*(dym_n_-2));
			//for (int j = 1; j < dym_n_ - 1; j ++) dym_vals_new[id][j] = dym_vals_(id, j);
			neurons_[id].UpdateSource(GetPtr(dym_vals_new,id), t, dt);
		} else {
			memcpy(GetPtr(dym_vals_new, id), GetPtr(dym_vals_, id), sizeof(double)*dym_n_);
			//for (int j = 0; j < dym_n_; j ++) dym_vals_new[id][j] = dym_vals_(id, j);
			neurons_[id].UpdateNeuronalState(GetPtr(dym_vals_new, id), t, dt, ext_inputs_[id], tmp_spikes);
			if (!tmp_spikes.empty()) {
				ADD.index = id;
				ADD.t = tmp_spikes.front();
				ADD.type = types_[id];
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

void NeuronalNetwork::SetRef(double t_ref) {
	for (int i = 0; i < neuron_number_; i ++) { neurons_[i].SetRef(t_ref); }
}

void NeuronalNetwork::InitializeNeuronalType(po::variables_map &vm) {
	int counter = 0;
	double p = vm["neuron.p"].as<double>();
	int neuron_mode = vm["neuron.mode"].as<int>();
	if (neuron_mode == 0) {
		counter = floor(neuron_number_*p);
		for (int i = 0; i < counter; i++) types_[i] = true;
	} else if (neuron_mode == 1) {
		double x = 0;
		for (int i = 0; i < neuron_number_; i++) {
			x = rand_distribution(rand_gen);
			if (x < p) {
				types_[i] = true;
				counter++;
			}
		}
	} else if (neuron_mode == 2) {
		vector<int> type_seq;
		Read1D(vm["neuron.file"].as<string>(), type_seq, 0, 0);
		for (int i = 0; i < neuron_number_; i ++) {
			if ( type_seq[i] ) {
				types_[i] = true;
				counter++;
			}
		}
	}
	printf(">> %d excitatory and %d inhibitory neurons in the network.\n", counter, neuron_number_-counter);
}

void NeuronalNetwork::InitializePoissonGenerator(po::variables_map &vm) {
	vector<vector<double> > poisson_settings;
	//	poisson_setting: 
	//		[:,0] excitatory Poisson rate;
	//		[:,1] excitatory Poisson strength;
	int driving_mode = vm["driving.mode"].as<int>();
	if (driving_mode == 0) {
		double pr = vm["driving.pr"].as<double>();
		double ps = vm["driving.ps"].as<double>();
		poisson_settings.resize(neuron_number_, vector<double>{pr, ps});
	} else if (driving_mode == 1) {
		// import the data file of feedforward driving rate:
		Read2D(vm["driving.file"].as<string>(), poisson_settings);
		if (poisson_settings.size() != neuron_number_) {
			cout << "Error inputing length! (Not equal to the number of neurons in the net)";
			return;
		}
	} else {
		throw runtime_error("wrong driving_mode");
	}

	bool poisson_output = vm["output.poi"].as<bool>();
	for (int i = 0; i < neuron_number_; i++) {
		pgs_[i].SetRate(poisson_settings[i][0]);
		pgs_[i].SetStrength(poisson_settings[i][1]);
		if (poisson_output) {
			pgs_[i].SetOuput( vm["prefix"].as<string>() + "pg" + to_string(i) + ".csv" );
		}
	}
	pg_mode = vm["driving.gmode"].as<bool>();

	if ( pg_mode ) {
		for (int i = 0; i < neuron_number_; i ++) {
			pgs_[i].GenerateNewPoisson( vm["time.T"].as<double>(), ext_inputs_[i] );
		}
	}
}

// Used in two layer network system;
// TODO: the number of sorting can be reduced;
void NeuronalNetwork::InNewSpikes(vector<vector<Spike> > & data) {
	for (int i = 0; i < neuron_number_; i++) {
		if (!data[i].empty()) {
			for (vector<Spike>::iterator it = data[i].begin(); it != data[i].end(); it++) {
				neurons_[i].InSpike(*it);
			}
		}
	}
}

void NeuronalNetwork::UpdateNetworkState(double t, double dt) {
	if ( !pg_mode ) {
		for (int i = 0; i < neuron_number_; i ++) {
			pgs_[i].GenerateNewPoisson(t + dt, ext_inputs_[i]);
		}
	}

	if (is_con_) {
		DymVals dym_vals_new(neuron_number_, dym_n_);
		memcpy(dym_vals_new.data(), dym_vals_.data(), sizeof(double)*neuron_number_*dym_n_);
		vector<SpikeElement> T;
		double newt;
		// Creating updating pool;
		vector<int> update_list, fired_list;
		for (int i = 0; i < neuron_number_; i++) update_list.push_back(i);
		newt = SortSpikes(dym_vals_new, update_list, fired_list, t, dt, T);
		while (newt > 0) {
			update_list.clear();
			int IND = (T.front()).index;
			fired_list.push_back(IND);
			Spike ADD_mutual;
			ADD_mutual.type = (T.front()).type;
			// erase used spiking events;
			T.erase(T.begin());
			neurons_[IND].Fire(t, newt);
			for (ConMat::InnerIterator it(s_mat_, IND); it; ++it) {
				ADD_mutual.s = it.value();
				ADD_mutual.t = t + newt + delay_mat_[it.index()][IND];
				neurons_[it.index()].InSpike(ADD_mutual);
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
			newt = SortSpikes(dym_vals_new, update_list, fired_list, t, dt, T);
		}
		memcpy(dym_vals_.data(), dym_vals_new.data(), sizeof(double)*neuron_number_*dym_n_);
		for (int i = 0; i < neuron_number_; i++) {
			neurons_[i].CleanUsedInputs(t + dt);
		}
	} else {
		vector<double> new_spikes;
		for (int i = 0; i < neuron_number_; i++) {
			neurons_[i].UpdateNeuronalState(GetPtr(dym_vals_, i), t, dt, ext_inputs_[i], new_spikes);
			if ( !new_spikes.empty() ) neurons_[i].Fire(t, new_spikes);
			neurons_[i].CleanUsedInputs(t + dt);
		}
	}
}

void NeuronalNetwork::PrintCycle() {
	for (int i = 0; i < neuron_number_; i++) {
		neurons_[i].GetCycle();
		cout << '\t';
	}
	cout << endl;
}

void NeuronalNetwork::OutPotential(FILEWRITE& file) {
	vector<double> potential(neuron_number_);
	for (int i = 0; i < neuron_number_; i++) {
		potential[i] = neurons_[i].GetPotential(GetPtr(dym_vals_, i));
	}
	file.Write(potential);
}

void NeuronalNetwork::OutConductance(FILEWRITE& file, bool type) {
	vector<double> conductance(neuron_number_);
	for (int i = 0; i < neuron_number_; i++) {
		conductance[i] = neurons_[i].GetConductance(GetPtr(dym_vals_, i), type);
	}
	file.Write(conductance);
}

void NeuronalNetwork::OutCurrent(FILEWRITE& file) {
	vector<double> current(neuron_number_);
	for (int i = 0; i < neuron_number_; i++) {
		current[i] = neurons_[i].GetCurrent(GetPtr(dym_vals_, i));
	}
	file.Write(current);
}

void NeuronalNetwork::SaveNeuronType(string neuron_type_file) {
	Print1D(neuron_type_file, types_, "trunc", 0);
}

void NeuronalNetwork::SaveConMat(string connecting_matrix_file) {
	Print2D(connecting_matrix_file, con_mat_, "trunc");
}

int NeuronalNetwork::OutSpikeTrains(vector<vector<double> >& spike_trains) {
	spike_trains.resize(neuron_number_);
	vector<double> add_spike_train;
	int spike_num = 0;
	for (int i = 0; i < neuron_number_; i++) {
		neurons_[i].OutSpikeTrain(add_spike_train);
		spike_trains[i] = add_spike_train;
		spike_num += add_spike_train.size();
	}
	//Print2D(path, spikes, "trunc");
	return spike_num;
}

void NeuronalNetwork::GetNewSpikes(double t, vector<vector<Spike> >& data) {
	data.clear();
	data.resize(neuron_number_);
	vector<Spike> x;
	for (int i = 0; i < neuron_number_; i++) {
		neurons_[i].GetNewSpikes(t, x);
		data[i] = x;
	}
}

int NeuronalNetwork::GetNeuronNumber() {
	return neuron_number_;
}

void NeuronalNetwork::GetConductance(int i, bool type) {
	neurons_[i].GetConductance(GetPtr(dym_vals_, i), type);
}

void NeuronalNetwork::RestoreNeurons() {
	for (int i = 0; i < neuron_number_; i++) {
		neurons_[i].Reset(GetPtr(dym_vals_, i));
		pgs_[i].Reset();
	}
	ext_inputs_.clear();
	ext_inputs_.resize(neuron_number_);
}
