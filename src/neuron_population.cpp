//******************************
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Description: Define class NeuronPopulation;
//	Date: 2019-06-09
//******************************
#include "neuron_population.h"

using namespace std;

inline double L2(vector<double> &x, vector<double> &y) {	
	return sqrt((x[0] - y[0])*(x[0] - y[0]) + (x[1] - y[1])*(x[1] - y[1]));
}

void Scan(vector<bool> & mat, bool target_value, vector<int> &output_indices) {
	output_indices.clear();
	for (int s = 0; s < mat.size(); s++) {
		if (mat[s] == target_value) output_indices.push_back(s);
	}
}

void NeuronPopulation::InitializeConnectivity(po::variables_map &vm) {
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

void NeuronPopulation::InitializeSynapticStrength(po::variables_map &vm) {
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

void NeuronPopulation::InitializeSynapticDelay(po::variables_map &vm) {
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

void NeuronPopulation::SetDelay(vector<vector<double> > &coordinates, double speed) {
	double meta_dis;
	for (int i = 0; i < neuron_number_; i ++) {
		for (int j = 0; j < i; j ++) {
			meta_dis = L2(coordinates[i], coordinates[j]) / speed;
			delay_mat_[i][j] = meta_dis;
			delay_mat_[j][i] = meta_dis;
		}
	}
}

void NeuronPopulation::SetRef(double t_ref) {
	neuron_sim_->SetRef(t_ref);
}

void NeuronPopulation::InitializeNeuronalType(po::variables_map &vm) {
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

void NeuronPopulation::InitializePoissonGenerator(po::variables_map &vm) {
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


// Used simple interaciton case network system;
//// TODO: the number of sorting can be reduced;
//void NeuronPopulation::InNewSpikes(vector<vector<Spike> > & data) {
//	for (int i = 0; i < neuron_number_; i++) {
//		if (!data[i].empty()) {
//			for (vector<Spike>::iterator it = data[i].begin(); it != data[i].end(); it++) {
//				neuron_sim_->InSpike(synaptic_drivens_[i], spike_trains_[i], *it);
//			}
//		}
//	}
//}

void NeuronPopulation::InjectPoisson(double tmax) {
	for (int i = 0; i < neuron_number_; i ++) {
		if ( !ext_inputs_[i].empty() ) {
			while ( ext_inputs_[i].front().t < tmax ) {
				synaptic_drivens_[i].push_back( ext_inputs_[i].front() );
				ext_inputs_[i].pop();
				if ( ext_inputs_[i].empty() ) break;
			}
			sort(synaptic_drivens_[i].begin(), synaptic_drivens_[i].end(), compSpike);
		}
	}
}

void NeuronPopulation::InjectSpike(Spike x, int id) {
	if (synaptic_drivens_[id].empty()) {
		synaptic_drivens_[id].push_back(x);
	} else {
		if (synaptic_drivens_[id].back().t < x.t) {
			synaptic_drivens_[id].push_back(x);
		} else {
			synaptic_drivens_[id].push_back(x);
			sort(synaptic_drivens_[id].begin(), synaptic_drivens_[id].end(), compSpike);
		}
	}
}

void NeuronPopulation::NewSpike(int id, double t, double spike_time) {
		spike_trains_[id].push_back(t + spike_time);
}

void NeuronPopulation::NewSpike(int id, double t, vector<double>& spike_times) {
	for (vector<double>::iterator it = spike_times.begin(); it != spike_times.end(); it ++) {
		spike_trains_[id].push_back(t + *it);
	}
}

void NeuronPopulation::CleanUsedInputs(double tmax) {
	// clean old synaptic driven;
	int slen, j;
	for (int i = 0; i < neuron_number_; i ++) {
		if (!synaptic_drivens_[i].empty()) {
			slen = synaptic_drivens_[i].size();
			j = 0;
			for (; j < slen; j ++) {
				if (synaptic_drivens_[i][j].t >= tmax) break;
			}
			synaptic_drivens_[i].erase(synaptic_drivens_[i].begin(), synaptic_drivens_[i].begin() + j);
		}
	}
}

void NeuronPopulation::RestoreNeurons() {
	for (int i = 0; i < neuron_number_; i++) {
		neuron_sim_->GetDefaultDymVal(GetPtr(dym_vals_, i));
		pgs_[i].Reset();
	}
	ext_inputs_.clear();
	ext_inputs_.resize(neuron_number_);
	synaptic_drivens_.clear();
	synaptic_drivens_.resize(neuron_number_);
	spike_trains_.clear();
	spike_trains_.resize(neuron_number_);
}

void NeuronPopulation::OutPotential(FILEWRITE& file) {
	vector<double> potential(neuron_number_);
	int id = neuron_sim_->GetIDV();
	for (int i = 0; i < neuron_number_; i++) {
		potential[i] = dym_vals_(i, id);
	}
	file.Write(potential);
}

void NeuronPopulation::OutConductance(FILEWRITE& file, bool type) {
	vector<double> conductance(neuron_number_);
	int id;
	if (type) {
		id = neuron_sim_->GetIDGE();
		for (int i = 0; i < neuron_number_; i++) {
			conductance[i] = dym_vals_(i, id);
		}
	} else {
		id = neuron_sim_->GetIDGI();
		for (int i = 0; i < neuron_number_; i++) {
			conductance[i] = dym_vals_(i, id);
		}
	}
	file.Write(conductance);
}

void NeuronPopulation::OutCurrent(FILEWRITE& file) {
	vector<double> current(neuron_number_);
	for (int i = 0; i < neuron_number_; i++) {
		current[i] = neuron_sim_->GetCurrent(GetPtr(dym_vals_, i));
	}
	file.Write(current);
}

void NeuronPopulation::SaveNeuronType(string neuron_type_file) {
	Print1D(neuron_type_file, types_, "trunc", 0);
}

void NeuronPopulation::SaveConMat(string connecting_matrix_file) {
	Print2D(connecting_matrix_file, con_mat_, "trunc");
}

int NeuronPopulation::OutSpikeTrains(vector<vector<double> >& spike_trains) {
	spike_trains.resize(neuron_number_);
	int spike_num = 0;
	for (int i = 0; i < neuron_number_; i++) {
		spike_trains[i] = spike_trains_[i];
		spike_num += spike_trains_[i].size();
	}
	//Print2D(path, spikes, "trunc");
	return spike_num;
}

int NeuronPopulation::GetNeuronNumber() {
	return neuron_number_;
}

double NeuronPopulation::GetConductance(int i, bool type) {
	int id;
	if (type) id = neuron_sim_->GetIDGE();
	else id = neuron_sim_->GetIDGI();
	return dym_vals_(i, id);
}
