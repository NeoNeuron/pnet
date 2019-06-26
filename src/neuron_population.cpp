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

void NeuronPopulation::InitializeSynapticStrength(po::variables_map &vm) {
	typedef Eigen::Triplet<double> T;
	vector<T> T_list;
	vector<vector<double> > s_vals;
	Read2D(vm["prefix"].as<string>() + vm["synapse.file"].as<string>(), s_vals);
	for (size_t i = 0; i < neuron_number_; i ++) {
		for (size_t j = 0; j < neuron_number_; j ++) {
			if (s_vals[i][j] > 0) T_list.push_back(T(i,j,s_vals[i][j]));
		}
	}
	s_mat_.setFromTriplets(T_list.begin(), T_list.end());
	s_mat_.makeCompressed();
	printf("(number of connections in sparse-mat %d)\n", (int)s_mat_.nonZeros());
	if (s_mat_.nonZeros()) is_con_ = true;
}

void NeuronPopulation::InitializeSynapticDelay(po::variables_map &vm) {
	int space_mode = vm["space.mode"].as<int>();
	if (space_mode == 0) {
		vector<vector<double> > coordinates;
		Read2D(vm["prefix"].as<string>() + vm["space.file"].as<string>(), coordinates);
		SetDelay(coordinates, vm["space.speed"].as<double>());
	} else if (space_mode == 1) {
		delay_mat_.clear();
		delay_mat_.resize(neuron_number_, vector<double>(neuron_number_, vm["space.delay"].as<double>()));
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
	vector<int> type_seq;
	Read1D(vm["prefix"].as<string>() + vm["neuron.file"].as<string>(), type_seq, 0, 1);
	for (int i = 0; i < neuron_number_; i ++) {
		if ( type_seq[i] ) {
			types_[i] = true;
			counter++;
		}
	}
	printf(">> %d excitatory and %d inhibitory neurons in the network.\n", counter, neuron_number_-counter);
}

void NeuronPopulation::InitializePoissonGenerator(po::variables_map &vm) {
	vector<vector<double> > poisson_settings;
	//	poisson_setting: 
	//		[:,0] excitatory Poisson rate;
	//		[:,1] excitatory Poisson strength;
	// import the data file of feedforward driving rate:
	Read2D(vm["prefix"].as<string>() + vm["driving.file"].as<string>(), poisson_settings);
	if (poisson_settings.size() != neuron_number_) {
		cout << "Error inputing length! (Not equal to the number of neurons in the net)";
		return;
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
