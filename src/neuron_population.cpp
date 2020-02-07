//******************************
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Description: Define class NeuronPopulation;
//	Date: 2019-06-09
//******************************
#include "neuron_population.h"
#include <cnpy.h>

using namespace std;

inline double L2(vector<double> &x, vector<double> &y) {	
	return sqrt((x[0] - y[0])*(x[0] - y[0]) + (x[1] - y[1])*(x[1] - y[1]));
}

void NeuronPopulation::InitializeSynapticStrength(po::variables_map &vm) {
	typedef Eigen::Triplet<double> T;
	vector<T> T_list;
	string sfname = vm["prefix"].as<string>() + vm["synapse.file"].as<string>();
  auto s_arr = cnpy::npy_load(sfname.c_str());
  double* s_vals = s_arr.data<double>();
  int counter = 0;
  for (size_t i = 0; i < neuron_number_; i ++) {
    for (size_t j = 0; j < neuron_number_; j ++) {
      if (s_vals[counter] > 0) T_list.push_back(T(i,j,s_vals[counter]));
      counter += 1;
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
    pge_[i].SetRate(poisson_settings[i][0]);
    pge_[i].SetStrength(poisson_settings[i][1]);
    pgi_[i].SetRate(poisson_settings[i][2]);
    pgi_[i].SetStrength(poisson_settings[i][3]);
		if (poisson_output) {
      pge_[i].SetOuput( vm["prefix"].as<string>() + "pge" + to_string(i) + ".csv" );
      pgi_[i].SetOuput( vm["prefix"].as<string>() + "pgi" + to_string(i) + ".csv" );
		}
	}
	pg_mode = vm["driving.gmode"].as<bool>();

	if ( pg_mode ) {
    for (int i = 0; i < neuron_number_; i ++) {
      pge_[i].GenerateNewPoisson( true,  vm["time.T"].as<double>(), ext_inputs_[i] );
      pgi_[i].GenerateNewPoisson( false, vm["time.T"].as<double>(), ext_inputs_[i] );
    }
	}
}

void NeuronPopulation::InitRasterOutput(string ras_path) {
	raster_file_.open(ras_path.c_str());	
}

// Used simple interaciton case network system;
//// TODO: the number of sorting can be reduced;
//void NeuronPopulation::InNewSpikes(vector<vector<Spike> > & data) {
//	for (int i = 0; i < neuron_number_; i++) {
//		if (!data[i].empty()) {
//			for (auto it = data[i].begin(); it != data[i].end(); it++) {
//				neuron_sim_->InSpike(synaptic_drivens_[i], spike_trains_[i], *it);
//			}
//		}
//	}
//}

void NeuronPopulation::InjectPoisson(double tmax) {
  for (int i = 0; i < neuron_number_; i ++) {
    if ( !ext_inputs_[i].empty() ) {
      while ( ext_inputs_[i].top().t < tmax ) {
        InjectSpike(ext_inputs_[i].top(), i);
        ext_inputs_[i].pop();
        if ( ext_inputs_[i].empty() ) break;
      }
    }
  }
}

void NeuronPopulation::NewSpike(int id, double t, double spike_time) {
  raster_file_ << (int)id << ',' << setprecision(18) << (double)(t+spike_time) << '\n';
  SPIKE_NUMBER ++;
  if (t+spike_time==dnan) {
    printf("Invalid spike time with neuron ID = %d, t = %f, SPIKE_NUMBER = %ld\n", id, t, SPIKE_NUMBER);
  }
}

void NeuronPopulation::NewSpike(int id, double t, vector<double>& spike_times) {
	for (auto it = spike_times.begin(); it != spike_times.end(); it ++) {
		raster_file_ << (int)id << ',' << setprecision(18) << (double)(t+*it) << '\n';
	}
}

// TODO: merge this step with updating function
void NeuronPopulation::CleanUsedInputs(double tmax) {
	for (int i = 0; i < neuron_number_; i ++) {
		synaptic_drivens_[i].Move(tmax);
		// clean old synaptic driven;
		synaptic_drivens_[i].Clear();
	}
}

void NeuronPopulation::RestoreNeurons() {
	for (int i = 0; i < neuron_number_; i++) {
		neuron_sim_->GetDefaultDymVal(GetPtr(dym_vals_, i));
    pge_[i].Reset();
    pgi_[i].Reset();
	}
	ext_inputs_.clear();
	ext_inputs_.resize(neuron_number_);
	synaptic_drivens_.clear();
	synaptic_drivens_.resize(neuron_number_);
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

int NeuronPopulation::GetNeuronNumber() {
	return neuron_number_;
}

double NeuronPopulation::GetConductance(int i, bool type) {
	int id;
	if (type) id = neuron_sim_->GetIDGE();
	else id = neuron_sim_->GetIDGI();
	return dym_vals_(i, id);
}
