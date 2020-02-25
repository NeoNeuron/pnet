//==============================
//	Copyright: Kyle Chen
//	Author: Kyle Chen
//	Description: Define class NeuronPopulation;
//	Created: 2019-06-09
//==============================
#include "neuron_population.h"
#include <cnpy.h>

using namespace std;

void NeuronPopulation::InitializeSynapticStrength(po::variables_map &vm) {
	typedef Eigen::Triplet<double> T;
	vector<T> T_list;
	string sfname = vm["prefix"].as<string>() + vm["synapse.file"].as<string>();
  auto s_arr = cnpy::npy_load(sfname.c_str());
  double* s_vals = s_arr.data<double>();
  int counter = 0;
  for (int i = 0; i < neuron_number_; i ++) {
    for (int j = 0; j < neuron_number_; j ++) {
      if (s_vals[counter] > 0) T_list.push_back(T(i,j,s_vals[counter]));
      counter += 1;
		}
	}
	s_mat_.setFromTriplets(T_list.begin(), T_list.end());
	s_mat_.makeCompressed();
  if (vm["verbose"].as<bool>()) {
    printf("(number of connections in sparse-mat %d)\n", (int)s_mat_.nonZeros());
  }
	if (s_mat_.nonZeros()) is_con_ = true;
}

void NeuronPopulation::InitializeSynapticDelay(po::variables_map &vm) {
	string sfname = vm["prefix"].as<string>() + vm["space.file"].as<string>();
  auto s_arr = cnpy::npy_load(sfname.c_str());
  double* s_vals = s_arr.data<double>();
	memcpy(delay_mat_.data(), s_vals, sizeof(double)*neuron_number_*neuron_number_);
}

void NeuronPopulation::SetRef(double t_ref) {
	neuron_sim_->SetRefTime(t_ref);
}

void NeuronPopulation::InitializePoissonGenerator(po::variables_map &vm, double buffer_size) {
	vector<vector<double> > poisson_settings;
	//	poisson_setting: 
	//		[:,0] Exc. Poisson rate;
	//		[:,1] Inh. Poisson rate;
	//		[:,2] Exc. Poisson strength;
	//		[:,3] Inh. Poisson strength;
	// import the data file of feedforward driving rate:
	Read2D(vm["prefix"].as<string>() + vm["driving.file"].as<string>(), poisson_settings);
	if (poisson_settings.size() != neuron_number_) {
		throw runtime_error("Error inputing length! (Not equal to the number of neurons in the net)");
	}
	for (int i = 0; i < neuron_number_; i++) {
    inputs_vec_.emplace_back(poisson_settings[i].data(), poisson_settings[i].data()+2, buffer_size);
	}
  // generate the first episode of Poisson sequence,
  // default episode length used here.
	for (int i = 0; i < neuron_number_; i++) {
    inputs_vec_[i].InitInput();
    inputs_vec_[i].CleanAndRefillPoisson(0.0); // fill the first portion of spikes;
    dbg_printf("number of Poisson spikes in %d's neuron: %ld", i, POISSON_CALL_TIME);
	}
}

void NeuronPopulation::InitRasterOutput(string ras_path) {
	raster_file_.open(ras_path.c_str());	
}

void NeuronPopulation::NewSpike(int id, double spike_time) {
  raster_file_ << (int)id << ',' << setprecision(18) << (double)spike_time << '\n';
  SPIKE_NUMBER ++;
  if (spike_time==dNaN) {
    printf("Invalid spike time with neuron ID = %d, SPIKE_NUMBER = %ld\n", id, SPIKE_NUMBER);
  }
}

void NeuronPopulation::NewSpike(vector<SpikeTimeId>& spikes) {
  sort(spikes.begin(), spikes.end(), std::less<SpikeTimeId>() );
  for (auto iter = spikes.begin(); iter != spikes.end(); iter ++ ) {
    if (iter->t != dNaN) {
      raster_file_ << (int)iter->index << ',' << setprecision(18) << (double)(iter->t) << '\n';
    } else {
      printf("Invalid spike time with neuron ID = %d, SPIKE_NUMBER = %ld\n", iter->index, SPIKE_NUMBER);
    }
  }
  SPIKE_NUMBER += spikes.size();
}

void NeuronPopulation::RestoreNeurons() {
	for (int i = 0; i < neuron_number_; i++) {
		neuron_sim_->GetDefaultDymVal(GetPtr(dym_vals_, i));
    inputs_vec_[i].InitInput();
    inputs_vec_[i].CleanAndRefillPoisson(0.0); // fill the first portion of spikes;
	}
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
