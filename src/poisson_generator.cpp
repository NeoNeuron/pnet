#include "poisson_generator.h"
using namespace std;

void PoissonGenerator::GenerateNewPoisson( double tmax, queue<Spike>& poisson_driven ) {
	Spike new_spike;
	new_spike.type = true;
	new_spike.s = strength_;
	//double x; 
	double tLast = last_poisson_time_;
	while (tLast < tmax) {
		new_spike.t = tLast;
		poisson_driven.push(new_spike);
		if (output_flag_) {
			outfile_ << setprecision(18) << tLast << ',';
		}
		// Generate new Poisson time point;
		//x = rand_distribution(rand_gen);
		//tLast -= log(x) / rate_;
		tLast += exp_dis(rand_gen);
	}
	last_poisson_time_ = tLast;
	//sort(poisson_driven.begin(), poisson_driven.end(), compSpike);
}
