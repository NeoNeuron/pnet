#include "poisson_generator.h"
using namespace std;

void PoissonGenerator::GenerateNewPoisson( bool type, double tmax, PoissonSeq & poisson_driven ) {
	//double x; 
	double tLast = last_poisson_time_;
	while (tLast < tmax) {
		poisson_driven.emplace(type, tLast, strength_);
		if (output_flag_) {
			outfile_ << setprecision(18) << tLast << ',';
		}
		// Generate new Poisson time point;
		//x = rand_distribution(rand_gen);
		//tLast -= log(x) / rate_;
		tLast += exp_dis(rand_gen);
	}
	last_poisson_time_ = tLast;
}
