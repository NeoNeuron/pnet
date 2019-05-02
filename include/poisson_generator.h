//#############################################################
// Copyright: Kyle Chen
// Author: Kyle Chen
// Description: Module of Poisson Generators
// Date: 2018-12-13
//#############################################################
#ifndef _POISSON_GENERATOR_H_
#define _POISSON_GENERATOR_H_

#include "io.h"
#include "neuron.h"
#include "common_header.h"
using namespace std;

class PoissonGenerator {
	private:
		double rate_;
		double strength_;
		double last_poisson_time_;
		bool output_flag_;
		ofstream outfile_;
	public:
		PoissonGenerator() {
			rate_								= 0.0;
			strength_						= 0.0;
			last_poisson_time_	= 0.0;
			output_flag_ = false;
		}
		PoissonGenerator(const PoissonGenerator&) {  }
		//~PoissonGenerator() {
		//	if (outfile_.is_open()) outfile_.close();
		//}
		
		void SetRate(double rate_val) { rate_ = rate_val; }

		void SetStrength(double strength_val) { strength_ = strength_val; }

		void SetOuput(std::string filename) {
			output_flag_ = true;
			outfile_.open(filename);
		}

		void Reset() {
			rate_								= 0.0;
			strength_						= 0.0;
			last_poisson_time_	= 0.0;
			output_flag_ = false;
		}

		// Generate new Poisson series and export to synpatic_driven; autosort after generatation if synaptic delay is nonzero;
		// tmax: maximum time of Poisson sequence;
		// synaptic_driven: container for new poisson spikes;
		// return: none;
		void GenerateNewPoisson( double tmax, queue<Spike>& poisson_driven );

};

#endif
