//#############################################################
// Copyright: Kyle Chen
// Author: Kyle Chen
// Description: Module of Poisson Generators
// Date: 2018-12-13
//#############################################################
#ifndef _POISSON_GENERATOR_H_
#define _POISSON_GENERATOR_H_

#include "io.h"
#include "common_header.h"
#define snan std::numeric_limits<size_t>::quiet_NaN()
#define dnan std::numeric_limits<double>::quiet_NaN()
using namespace std;

struct Spike {
	bool type; // type of spike: true for excitation(AMPA, NMDA), false for inhibition(GABA);
	double t; // Exact spiking time;
	double s; // strength of spikes;
	Spike() : type(false), t(dnan), s(dnan) {  }
	Spike(bool type_val, double t_val, double s_val) 
		: type(type_val), t(t_val), s(s_val) {  }

	bool operator < (const Spike &b) const
  { return t < b.t; }
  bool operator > (const Spike &b) const
  { return t > b.t; }
  bool operator == (const Spike &b) const
  { return t == b.t && s == b.s; }

	bool operator < (const double &time) const
  { return t < time; }
	bool operator <= (const double &time) const
  { return t <= time; }
  bool operator > (const double &time) const
  { return t > time; }
  bool operator >= (const double &time) const
  { return t >= time; }
  bool operator == (const double &time) const
  { return t == time; }
  bool operator != (const double &time) const
  { return t != time; }
};

inline bool compSpike(const Spike &x, const Spike &y) { return x < y; }

typedef priority_queue< Spike, std::vector<Spike>, std::greater<Spike> > SpikeQueue;

// Neuronal Inputs
class TyNeuronalInput {
	private:
		vector<Spike> spike_seq_;
		size_t ptr;
		const size_t soft_max_ = 8192;
	public:
		TyNeuronalInput() {
			ptr = snan;
		}
		TyNeuronalInput(vector<Spike> &spikes) : spike_seq_(spikes), ptr(0) {
			if (spike_seq_.empty()) ptr = snan;
		}
		void Reset(vector<Spike> &spikes) {
			spike_seq_.clear();
			spike_seq_ = spikes;
			if (spike_seq_.empty()) {
				ptr = snan;
			} else {
				ptr = 0;
			}
		}

		size_t Inject(Spike &new_spike) {
			spike_seq_.push_back(new_spike);
			if (isnan(ptr)) {
				ptr = 0;
			} else {
				std::sort( spike_seq_.begin() + ptr, spike_seq_.end(), std::greater<Spike>() );
			}
			return ptr;
		}
		size_t Inject(vector<Spike> &new_spikes) {
			spike_seq_.insert(spike_seq_.end(), new_spikes.begin(), new_spikes.end());
			if (isnan(ptr)) {
				ptr = 0;
			}
			std::sort( spike_seq_.begin() + ptr, spike_seq_.end(), std::greater<Spike>() );
			return ptr;
		}

		// Get function
		inline const size_t Where() const { return ptr; }
		inline const size_t Size() const { return spike_seq_.size() - ptr; } // remaining size
		inline Spike At(size_t pointer = 0) const { 
			if (ptr + pointer >= spike_seq_.size()) {
				return Spike();
			} else {
				return spike_seq_[ptr + pointer];
			}
		}

		// Ptr manip functions
		// move N steps
		inline const size_t Move(size_t steps) { 
			ptr += steps;
			return ptr; 
		}
		// move until time
		inline const size_t Move(double time) { 
			if (ptr < spike_seq_.size()) {
				while (spike_seq_[ptr] < time) {
					ptr += 1;
					if (ptr == spike_seq_.size()) {
						break;
					}
				}
			}
			return ptr; 
		}
		// Clean used synaptic inputs
		inline size_t Clear() {
			if (ptr >= soft_max_) {
				spike_seq_.erase(spike_seq_.begin(), spike_seq_.begin() + ptr);
				if (spike_seq_.empty()) {
					ptr = snan;
				} else {
					ptr = 0;
				}
			}
			return ptr;
		}
		
};
typedef vector<TyNeuronalInput> TyNeuronalInputVec;

class PoissonGenerator {
	private:
		double rate_;
		double strength_;
		double last_poisson_time_;
		bool output_flag_;
		exponential_distribution<> exp_dis;
		ofstream outfile_;
	public:
		PoissonGenerator() 
			: rate_(0.0), strength_(0.0), last_poisson_time_(0.0), output_flag_(false) {  }

		PoissonGenerator(double rate, double strength, bool output) 
			: rate_(rate), strength_(strength), last_poisson_time_(0.0), output_flag_(output) {  }

		PoissonGenerator(const PoissonGenerator&) {  }
		//~PoissonGenerator() {
		//	if (outfile_.is_open()) outfile_.close();
		//}
		
		void SetRate(double rate_val) { 
			rate_ = rate_val; 
			exp_dis = exponential_distribution<>(rate_);
		}

		void SetStrength(double strength_val) { 
			strength_ = strength_val;
		}

		void SetOuput(std::string filename) {
			output_flag_ = true;
			outfile_.open(filename);
		}

		void Reset() {
			rate_								= 0.0;
			exp_dis = exponential_distribution<>();
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
