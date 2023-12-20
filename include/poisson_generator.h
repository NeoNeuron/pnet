// =========================================
// Copyright: Kyle Chen
// Author: Kyle Chen
// Description: Module of Poisson Generators
// Created: 2018-12-13
// =========================================
#ifndef _POISSON_GENERATOR_H_
#define _POISSON_GENERATOR_H_

#include "io.h"
#include "common_header.h"

struct Spike {
	bool type; // type of spike: true for excitation(AMPA, NMDA), false for inhibition(GABA);
	double t; // Exact spiking time;
	double s; // strength of spikes;
	Spike() : type(false), t(dNaN), s(dNaN) {  }
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

// Neuronal Inputs
class TyNeuronalInput {
	private:
    std::vector<Spike> _spike_que;
		size_t ptr;
		const double _soft_size; // (ms)
    double _current_up_bound;
	public:
    TyNeuronalInput() : ptr(sNaN), _soft_size(100) {
      _current_up_bound = 0.0;
    }
		TyNeuronalInput(const double& size) 
      : ptr(sNaN), _soft_size(size) { 
        _current_up_bound = 0.0;
      }

    void Reset() {
      _spike_que.clear();
      ptr=sNaN;
      _current_up_bound = 0.0;
    }

		// Get function
		Spike At(size_t pointer = 0) const { 
			if (ptr + pointer >= _spike_que.size()) {
				return Spike();
			} else {
				return *(_spike_que.data() + ptr + pointer);
			}
		}

    void Inject(Spike& new_spike);
    void Move(double t);    // move ptr until time
    double Clean(double t); // clean used synaptic inputs, return the destination 
                            // of next episode if cleaned.
		
};

class PoissonTimeGenerator {
  private:
		double rate_;
		double last_poisson_ = Inf; // last poisson time, used;
		bool output_flag_ = false;
		exponential_distribution<> exp_dis;
    std::mt19937 rand_gen;
		ofstream outfile_;
  public:
    PoissonTimeGenerator() : rate_(0.0) {  }

		PoissonTimeGenerator(double rate) : rate_(rate) {  }

    // TODO figure out why cannot define copy constructor
		//PoissonTimeGenerator(const PoissonTimeGenerator&) {  }

    double Rate() const {
      return rate_;
    }

		void SetRate(double rate_val) { 
      assert(rate_val >= 0);
      rate_ = rate_val; 
		}

		void SetOuput(std::string filename) {
			output_flag_ = true;
			outfile_.open(filename);
		}

    double Init(int seed) {
      if (rate_ == 0) {
        last_poisson_ = Inf;
        return Inf;
      } else if (rate_ > 0) {
        last_poisson_	= 0;
        exp_dis = exponential_distribution<>(rate_);
        rand_gen.seed(seed);
        return NextPoisson();
      } else {
        throw runtime_error("Error: negative Poisson rate.");
      }
    }

    double NextPoisson();
  
};

class TyPoissonInput: public TyNeuronalInput {
	private:
    PoissonTimeGenerator gen_exc, gen_inh;
    double pse, psi;
    double pe_toggle, pi_toggle;

	public:
    TyPoissonInput() : gen_exc(), gen_inh(), pse(0.0), psi(0.0) {  }
		TyPoissonInput(double* rate, double* strength) 
      : gen_exc(rate[0]), gen_inh(rate[1]),
        pse(strength[0]), psi(strength[1]) {  }
		TyPoissonInput(double* rate, double* strength, const double& max_capacity) 
      : TyNeuronalInput(max_capacity),
        gen_exc(rate[0]), gen_inh(rate[1]),
        pse(strength[0]), psi(strength[1]) {  }

		//TyPoissonInput(const TyPoissonInput&) {  }

    // Initialize Poisson generators
		void InitInput(int seed); 

    // Generate new Poisson series until tmax; 
		void GenerateNewPoisson(double tmax);

    // Generate new poisson and fill the neuronal input sequence;
		void CleanAndRefillPoisson(double tmax);

};

#endif
