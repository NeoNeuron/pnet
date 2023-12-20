#include "poisson_generator.h"
using namespace std;

void TyNeuronalInput::Inject(Spike& new_spike) {
  if (isnan(ptr)) {
    _spike_que.push_back(new_spike);
    ptr = 0;
    dbg_printf("inject a spike (type:%b, time:%5.2e, strength:%5.2e) into empty input sequence", new_spike.type, new_spike.t, new_spike.s);
  } else {
    // preventing new spike alter the order of current ptr.
    if (ptr >= 1) {
      if (_spike_que[ptr-1] > new_spike)
        throw runtime_error("Error: injecting new spike earlier than time stamp.");
    }
    _spike_que.push_back(new_spike);
    std::sort( _spike_que.begin() + ptr, _spike_que.end() );
  }
}

void TyNeuronalInput::Move(double t) { 
  if (ptr < _spike_que.size()) {
    while (_spike_que[ptr] < t) {
      ptr ++;
      if (ptr == _spike_que.size())
        break;
    }
  }
}

double TyNeuronalInput::Clean(double t) {
  if ( _current_up_bound <= t ) {
    _spike_que.erase(_spike_que.begin(), _spike_que.begin()+ptr);
    ptr = 0;
    return _current_up_bound += _soft_size;
  }
  return dNaN;
}

double PoissonTimeGenerator::NextPoisson() {
  POISSON_CALL_TIME ++;
  //if (output_flag_) {
  //  outfile_ << setprecision(18) << last_poisson_ << ',';
  //}
  return last_poisson_ += exp_dis(rand_gen);
}

void TyPoissonInput::InitInput(int seed) {
  TyNeuronalInput::Reset();
  double init_time_buffer;
  Spike spike_buffer;
  init_time_buffer = gen_exc.Init(seed);
  if (std::isinf(init_time_buffer)) {
    pe_toggle = false;
  } else {
    pe_toggle = true;
    spike_buffer = Spike(true, init_time_buffer, pse);
    TyNeuronalInput::Inject(spike_buffer);
  }
  // TODO: temporally use the same seed, yet the inh Poisson is usually not used.
  init_time_buffer = gen_inh.Init(seed);
  if (std::isinf(init_time_buffer)) {
    pi_toggle = false;
  } else {
    pi_toggle = true;
    spike_buffer = Spike(false, init_time_buffer, psi);
    TyNeuronalInput::Inject(spike_buffer);
  }
}
void TyPoissonInput::GenerateNewPoisson(double tmax) {
  Spike spike_buffer;
  // Exc. generation
  if (pe_toggle) {
    spike_buffer = Spike(true, 0, pse);
    // no zero point at the begining
    while (spike_buffer.t < tmax) {
      spike_buffer.t = gen_exc.NextPoisson();
      TyNeuronalInput::Inject(spike_buffer);
    }
  }
  // Inh. generation
  if (pi_toggle) {
    spike_buffer = Spike(false, 0, psi);
    while (spike_buffer.t < tmax) {
      spike_buffer.t = gen_inh.NextPoisson();
      TyNeuronalInput::Inject(spike_buffer);
    }
  }
}

void TyPoissonInput::CleanAndRefillPoisson(double t) {
  TyNeuronalInput::Move(t);
  double refill = TyNeuronalInput::Clean(t);
  if (!isnan(refill)) {
    GenerateNewPoisson(refill);
    dbg_printf("Refill Poisson spikes at %5.2f to %5.2f.\n", t, refill);
  }
}
