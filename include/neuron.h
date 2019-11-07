//******************************
//  Copyright: Kyle Chen
//  Author: Kyle Chen
//  Description: Define base class of Neuron model and Neuron simulator, structure Spike;
//  Date: 2019-05-05
//******************************
#ifndef _NEURON_H_
#define _NEURON_H_

#include "common_header.h"
#include "poisson_generator.h"
#include "math_helper.h"
// Fast code for exp(x)
#include "fmath.hpp"
//#define exp(x) fmath::expd(x)


template<typename T>
inline T my_exp(const T &x)
{ return exp(x); }

template<>
inline double my_exp(const double &x)
{ return fmath::expd(x); }

//#define exp(x) my_expd(x)
using namespace std;

// Neuron Base:
class NeuronShellBase {
  public:
    // "Get parameter" functions
    virtual int GetDymNum() const = 0; 
    virtual int GetIDV() const = 0;
    virtual int GetIDGE() const = 0; 
    virtual int GetIDGI() const = 0; 
    virtual int GetIDTR() const = 0; 
    virtual int GetIDGEInject() const = 0; 
    virtual int GetIDGIInject() const = 0; 
    virtual double GetRestingPotential() const = 0;
    virtual double GetRefTime() const = 0; 
    virtual double GetCurrent(double* dym_val) const = 0;
    virtual void GetDefaultDymVal(double *dym_val) const = 0;
    // "Set parameter" function
    virtual void SetRefTime(double t_ref) = 0;
    // "Evolving Dynamics" functions
    virtual void ManuallyFire(double* dym_val) const = 0;
    virtual double DymCore(double *dym_val, double dt) const = 0;
    virtual void UpdateSource(double *dym_val, double dt) const = 0;

    virtual ~NeuronShellBase() {  }
};


// Class Neuron: Based on integrate and fire neuron model;
class LIF_G_Core {
  public:
    // PARAMETERS:
    double tau_E_ = 2.0;  // (ms) time const for excitatory conductance;
    double tau_I_ = 5.0;  // (ms) time const for inhibitory conductance;
    double g_m_ = 5e-2;   // (1/ms) normalized membrane conductance;
    double tau_ = 2.0;    // (ms) refractory Period;
    double resting_potential_ = 0.0;
    double threshold_potential_ = 1.0;
    double excitatory_reversal_potential_ = 14.0 / 3.0;
    double inhibitory_reversal_potential_ = -2.0 / 3.0;
    
    // indices of dynamical variables
    static const int dym_n_ = 4;
    static const int id_v_  = 0;
    static const int id_gE_ = 1;
    static const int id_gI_ = 2;
    static const int id_tr_ = 3;
    static const int id_gE_inject_ = id_gE_;
    static const int id_gI_inject_ = id_gI_;
    // index of remaining refractory period time. if negative, remaining refractory period equals to zero;
    void GetDefaultDymVal(double* dym_val) const {
      dym_val[id_v_]  = 0.0;
      dym_val[id_gE_] = 0.0;
      dym_val[id_gI_] = 0.0;
      dym_val[id_tr_] = -1;
    }
    // DYNAMICS:

    //  Purely update conductance after single time step dt;
    //  dym_val: dynamical variables;
    //  dt: time step;
    //  return: none;
    void UpdateG(double *dym_val, double dt) const {
      dym_val[id_gE_] *= my_exp( -dt / tau_E_ );
      dym_val[id_gI_] *= my_exp( -dt / tau_I_ );
    }

    // ODE govern the dynamic of IF neuron;
    // dym_val: dynamical variables;
    // return: dV/dt, the derivative of V;
    double GetDv(double *dym_val) const {
      return - g_m_ * (dym_val[id_v_] - resting_potential_)
        - dym_val[id_gE_] * (dym_val[id_v_] - excitatory_reversal_potential_)
        - dym_val[id_gI_] * (dym_val[id_v_] - inhibitory_reversal_potential_)
        ;
    }
    
    //  Update the conductance and membrane potential for t = [t_n, t_n + dt];
    //  Description: 4th-order Runge Kutta integration scheme is applied;
    //  *voltage: pointer of voltage, updated after excecution;
    //  dt: size of time step, unit ms;
    //  return: derivative of membrane potential at t = t(n);
    double DymInplaceRK4(double *dym_val, double dt) const {
      double exp_E = my_exp(-0.5 * dt / tau_E_);
      double exp_I = my_exp(-0.5 * dt / tau_I_);
      // k1 = GetDv(t_n, v_n);
      // k2 = GetDv(t_n+1/2, v_n + k1*dt / 2);
      // k3 = GetDv(t_n+1/2, v_n + k2*dt / 2);
      // k4 = GetDv(t_n+1, v_n + k3*dt);
      // v_n+1 = v_n + dt/6*(k1 + 2*k2 + 2*k3 + k4);
      double v_n = dym_val[id_v_];
      double k1, k2, k3, k4;
      k1 = GetDv(dym_val);
      // Update G:
      dym_val[id_gE_] *= exp_E;
      dym_val[id_gI_] *= exp_I;
      dym_val[id_v_] = v_n + 0.5*k1*dt;
      k2 = GetDv(dym_val);
      dym_val[id_v_] = v_n + 0.5*k2*dt;
      k3 = GetDv(dym_val);
      // Update G:
      dym_val[id_gE_] *= exp_E;
      dym_val[id_gI_] *= exp_I;
      dym_val[id_v_] = v_n + k3*dt;
      k4 = GetDv(dym_val);
      // Get v_n+1;
      dym_val[id_v_] = v_n + dt / 6 *(k1 + 2 * k2 + 2 * k3 + k4);
      return k1;
    }
};

class LIF_GH_Core {
  public:
    // PARAMETERS:
    double tau_Er_ = 1.0; // (ms) rising const for exc conductance;
    double tau_Ed_ = 2.0; // (ms) decay  const for exc conductance;
    double tau_Ir_ = 1.0; // (ms) rising const for inh conductance;
    double tau_Id_ = 10.0;  // (ms) decay  const for inh conductance;
    double g_m_ = 5e-2;   // (1/ms) normalized membrane conductance;
    double tau_ = 2.0;    // (ms) default refractory Period;
    double resting_potential_ = 0.0;
    double threshold_potential_ = 1.0;
    double excitatory_reversal_potential_ = 14.0 / 3.0;
    double inhibitory_reversal_potential_ = -2.0 / 3.0;
    
    // excitatory and inhibitory conductance; evolve precisely with the given expression;
    static const int dym_n_ = 6;
    static const int id_v_  = 0;
    static const int id_gE_ = 1;
    static const int id_gI_ = 2;
    static const int id_hE_ = 3;
    static const int id_hI_ = 4;
    static const int id_tr_ = 5;
    static const int id_gE_inject_ = id_hE_;
    static const int id_gI_inject_ = id_hI_;
    // index of remaining refractory period time. if negative, remaining refractory period equals to zero;
    void GetDefaultDymVal(double* dym_val) const {
      dym_val[id_v_]  = 0.0;
      dym_val[id_gE_] = 0.0;
      dym_val[id_gI_] = 0.0;
      dym_val[id_hE_] = 0.0;
      dym_val[id_hI_] = 0.0;
      dym_val[id_tr_] = -1;
    }

    // DYNAMICS:
    //
    //  Only update conductance after dt;
    //    g(t) = S * ( exp(-t/td) - exp(-t/tr) )
    //
    //  ODEs:
    //    g' = -1/td * g + h
    //    h' = -1/tr * h
    //  Solutions:
    //    g[t] = exp(-t/td)*g[0] + td*tr/(td-tr)*(exp(-t/td) - exp(-t/tr))*h[0]
    //    h[t] = exp(-t/tr)*h[0]
    //
    //  dym_val: dynamical variables;
    //  dt: time step;
    //  return: none;
    void UpdateG(double *dym_val, double dt) const {
      // excitatory
      double exp_r = my_exp(-dt / tau_Er_);
      double exp_d = my_exp(-dt / tau_Ed_);
      dym_val[id_gE_] = exp_d*dym_val[id_gE_] + (exp_d - exp_r)*tau_Ed_*tau_Er_/(tau_Ed_ - tau_Er_)*dym_val[id_hE_];
      dym_val[id_hE_] *= exp_r;
      // inhibitory
      exp_r = my_exp(-dt / tau_Ir_);
      exp_d = my_exp(-dt / tau_Id_);
      dym_val[id_gI_] = exp_d*dym_val[id_gI_] + (exp_d - exp_r)*tau_Id_*tau_Ir_/(tau_Id_ - tau_Ir_)*dym_val[id_hI_];
      dym_val[id_hI_] *= exp_r;
    }

    // ODE govern the dynamic of IF neuron;
    // dym_val: dynamical variables;
    // return: dV/dt, the derivative of V;
    double GetDv(double *dym_val) const {
      return - g_m_ * (dym_val[id_v_] - resting_potential_)
        - dym_val[id_gE_] * (dym_val[id_v_] - excitatory_reversal_potential_)
        - dym_val[id_gI_] * (dym_val[id_v_] - inhibitory_reversal_potential_)
        ;
    }
    
    //  Update the conductance and membrane potential for t = [t_n, t_n + dt];
    //  Description: 4th-order Runge Kutta integration scheme is applied;
    //  *voltage: pointer of voltage, updated after excecution;
    //  dt: size of time step, unit ms;
    //  return: derivative of membrane potential at t = t(n);
    double DymInplaceRK4(double *dym_val, double dt) const {
      double exp_Er = my_exp(-0.5 * dt / tau_Er_);
      double exp_Ed = my_exp(-0.5 * dt / tau_Ed_);
      double exp_Ir = my_exp(-0.5 * dt / tau_Ir_);
      double exp_Id = my_exp(-0.5 * dt / tau_Id_);
      double exp_E_comb = (exp_Ed - exp_Er)*tau_Ed_*tau_Er_/(tau_Ed_ - tau_Er_);
      double exp_I_comb = (exp_Id - exp_Ir)*tau_Id_*tau_Ir_/(tau_Id_ - tau_Ir_);
      // k1 = GetDv(t_n, v_n);
      // k2 = GetDv(t_n+1/2, v_n + k1*dt / 2);
      // k3 = GetDv(t_n+1/2, v_n + k2*dt / 2);
      // k4 = GetDv(t_n+1, v_n + k3*dt);
      // v_n+1 = v_n + dt/6*(k1 + 2*k2 + 2*k3 + k4);
      double v_n = dym_val[id_v_];
      double k1, k2, k3, k4;
      k1 = GetDv(dym_val);
      // Update G:
      dym_val[id_gE_] = exp_Ed * dym_val[id_gE_] + exp_E_comb*dym_val[id_hE_];
      dym_val[id_hE_] *= exp_Er;
      dym_val[id_gI_] = exp_Id * dym_val[id_gI_] + exp_I_comb*dym_val[id_hI_];
      dym_val[id_hI_] *= exp_Ir;
      dym_val[id_v_] = v_n + 0.5*k1*dt;
      k2 = GetDv(dym_val);
      dym_val[id_v_] = v_n + 0.5*k2*dt;
      k3 = GetDv(dym_val);
      // Update G:
      dym_val[id_gE_] = exp_Ed * dym_val[id_gE_] + exp_E_comb*dym_val[id_hE_];
      dym_val[id_hE_] *= exp_Er;
      dym_val[id_gI_] = exp_Id * dym_val[id_gI_] + exp_I_comb*dym_val[id_hI_];
      dym_val[id_hI_] *= exp_Ir;
      dym_val[id_v_] = v_n + k3*dt;
      k4 = GetDv(dym_val);
      // Get v_n+1;
      dym_val[id_v_] = v_n + dt / 6 *(k1 + 2 * k2 + 2 * k3 + k4);
      return k1;
    }
};

class LIF_I_Core {
  public:
    // PARAMETERS:
    double tau_E_ = 2.0;  // (ms) time const for exc synaptic current;
    double tau_I_ = 2.0;  // (ms) time const for inh synaptic current;
    double g_m_ = 5e-2;   // (1/ms) normalized membrane conductance;
    double tau_ = 2.0;    // (ms) refractory Period;
    double resting_potential_ = 0.0;  // scaled resting membrane potential;
    double threshold_potential_ = 1.0;// scaled threshold membrane potential;
    
    // excitatory and inhibitory conductance; evolve precisely with the given expression;
    static const int dym_n_ = 4;
    static const int id_v_  = 0;
    static const int id_gE_ = 1;
    static const int id_gI_ = 2;
    static const int id_tr_ = 3;
    static const int id_gE_inject_ = id_gE_;
    static const int id_gI_inject_ = id_gI_;
    // index of remaining refractory period time. if negative, remaining refractory period equals to zero;
    void GetDefaultDymVal(double* dym_val) const {
      dym_val[id_v_]  = 0.0;
      dym_val[id_gE_] = 0.0;
      dym_val[id_gI_] = 0.0;
      dym_val[id_tr_] = -1;
    }

    // DYNAMICS:

    //  Purely update current after single time step dt;
    //  dym_val: dynamical variables;
    //  dt: time step;
    //  return: none;
    void UpdateG(double *dym_val, double dt) const {
      dym_val[id_gE_] *= my_exp( -dt / tau_E_ );
      dym_val[id_gI_] *= my_exp( -dt / tau_I_ );
    }

    // ODE govern the dynamic of IF neuron;
    // dym_val: dynamical variables;
    // return: dV/dt, the derivative of V;
    double GetDv(double *dym_val) const {
      return - g_m_ * (dym_val[id_v_] - resting_potential_) 
        + dym_val[id_gE_] 
        - dym_val[id_gI_];
    }
    
    //  Update the conductance and membrane potential for t = [t_n, t_n + dt];
    //  Description: 4th-order Runge Kutta integration scheme is applied;
    //  *voltage: pointer of voltage, updated after excecution;
    //  dt: size of time step, unit ms;
    //  return: derivative of membrane potential at t = t(n);
    double DymInplaceRK4(double *dym_val, double dt) const {
      double exp_E = my_exp(-0.5 * dt / tau_E_);
      double exp_I = my_exp(-0.5 * dt / tau_I_);
      // k1 = GetDv(t_n, v_n);
      // k2 = GetDv(t_n+1/2, v_n + k1*dt / 2);
      // k3 = GetDv(t_n+1/2, v_n + k2*dt / 2);
      // k4 = GetDv(t_n+1, v_n + k3*dt);
      // v_n+1 = v_n + dt/6*(k1 + 2*k2 + 2*k3 + k4);
      double v_n = dym_val[id_v_];
      double k1, k2, k3, k4;
      k1 = GetDv(dym_val);
      // Update current:
      dym_val[id_gE_] *= exp_E;
      dym_val[id_gI_] *= exp_I;
      dym_val[id_v_] = v_n + 0.5*k1*dt;
      k2 = GetDv(dym_val);
      dym_val[id_v_] = v_n + 0.5*k2*dt;
      k3 = GetDv(dym_val);
      // Update current:
      dym_val[id_gE_] *= exp_E;
      dym_val[id_gI_] *= exp_I;
      dym_val[id_v_] = v_n + k3*dt;
      k4 = GetDv(dym_val);
      // Get v_n+1;
      dym_val[id_v_] = v_n + dt / 6 *(k1 + 2 * k2 + 2 * k3 + k4);
      return k1;
    }
};

// LIF_GH neuron with external current input.
template <class ExtCurrent>
class LIF_GH_EXT_Core : public ExtCurrent, public LIF_GH_Core {
  public:
    typedef typename ExtCurrent::TyData TyParaData;
    using ExtCurrent::GetI;
    //// PARAMETERS:
    using LIF_GH_Core::tau_;
    using LIF_GH_Core::resting_potential_;
    using LIF_GH_Core::threshold_potential_;
    using LIF_GH_Core::dym_n_;
    using LIF_GH_Core::id_v_ ;
    using LIF_GH_Core::id_gE_;
    using LIF_GH_Core::id_gI_;
    using LIF_GH_Core::id_hE_;
    using LIF_GH_Core::id_hI_;
    using LIF_GH_Core::id_tr_;
    using LIF_GH_Core::id_gE_inject_;
    using LIF_GH_Core::id_gI_inject_;
    // index of remaining refractory period time. if negative, remaining refractory period equals to zero;
    using LIF_GH_Core::GetDefaultDymVal;

    // DYNAMICS:
    using LIF_GH_Core::UpdateG; 

    // ODE govern the dynamic of IF neuron;
    // dym_val    : dynamical variables;
    // t          : time point;
    // TyParaData : Parametric data of external current;
    // return     : dV/dt, the derivative of V;
    inline double GetDv(double *dym_val, double t, TyParaData &x) const {
      return - g_m_ * (dym_val[id_v_] - resting_potential_)
        - dym_val[id_gE_] * (dym_val[id_v_] - excitatory_reversal_potential_)
        - dym_val[id_gI_] * (dym_val[id_v_] - inhibitory_reversal_potential_)
        + GetI(t, x);
    }
    
    //  Update the conductance and membrane potential for t = [t_n, t_n + dt];
    //  Description: 4th-order Runge Kutta integration scheme is applied;
    //  *voltage: pointer of voltage, updated after excecution;
    //  dt: size of time step, unit ms;
    //  return: derivative of membrane potential at t = t(n);
    double DymInplaceRK4(double *dym_val, double t, double dt, TyParaData &x) const {
      double exp_Er = my_exp(-0.5 * dt / tau_Er_);
      double exp_Ed = my_exp(-0.5 * dt / tau_Ed_);
      double exp_Ir = my_exp(-0.5 * dt / tau_Ir_);
      double exp_Id = my_exp(-0.5 * dt / tau_Id_);
      double exp_E_comb = (exp_Ed - exp_Er)*tau_Ed_*tau_Er_/(tau_Ed_ - tau_Er_);
      double exp_I_comb = (exp_Id - exp_Ir)*tau_Id_*tau_Ir_/(tau_Id_ - tau_Ir_);
      // k1 = GetDv(t_n, v_n);
      // k2 = GetDv(t_n+1/2, v_n + k1*dt / 2);
      // k3 = GetDv(t_n+1/2, v_n + k2*dt / 2);
      // k4 = GetDv(t_n+1, v_n + k3*dt);
      // v_n+1 = v_n + dt/6*(k1 + 2*k2 + 2*k3 + k4);
      double v_n = dym_val[id_v_];
      double k1, k2, k3, k4;
      k1 = GetDv(dym_val, t, x);
      // Update G:
      dym_val[id_gE_] = exp_Ed * dym_val[id_gE_] + exp_E_comb*dym_val[id_hE_];
      dym_val[id_hE_] *= exp_Er;
      dym_val[id_gI_] = exp_Id * dym_val[id_gI_] + exp_I_comb*dym_val[id_hI_];
      dym_val[id_hI_] *= exp_Ir;
      dym_val[id_v_] = v_n + 0.5*k1*dt;
      k2 = GetDv(dym_val, t+dt/2, x);
      dym_val[id_v_] = v_n + 0.5*k2*dt;
      k3 = GetDv(dym_val, t+dt/2, x);
      // Update G:
      dym_val[id_gE_] = exp_Ed * dym_val[id_gE_] + exp_E_comb*dym_val[id_hE_];
      dym_val[id_hE_] *= exp_Er;
      dym_val[id_gI_] = exp_Id * dym_val[id_gI_] + exp_I_comb*dym_val[id_hI_];
      dym_val[id_hI_] *= exp_Ir;
      dym_val[id_v_] = v_n + k3*dt;
      k4 = GetDv(dym_val, t+dt, x);
      // Get v_n+1;
      dym_val[id_v_] = v_n + dt / 6 *(k1 + 2 * k2 + 2 * k3 + k4);
      return k1;
    }
};

// class Neuron_LIF:
// Implement basic operations for sub-timestep dynamics;
template <class NeuronCore>
class Neuron_LIF: public NeuronCore, public NeuronShellBase {
  using NeuronCore::tau_;   // (ms) refractory Period;
  using NeuronCore::resting_potential_;
  using NeuronCore::threshold_potential_;
  using NeuronCore::dym_n_;
  using NeuronCore::id_v_;
  using NeuronCore::id_gE_;
  using NeuronCore::id_gI_;
  using NeuronCore::id_tr_;
  using NeuronCore::id_gE_inject_;
  using NeuronCore::id_gI_inject_;
  using NeuronCore::UpdateG;
  using NeuronCore::GetDv;
  using NeuronCore::DymInplaceRK4;
  public:
    int GetDymNum() const override { return dym_n_; }
    int GetIDV()    const override { return id_v_; }
    int GetIDGE()   const override { return id_gE_; }
    int GetIDGI()   const override { return id_gI_; }
    int GetIDTR()   const override { return id_tr_; }
    int GetIDGEInject() const override { return id_gE_inject_; } 
    int GetIDGIInject() const override { return id_gI_inject_; } 
    double GetRestingPotential() const override { return resting_potential_; }
    double GetRefTime() const override { return tau_; }
    double GetCurrent(double *dym_val) const override { return GetDv(dym_val); }
    void SetRefTime(double t_ref) override { tau_ = t_ref; }
    void ManuallyFire(double *dym_val) const override {
      dym_val[id_v_] = resting_potential_;
      dym_val[id_tr_] = tau_;
    }
    //  Core operation for updating neuronal state within single timing step dt;
    //  Description: operation to update neuronal state in primary level, including updating conductances, membrane potential and checking spiking events; 
    //  dym_val: array of dynamic variables;
    //  dt: size of time step, unit ms;
    //  return: -1 for no spiking events; otherwise, return relative spiking time respect to the begining of the time step;
    //  Remark: if the input current (strength of synaptic input) is too large, or the neuron are at bursting state, the function might fail;
    double DymCore(double *dym_val, double dt) const override {
      double vn = dym_val[id_v_];
      // Update conductance;
      double dvn, dv_new;
      double t_spike = -1; // spike time within dt;
      if (dym_val[id_tr_] <= 0) { // neuron is not in the refractory period;
        dvn = DymInplaceRK4(dym_val, dt);
        // Check whether fire or not;
        if (dym_val[id_v_] > threshold_potential_) {
          dv_new = GetDv(dym_val);
          t_spike = cubic_hermite_root(dt, vn, dym_val[id_v_], dvn, dv_new, threshold_potential_);
          dym_val[id_v_] = resting_potential_;
          // update remaining fractory period
          dym_val[id_tr_] = tau_ + t_spike - dt;
          // if the refractory period is short enough, the neuron will be reactivate;
          if (dym_val[id_tr_] < 0) {
            // restore the source (driving current or conductance);
            UpdateG( dym_val, dym_val[id_tr_] );
            DymInplaceRK4( dym_val, -dym_val[id_tr_] );
          }
        } 
      } else { // neuron is about to exit the refractory period;
        if (dym_val[id_tr_] < dt) {
          UpdateG(dym_val, dym_val[id_tr_]);
          dvn = DymInplaceRK4(dym_val, dt - dym_val[id_tr_]);
          // Check whether fire or not;
          if (dym_val[id_v_] >= threshold_potential_) {
            dv_new = GetDv(dym_val);
            t_spike = cubic_hermite_root(dt - dym_val[id_tr_], vn, dym_val[id_v_], dvn, dv_new, threshold_potential_);
            dym_val[id_v_] = resting_potential_;
            // update remaining fractory period
            t_spike += dym_val[id_tr_];
            dym_val[id_tr_] = tau_ + t_spike;
          }
        } else { // neuron is in the refractory period;
          UpdateG(dym_val, dt);
        }
        dym_val[id_tr_] -= dt;
      }
      return t_spike;
    }
    //  Update conductance of fired neuron within single time step dt; it has the same hierachy level as the PrimelyUpdateState(double*, bool, Spike, double, bool);
    //  Description: operation to update neuronal state in primary level, ONE synaptic input most which arrives at the begining of time step;
    //  dym_val: array of dynamic variables;
    //  dt: size of time step, unit millisecond;
    //  return: none;
    inline void UpdateSource(double *dym_val, double dt) const override { UpdateG(dym_val, dt); }

    inline void GetDefaultDymVal(double* dym_val) const override {
      NeuronCore::GetDefaultDymVal(dym_val);
    }
};

typedef Neuron_LIF<LIF_G_Core>  Ty_LIF_G;
typedef Neuron_LIF<LIF_GH_Core> Ty_LIF_GH;
typedef Neuron_LIF<LIF_I_Core>  Ty_LIF_I;

// class Neuron_LIF_EXT:
// Implement basic operations for sub-timestep dynamics;
template <template<class> class NeuronCore, class ExtCurrent>
class Neuron_LIF_EXT: public NeuronCore<ExtCurrent>, public NeuronShellBase {
  using NeuronCore<ExtCurrent>::tau_;    // (ms) refractory Period;
  using NeuronCore<ExtCurrent>::resting_potential_;
  using NeuronCore<ExtCurrent>::threshold_potential_;
  using NeuronCore<ExtCurrent>::dym_n_;
  using NeuronCore<ExtCurrent>::id_v_;
  using NeuronCore<ExtCurrent>::id_gE_;
  using NeuronCore<ExtCurrent>::id_gI_;
  using NeuronCore<ExtCurrent>::id_tr_;
  using NeuronCore<ExtCurrent>::id_gE_inject_;
  using NeuronCore<ExtCurrent>::id_gI_inject_;
  using NeuronCore<ExtCurrent>::UpdateG;
  using NeuronCore<ExtCurrent>::GetDv;
  using NeuronCore<ExtCurrent>::DymInplaceRK4;
  public:
    typedef typename NeuronCore<ExtCurrent>::TyParaData T0;
    double DymCore(double * dym_val, double dt) const override { return 0.0; }
    int GetDymNum() const override { return dym_n_; }
    int GetIDV()    const override { return id_v_; }
    int GetIDGE()   const override { return id_gE_; }
    int GetIDGI()   const override { return id_gI_; }
    int GetIDTR()   const override { return id_tr_; }
    int GetIDGEInject() const override { return id_gE_inject_; } 
    int GetIDGIInject() const override { return id_gI_inject_; } 
    double GetRestingPotential() const override { return resting_potential_; }
    double GetRefTime() const override { return tau_; }

    double GetCurrent(double *dym_val) const override { 
      // Temporarily close this interface;
    }
    void SetRefTime(double t_ref) override { tau_ = t_ref; }
    void ManuallyFire(double *dym_val) const override {
      dym_val[id_v_] = resting_potential_;
      dym_val[id_tr_] = tau_;
    }
    //  Core operation for updating neuronal state within single timing step dt;
    //  Description: operation to update neuronal state in primary level, including updating conductances, membrane potential and checking spiking events; 
    //  dym_val: array of dynamic variables;
    //  dt: size of time step, unit ms;
    //  return: -1 for no spiking events; otherwise, return relative spiking time respect to the begining of the time step;
    //  Remark: if the input current (strength of synaptic input) is too large, or the neuron are at bursting state, the function might fail;
    double DymCore(double *dym_val, double dt, double t, T0 &x) const {
      double vn = dym_val[id_v_];
      // Update conductance;
      double dvn, dv_new;
      double t_spike = -1; // spike time within dt;
      if (dym_val[id_tr_] <= 0) { // neuron is not in the refractory period;
        dvn = DymInplaceRK4(dym_val, t, dt, x);
        // Check whether fire or not;
        if (dym_val[id_v_] > threshold_potential_) {
          dv_new = GetDv(dym_val, t, x);
          t_spike = cubic_hermite_root(dt, vn, dym_val[id_v_], dvn, dv_new, threshold_potential_);
          dym_val[id_v_] = resting_potential_;
          // update remaining fractory period
          dym_val[id_tr_] = tau_ + t_spike - dt;
          // if the refractory period is short enough, the neuron will be reactivate;
          if (dym_val[id_tr_] < 0) {
            // restore the source (driving current or conductance);
            UpdateG( dym_val, dym_val[id_tr_] );
            DymInplaceRK4( dym_val, t+dt+dym_val[id_tr_], -dym_val[id_tr_], x );
          }
        } 
      } else { // neuron is about to exit the refractory period;
        if (dym_val[id_tr_] < dt) {
          UpdateG(dym_val, dym_val[id_tr_]);
          dvn = DymInplaceRK4(dym_val, t, dt - dym_val[id_tr_], x);
          // Check whether fire or not;
          if (dym_val[id_v_] >= threshold_potential_) {
            dv_new = GetDv(dym_val, t, x);
            t_spike = cubic_hermite_root(dt - dym_val[id_tr_], vn, dym_val[id_v_], dvn, dv_new, threshold_potential_);
            dym_val[id_v_] = resting_potential_;
            // update remaining fractory period
            t_spike += dym_val[id_tr_];
            dym_val[id_tr_] = tau_ + t_spike;
          }
        } else { // neuron is in the refractory period;
          UpdateG(dym_val, dt);
        }
        dym_val[id_tr_] -= dt;
      }
      return t_spike;
    }
    //  Update conductance of fired neuron within single time step dt; it has the same hierachy level as the PrimelyUpdateState(double*, bool, Spike, double, bool);
    //  Description: operation to update neuronal state in primary level, ONE synaptic input most which arrives at the begining of time step;
    //  dym_val: array of dynamic variables;
    //  dt: size of time step, unit millisecond;
    //  return: none;
    inline void UpdateSource(double *dym_val, double dt) const override { UpdateG(dym_val, dt); }

    inline void GetDefaultDymVal(double* dym_val) const override {
      NeuronCore<ExtCurrent>::GetDefaultDymVal(dym_val);
    }
};

//typedef Neuron_LIF_EXT<LIF_G_Core>  LIF_G;
typedef Neuron_LIF_EXT<LIF_GH_EXT_Core, SineCore> LIF_GH_Sine;
//typedef Neuron_LIF_EXT<LIF_I_Core>  LIF_I;


// Baes class of neuronal simulator
class NeuronSimulatorBase {
  public:
    // "Get parameter" functions
    virtual int GetDymNum() const = 0; 
    virtual int GetIDV() const = 0;
    virtual int GetIDGE() const = 0; 
    virtual int GetIDGI() const = 0; 
    virtual void GetDefaultDymVal(double *dym_val) const = 0;
    virtual double GetCurrent(double* dym_val) const = 0;
    // "Set parameter" function
    virtual void SetRef(double t_ref) = 0;
    // "Evolving Dynamics" functions
    virtual double UpdateNeuronalState(double *dym_val, TyNeuronalInput &synaptic_driven, double t, double dt, vector<double>& new_spikes) const = 0;
    virtual void UpdateConductance(double *dym_val, TyNeuronalInput &synaptic_driven, double t, double dt) const = 0;

    virtual ~NeuronSimulatorBase() {  }
};

// NeuronSimulator: Based on integrate and fire neuron model;
template <class Neuron> // Neuron is based on NeuronShellBase
class NeuronSimulator : public Neuron, public NeuronSimulatorBase {
  using Neuron::GetIDGEInject;
  using Neuron::GetIDGIInject;
  using Neuron::DymCore;
  using Neuron::UpdateSource;
  public:
    int GetDymNum() const override { return Neuron::GetDymNum(); }
    int GetIDV() const override { return Neuron::GetIDV(); }
    int GetIDGE() const override { return Neuron::GetIDGE(); }
    int GetIDGI() const override { return Neuron::GetIDGI(); }
    void SetRef(double t_ref) override { Neuron::SetRefTime(t_ref); }
    void GetDefaultDymVal(double *dym_val) const override { Neuron::GetDefaultDymVal(dym_val); }
    double GetCurrent(double* dym_val) const override { return Neuron::GetCurrent(dym_val); }
    //  Update neuronal state:
    //  Description: update neuron within single time step, including its membrane potential, conductances and counter of refractory period;
    //  dym_val: dynamic variables;
    //  synaptic_driven: synaptic drivens;
    //  double t: time point of the begining of the time step;
    //  double dt: size of time step;
    //  vector<double> new_spikes: new spikes generated during dt;
    //  Return: membrane potential at t = t + dt;
    double UpdateNeuronalState(double *dym_val, TyNeuronalInput &synaptic_driven, double t, double dt, vector<double>& new_spikes) const override {
      new_spikes.clear();
      double tmax = t + dt;
      double t_spike;
      if (isnan(synaptic_driven.At().t) || synaptic_driven.At() >= tmax) {
        t_spike = DymCore(dym_val, dt);
        //cycle_ ++;
        if (t_spike >= 0) new_spikes.push_back(t_spike);
      } else {
        if (synaptic_driven.At() != t) {
          t_spike = DymCore(dym_val, synaptic_driven.At().t - t);
          //cycle_ ++;
          if (t_spike >= 0) new_spikes.push_back(t_spike);
        }
        size_t iter = 0;
        while (true) {
          // Update conductance due to the synaptic inputs;
          if (synaptic_driven.At(iter).type) dym_val[ GetIDGEInject() ] += synaptic_driven.At(iter).s;
          else dym_val[ GetIDGIInject() ] += synaptic_driven.At(iter).s;
          if (isnan(synaptic_driven.At(iter + 1).t) || synaptic_driven.At(iter + 1) >= tmax) {
            t_spike = DymCore(dym_val, tmax - synaptic_driven.At(iter).t);
            //cycle_ ++;
            if (t_spike >= 0) new_spikes.push_back(t_spike);
            break;
          } else {
            t_spike = DymCore(dym_val, synaptic_driven.At(iter + 1).t - synaptic_driven.At(iter).t);
            //cycle_ ++;
            if (t_spike >= 0) new_spikes.push_back(t_spike);
          }
          iter ++;
        }
      }
      return dym_val[Neuron::GetIDV()];
    }

    // Purely update conductances for fired neurons;
    void UpdateConductance(double *dym_val, TyNeuronalInput &synaptic_driven, double t, double dt) const override {
      double tmax = t + dt;
      if (isnan(synaptic_driven.At().t) || synaptic_driven.At() >= tmax) {
        UpdateSource(dym_val, dt);
      } else {
        if (synaptic_driven.At() != t) {
          UpdateSource(dym_val, synaptic_driven.At().t - t);
        }
        size_t iter = 0;
        while (true) {
          if (synaptic_driven.At(iter).type) dym_val[ GetIDGEInject() ] += synaptic_driven.At(iter).s;
          else dym_val[ GetIDGIInject() ] += synaptic_driven.At(iter).s;
          if (isnan(synaptic_driven.At(iter + 1).t) || synaptic_driven.At(iter + 1) >= tmax) {
            UpdateSource(dym_val, tmax - synaptic_driven.At(iter).t);
            break;
          } else {
            UpdateSource(dym_val, synaptic_driven.At(iter + 1).t - synaptic_driven.At(iter).t);
          }
          iter ++;
        }
      }
    }
};

typedef NeuronSimulator<Ty_LIF_G>  Sim_LIF_G;
typedef NeuronSimulator<Ty_LIF_GH> Sim_LIF_GH;
typedef NeuronSimulator<Ty_LIF_I>  Sim_LIF_I;

// NeuronSimulator: Based on integrate and fire neuron model;
template <class Neuron> // Neuron is based on NeuronShellBase
class NeuronExtCurrentSimulator : public Neuron, public NeuronSimulatorBase {
  using Neuron::GetIDGEInject;
  using Neuron::GetIDGIInject;
  using Neuron::DymCore;
  using Neuron::UpdateSource;
  public:
    double UpdateNeuronalState(double *dym_val, TyNeuronalInput &synaptic_driven, double t, double dt, vector<double>& new_spikes) const override { return 0.0; }
    typedef typename Neuron::T0 TyNeuronPara;
    int GetDymNum() const override { return Neuron::GetDymNum(); }
    int GetIDV() const override { return Neuron::GetIDV(); }
    int GetIDGE() const override { return Neuron::GetIDGE(); }
    int GetIDGI() const override { return Neuron::GetIDGI(); }
    void GetDefaultDymVal(double *dym_val) const override { Neuron::GetDefaultDymVal(dym_val); }
    double GetCurrent(double* dym_val) const override {  } // currently not supported

    void SetRef(double t_ref) override { Neuron::SetRefTime(t_ref); }
    //  Update neuronal state:
    //  Description: update neuron within single time step, including its membrane potential, conductances and counter of refractory period;
    //  dym_val: dynamic variables;
    //  synaptic_driven: synaptic drivens;
    //  x              : Parameters for external sine drive;
    //  double t: time point of the begining of the time step;
    //  double dt: size of time step;
    //  vector<double> new_spikes: new spikes generated during dt;
    //  Return: membrane potential at t = t + dt;
    double UpdateNeuronalState(double *dym_val, TyNeuronalInput &synaptic_driven, TyNeuronPara & x, double t, double dt, vector<double>& new_spikes) const {
      new_spikes.clear();
      double tmax = t + dt;
      double t_spike;
      if (isnan(synaptic_driven.At().t) || synaptic_driven.At() >= tmax) {
        t_spike = DymCore(dym_val, dt, t, x);
        //cycle_ ++;
        if (t_spike >= 0) new_spikes.push_back(t_spike);
      } else {
        if (synaptic_driven.At() != t) {
          t_spike = DymCore(dym_val, synaptic_driven.At().t - t, t, x);
          //cycle_ ++;
          if (t_spike >= 0) new_spikes.push_back(t_spike);
        }
        size_t iter = 0;
        while (true) {
          // Update conductance due to the synaptic inputs;
          if (synaptic_driven.At(iter).type) dym_val[ GetIDGEInject() ] += synaptic_driven.At(iter).s;
          else dym_val[ GetIDGIInject() ] += synaptic_driven.At(iter).s;
          if (isnan(synaptic_driven.At(iter + 1).t) || synaptic_driven.At(iter + 1) >= tmax) {
            t_spike = DymCore(dym_val, tmax - synaptic_driven.At(iter).t, t, x);
            //cycle_ ++;
            if (t_spike >= 0) new_spikes.push_back(t_spike);
            break;
          } else {
            t_spike = DymCore(dym_val, synaptic_driven.At(iter + 1).t - synaptic_driven.At(iter).t, t, x);
            //cycle_ ++;
            if (t_spike >= 0) new_spikes.push_back(t_spike);
          }
          iter ++;
        }
      }
      return dym_val[Neuron::GetIDV()];
    }

    // Purely update conductances for fired neurons;
    void UpdateConductance(double *dym_val, TyNeuronalInput &synaptic_driven, double t, double dt) const override {
      double tmax = t + dt;
      if (isnan(synaptic_driven.At().t) || synaptic_driven.At() >= tmax) {
        UpdateSource(dym_val, dt);
      } else {
        if (synaptic_driven.At() != t) {
          UpdateSource(dym_val, synaptic_driven.At().t - t);
        }
        size_t iter = 0;
        while (true) {
          if (synaptic_driven.At(iter).type) dym_val[ GetIDGEInject() ] += synaptic_driven.At(iter).s;
          else dym_val[ GetIDGIInject() ] += synaptic_driven.At(iter).s;
          if (isnan(synaptic_driven.At(iter + 1).t) || synaptic_driven.At(iter + 1) >= tmax) {
            UpdateSource(dym_val, tmax - synaptic_driven.At(iter).t);
            break;
          } else {
            UpdateSource(dym_val, synaptic_driven.At(iter + 1).t - synaptic_driven.At(iter).t);
          }
          iter ++;
        }
      }
    }
};

typedef NeuronExtCurrentSimulator<LIF_GH_Sine> Sim_LIF_GH_Sine;

#endif  // _NEURON_H_
