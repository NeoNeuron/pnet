import numpy as np
from tabulate import tabulate
import configparser as cp
import subprocess as sp
import simtools.adjmat as adjmat
import simtools.spacemat as spacemat

# rewrite the configparser.ConfigParser
class MyConfigParser(cp.ConfigParser):
    def __init__(self,defaults=None):
        cp.ConfigParser.__init__(self,defaults=None)
    def optionxform(self, optionstr):
        return optionstr
#---------------------

class network:
    def __init__(self, Ne, Ni=0):
        self.Ne = Ne   # No. of exc neuron
        self.Ni = Ni   # No. of inh neuron
        self.K  = 0    # connection degree

        self.see = 0.0
        self.sie = 0.0
        self.sei = 0.0
        self.sii = 0.0

        self.pre = 0.0     # unit Hz
        self.pse = 0.0     # 
        self.pri = 0.0     # unit Hz
        self.psi = 0.0     # 

        neuron_number = Ne+Ni
        self.mat = np.empty((neuron_number, neuron_number), dtype=int)
        self.smat = np.empty((neuron_number, neuron_number), dtype=float)
        self.gd = np.empty((neuron_number, 2), dtype=float)
        self.pmat = np.empty((neuron_number, 4), dtype=float)
        self.sine_mat = np.empty((neuron_number, 3), dtype=float)
        self.visual_mask = np.zeros(neuron_number,dtype=int)
        self.attend_type_mask = np.zeros(neuron_number, dtype=int)
        self.attend_mask = np.zeros(neuron_number, dtype=int)
        self.visual_strength = 0.0
        self.attend_strength = 0.0
        # dictionary of config
        self.config = MyConfigParser()
        self.config.add_section('network')
        self.config['network']['Ne']   = str(Ne) 
        self.config['network']['Ni']   = str(Ni) 
        #---
        self.config.add_section('neuron')
        self.config['neuron']['model']   = 'LIF_GH'
        self.config['neuron']['tref']    = '2.0'
        #---
        self.config.add_section('synapse')
        self.config['synapse']['file']   = 'smat.npy'
        #---
        self.config.add_section('space')
        self.config['space']['mode']     = '-1'
        self.config['space']['delay']    = '3.0'
        self.config['space']['speed']    = '0.3'
        self.config['space']['file']     = 'coordinate.csv'
        #---
        self.config.add_section('driving')
        self.config['driving']['file']   = 'PoissonSetting.csv'
        self.config['driving']['seed']   = '3'
        self.config['driving']['gmode']  = 'true'
        #---
        self.config.add_section('sine')
        self.config['sine']['file']    = 'sine_para.csv'
    #    self.config['sine']['amplitude'] = '0.0'
    #    self.config['sine']['frequency'] = '0.0'
    #    self.config['sine']['phase']     = '0.0'
        #---
        self.config.add_section('time')
        self.config['time']['T']         = '' 
        self.config['time']['dt']        = '0.03125'
        self.config['time']['stp']       = '0.5'
        #---
        self.config.add_section('output')
        self.config['output']['poi']     = 'false'
        self.config['output']['V']       = 'false'
        self.config['output']['I']       = 'false'
        self.config['output']['GE']      = 'false'
        self.config['output']['GI']      = 'false'
        #---

    def show(self):
        """
        Print info of current network.
        """
        print("="*20)
        print("Neuron Population:")
        print(tabulate([[self.Ne, self.Ni, self.mat[:,:self.Ne].sum(0).mean(0), self.mat[:,self.Ne:].sum(0).mean(0)]],headers=['Ne','Ni','Ke','Ki'], tablefmt="grid"))
        print("Synapses:")
        print(tabulate([[self.see, self.sie, self.sei, self.sii]],headers=['see','sie','sei','sii'], tablefmt="grid"))
        print("FFWD Poisson:")
        print(tabulate([[self.pre, self.pse, self.pri, self.psi]],headers=['pre','pse','pri','psi'], tablefmt="grid"))
        print("Adjacent Matrix:")
        if self.Ne+self.Ni >= 10:
            print(tabulate(self.mat[:10,:10]))
        else:
            print(tabulate(self.mat))
        print("="*20)
    def showConfig(self):
        print("Current Configurations:")
        for section in self.config.sections():
            print(section)
            print(tabulate(self.config.items(section)))

    def SimSetting(self, neuron_model=None,
            synapse_file=None, space_file=None, poisson_file=None, 
            T=None, dt=None, stp=None, poisson_seed=None,
            poisson_flag=None, V_flag=None, I_flag=None, GE_flag=None, GI_flag=None):
        """
        Edit config file

        Parameters
        ==========
        neuron_model : string
            type of neuronal model

        synapse_file : string
            file of synaptic matrix

        space_file : string
            file of spatial grid list

        poisson_file : string
            file of Poisson drive

        T : float
            Total simulation time period

        dt : float
            Simulation time step

        stp : float
            Recording time step

        poisson_flag : bool
            True for outputting Poisson time sequence, otherwise False

        V_flag : bool
            True for outputting V time series, otherwise False

        I_flag : bool
            True for outputting I time series, otherwise False

        GE_flag : bool
            True for outputting GE time series, otherwise False

        GI_flag : bool
            True for outputting GI time series, otherwise False

        """
        #---
        if neuron_model != None:
            self.config['neuron']['model']  = neuron_model 
        #    self.config['neuron']['tref']    = '2.0'
        #---
        if synapse_file != None:
            self.config['synapse']['file']  = synapse_file
        #---
        #    self.config['space']['mode']     = '-1'
        #    self.config['space']['delay']    = '3.0'
        #    self.config['space']['speed']    = '0.3'
        if space_file != None:
            self.config['space']['file']    = space_file
        #---
        if poisson_file != None:
            self.config['driving']['file']  = poisson_file
        if poisson_seed != None:
            self.config['driving']['seed']   = str(poisson_seed)
        else:
            self.config['driving']['seed']   = str(np.random.randint(1e18))
        #    self.config['driving']['gmode']  = 'true'
        #---
        if T != None:
            self.config['time']['T']        = str(T)
        if dt != None:
            self.config['time']['dt']       = str(dt)
        if stp != None:
            self.config['time']['stp']      = str(stp)
        #---
        if poisson_flag != None:
            self.config['output']['poi']    = str(poisson_flag).lower()
        if V_flag != None:
            self.config['output']['V']      = str(V_flag).lower()
        if I_flag != None:
            self.config['output']['I']      = str(I_flag).lower()
        if GE_flag != None:
            self.config['output']['GE']     = str(GE_flag).lower()
        if GI_flag != None:
            self.config['output']['GI']     = str(GI_flag).lower()

    def UpdateConfig(self, prefix='./'):
#        # =============
#        # set sine para
#        # =============
#        self.config['sine']['amplitude'] = ' '.join(map(str, self.sine_mat[:,0]))
#        self.config['sine']['frequency'] = ' '.join(map(str, self.sine_mat[:,1]))
#        self.config['sine']['phase']     = ' '.join(map(str, self.sine_mat[:,2]))

        with open(prefix + '/config.ini', 'w') as configfile:
            self.config.write(configfile)
        #========================================

        np.savetxt(prefix + 'visual_stimuli_mask.csv', self.visual_mask, fmt = '%d')
        np.savetxt(prefix + 'attend_type_mask.csv', self.attend_type_mask, fmt = '%d')
        np.savetxt(prefix + 'attend_mask.csv', self.attend_mask, fmt = '%d')
        
        self.smat[:self.Ne,:self.Ne] = self.mat[:self.Ne,:self.Ne] * self.see
        self.smat[self.Ne:,:self.Ne] = self.mat[self.Ne:,:self.Ne] * self.sie
        self.smat[:self.Ne,self.Ne:] = self.mat[:self.Ne,self.Ne:] * self.sei
        self.smat[self.Ne:,self.Ne:] = self.mat[self.Ne:,self.Ne:] * self.sii

        np.save(prefix + self.config['synapse']['file'], self.smat)

        np.savetxt(prefix + self.config['space']['file'], self.gd, delimiter = ',', fmt = '%f')

        self.pmat[:,0] = self.pre
        self.pmat[:,1] = self.pse
        self.pmat[:,2] = self.pri
        self.pmat[:,3] = self.psi

        #self.pmat[:,1] += self.visual_strength*self.visual_mask
        self.pmat[:,1] += self.attend_strength*(self.attend_mask*(self.attend_type_mask == 1))
        self.pmat[:,3] += self.attend_strength*(self.attend_mask*(self.attend_type_mask == 2))
        np.savetxt(prefix + self.config['driving']['file'], self.pmat, delimiter = ',', fmt = '%e')
        np.savetxt(prefix + self.config['sine']['file'], self.sine_mat, delimiter = ',', fmt = '%e')
        print("Config prepared!")

    def Run(self, prefix = './'):
        sp.call(['/Users/kchen/github/pnet/bin/net_sim', '-c', 'config.ini', '--prefix', prefix])

    def SetAdjMat(self, mat):
        self.mat = mat

    def SetPoisson(self, pre, pse, pri=0.0, psi=0.0):
        self.pre = pre
        self.pse = pse
        self.pri = pri
        self.psi = psi

    def SetSine(self, amplitude, frequency, phase):
        if (type(amplitude)==float):
            self.sine_mat[:,0] = np.ones(self.Ne + self.Ni)*amplitude
        elif (type(amplitude)==np.ndarray):
            self.sine_mat[:,0] = amplitude
        if (type(frequency)==float):
            self.sine_mat[:,1] = np.ones(self.Ne + self.Ni)*frequency
        elif (type(frequency)==np.ndarray):
            self.sine_mat[:,1] = frequency
        if (type(phase)==float):
            self.sine_mat[:,2] = np.ones(self.Ne + self.Ni)*phase
        elif (type(phase)==np.ndarray):
            self.sine_mat[:,2] = phase


    def SetSynapse(self, see=0.0, sie=0.0, sei=0.0, sii=0.0):
        self.see = see
        self.sie = sie
        self.sei = sei
        self.sii = sii

    def SetSpatialMat(self, gd):
        self.gd = gd

    def SetVisualStimuli(self, pos_x, pos_y, radius, strength):
        """
        Set mask for visual stimuli, overlapping is permitted.

        """
        vis_pos = np.array([pos_x, pos_y])
        self.visual_mask += (((self.gd - vis_pos)**2).sum(1)<=radius**2).astype(int)
        self.visual_strength = strength

    def SetAttentionType(self, n_ae, n_ai, seed=None):
        # 0 for un-tunned type
        # 1 for attention excitatory
        # 2 for attention inhibitory
        if seed != None:
            np.random.seed(seed)
        chosen_idx = np.random.choice(np.arange(int(self.Ne + self.Ni)), n_ae + n_ai, replace=False)
        self.attend_type_mask[chosen_idx[:n_ae]] = 1
        self.attend_type_mask[chosen_idx[n_ae:]] = 2

    def SetAttention(self, pos_x, pos_y, radius, strength):
        attend_pos = np.array([pos_x, pos_y])
        self.attend_mask = (((self.gd - attend_pos)**2).sum(1)<=radius**2).astype(int)
        self.attend_strength = strength
