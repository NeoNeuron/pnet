import numpy as np
import configparser as cp
import subprocess as sp

class network:
    '''
    network(ne, ni=0)

    Class to config and run network simulation.

    Define network structure, config simulation parameters and run executable 
    network simulation program.

    Parameters
    ----------
    ne : int
        number of excitatory neurons
    ni : int, optional
        number of inhibitory neurons

    Examples
    --------
    >>> # simulate an all connected 100 LIF_GH neurons network
    >>> a = network(80, 20)
    >>> mat = np.zeros((100,100), dtype=int)
    >>> np.fill_diagonal(mat, 0)
    >>> dmat = np.zeros((100,100))
    >>> pm = {'model': 'LIF_GH', 'simulator' : 'SSC', 'tref': 2.0,
    ...:  'see' : 1e-3, 'sie' : 1e-3, 'sei' : 5e-3, 'sii' : 5e-3,
    ...:  'synapse_file': 'smat.npy',
    ...:  'con_mat' : mat,
    ...:  'space_file': 'dmat.npy',
    ...:  'delay_mat' : dmat,
    ...:  'pre_e' : 1.5, 'pse_e' : 5e-3,
    ...:  'pre_i' : 1.5, 'pse_i' : 5e-3,
    ...:  'poisson_file': 'PoissonSetting.csv', 'poisson_seed': 3,
    ...:  'T': 1e3, 'dt': 0.03125, 'stp': 0.5,
    ...:  'v_flag': True}
    >>> a.add(**pm)
    >>> a.show()
    ========================================
    Neuron Population:
            ne        ni        ke        ki
            80        20      99.0      99.0
    ----------------------------------------
    Synapses:
           see       sie       sei       sii
      1.00e-03  1.00e-03  5.00e-03  5.00e-03
    ----------------------------------------
    FFWD Poisson:
           pre       pse       pri       psi
      1.50e+00  5.00e-03  0.00e+00  0.00e+00
    ----------------------------------------
    Afferent Connection:
       99   99   99   99   99   99   99   99
    ========================================
    >>> a.updatefiles()
    >>> a.run('verbose')
    (number of connections in sparse-mat 9900)
    >> Initialization :     0.007 s
    >> Done!
    >> Simulation :         0.341 s
    Total inter-neuronal interaction : 112563
    Mean firing rate : 11.37 Hz

    '''
    def __init__(self, ne, ni=0):
        self.ne = ne    # No. of exc neurons
        self.n = ne+ni  # No. of neurons

        self.see = 0.0  # Synaptic strength from E to E
        self.sie = 0.0  # Synaptic strength from E to I
        self.sei = 0.0  # Synaptic strength from I to E
        self.sii = 0.0  # Synaptic strength from I to I

        self.mat  = np.zeros((self.n, self.n), dtype=np.int32)
        self.smat = np.zeros((self.n, self.n), dtype=np.float64)
        self.dmat = np.zeros((self.n, 2), dtype=np.float64)
        self.pmat = np.zeros((self.n, 4), dtype=np.float64) # Poisson rate in unit kHz
        # pmat[:,0] : Exc. Poisson rate
        # pmat[:,1] : Inh. Poisson rate
        # pmat[:,2] : Exc. Poisson strength
        # pmat[:,3] : Inh. Poisson strength

        # dictionary of config
        self.config = cp.ConfigParser()
        self.config['network'] = {
                'ne' : str(ne),
                'ni' : str(ni),
                'simulator' : 'Simple',
                }
        self.config['neuron'] = {
                'model' : 'LIF_GH',
                'tref'  : '2.0',
                }
        self.config['synapse'] = {
                'file' : 'smat.npy',
                }
        self.config['space'] = {
                'file'  : 'dmat.npy',
                }
        self.config['driving'] = {
                'file'  : 'PoissonSetting.csv',
                'seed'  : '3',
                }
        self.config['time'] = {
                't'   : '1e3',
                'dt'  : str(1/32),
                'stp' : '0.5',
                }
        self.config['output'] = {
                'poi' : 'false',
                'v'   : 'false',
                'i'   : 'false',
                'ge'  : 'false',
                'gi'  : 'false',
                }

    def run(self, prefix = './', *args):
        '''
        Run network simulation.

        Parameters
        ----------
        prefix : string
            Data ouput folder.

        '''
        cml_options = 'bin/net_sim --prefix ' + prefix + ' '
        for section in self.config.sections():
            for option in list(self.config[section]):
                cml_options += '--'+section+'.'+option + ' ' + (self.config[section][option]) + ' '
        if 'verbose' in args:
            cml_options += '-v'
        sp.call(cml_options.split(' '))

    def show(self):
        '''
        Print info of current network.

        '''
        lline = lambda x : print('='*x)
        line  = lambda x : print('-'*x)
        lline(40)
        print("Neuron Population:")
        print(('{:>10}'*4).format('ne','ni','ke','ki'))
        print(('{:>10}'*4).format(self.ne, self.n - self.ne, self.mat[:,:self.ne].sum(0).mean(0), self.mat[:,self.ne:].sum(0).mean(0)))
        line(40)
        print("Synapses:")
        print(('{:>10s}'*4).format('see','sie','sei','sii'))
        print(('{:>10.2e}'*4).format(self.see, self.sie, self.sei, self.sii))
        line(40)
        print("FFWD Poisson:")
        print(('{:>10s}'*4).format('pre','pri','pse','psi'))
        print(('{:>10.2e}'*4).format(*self.pmat.mean(0)))
        line(40)
        print("Afferent Connection:")
        if self.n >= 8:
            print(('{:>5.0f}'*8).format(*self.mat.sum(1)[:8]))
        else:
            print(('{:>5.0f}'*self.n).format(*self.mat.sum(1)))
        lline(40)


    def config_dict(self):
        '''
        Return dictionary of configurations.

        '''
        config_dict = {key:dict(self.config[key]) for key in self.config if key != 'DEFAULT'}
        return config_dict

    def add(self, model = None, simulator = None,
            tref = None,
            con_mat = None,
            see = None, sie = None, sei = None, sii = None,
            synapse_file = None,
            space_file = None,
            delay_mat = None,
            pre_e = None, pse_e = None, pri_e = None, psi_e = None,
            pre_i = None, pse_i = None, pri_i = None, psi_i = None,
            poisson_file = None,
            poisson_seed = None,
            T = None, dt = None, stp = None,
            poisson_flag = None,
            v_flag = None, i_flag = None, ge_flag = None, gi_flag = None
            ):
        '''
        Add config file.

        Parameters
        ----------

        model : string
            Type of neuronal model.
        simulator : string
            Type of network simulator.
        tref : float
            Refractory period.
        con_mat : array_like data of int
            Adjacent matrix.
        see : float
            synaptic strength from E to E.
        sie : float
            synaptic strength from E to I.
        sei : float
            synaptic strength from I to E.
        sii : float
            synaptic strength from I to I.
        synapse_file : string
            File of synaptic matrix.
        space_file : string
            File of delay matrix.
        delay_mat : array_like data of float
            Matrix of interaction delay between neurons.
        pre_e : float
            FFWD Exc. Poisson rate for Exc. neurons.
        pse_e : float
            FFWD Exc. Poisson strength for Exc. neurons.
        pri_e : float
            FFWD Inh. Poisson rate for Exc. neurons.
        psi_e : float
            FFWD Inh. Poisson strength for Exc. neurons.
        pre_i : float
            FFWD Exc. Poisson rate for Inh. neurons.
        pse_i : float 
            FFWD Exc. Poisson strength for Inh. neurons.
        pri_i : float
            FFWD Inh. Poisson rate for Inh. neurons.
        psi_i : float
            FFWD Inh. Poisson strength for Inh. neurons.
        poisson_file : string
            File of Poisson drive.
        poisson_seed :
            Seed of FFWD Poisson process
        T : float
            Total simulation time period
        dt : float
            Simulation time step
        stp : float
            Recording time step
        poisson_flag : bool
            Output Poisson time sequence if True
        v_flag : bool
            Output V time series if True
        i_flag : bool
            Output I time series if True
        ge_flag : bool
            Output GE time series if True
        gi_flag : bool
            Output GI time series if True

        '''

        if model is not None:
            model_pool = ('LIF_I', 'LIF_G', 'LIF_GH')
            if model in model_pool:
                self.config['neuron']['model']  = model 
            else:
                print('Warning: invalid model type. Default model (LIF_GH) applied')

        if simulator is not None:
            simulator_pool = ('Simple', 'SSC', 'SSC_Sparse')
            if simulator in simulator_pool:
                self.config['network']['simulator']  = simulator 
            else:
                print('Warning: invalid simulator type. Default simulator (Simple) applied')

        if tref is not None:
            self.config['neuron']['tref']   = str(tref)
        if con_mat is not None:
            if type(con_mat) in (np.ndarray, list):
                if len(con_mat) == self.n and len(con_mat[0]) == self.n:
                    self.mat = np.array(con_mat)
                else:
                    print('Warning: invalid data. ({:d},{:d}) array_like data required'
                            .format(self.n, self.n))
            else:
                print('Warning: invalid data. ({:d},{:d}) array like data required'
                        .format(self.n, self.n))
            
        if see is not None:
            self.see = float(see)
        if sie is not None:
            self.sie = float(sie)
        if sei is not None:
            self.sei = float(sei)
        if sii is not None:
            self.sii = float(sii)
        if synapse_file is not None:
            self.config['synapse']['file']  = synapse_file
        if delay_mat is not None:
            if type(delay_mat) in (np.ndarray, list):
                if len(delay_mat) == self.n and len(delay_mat[0]) == self.n:
                    self.dmat = np.array(delay_mat)
                else:
                    print('Warning: invalid delay matrix. ({:d},{:d}) array like data required'
                            .format(self.n, self.n))
            else:
                print('Warning: invalid delay matrix. ({:d},{:d}) array like data required'
                        .format(self.n, self.n))

        if space_file is not None:
            self.config['space']['file']    = space_file
        if pre_e is not None:
            self.pmat[:self.ne, 0] = float(pre_e)
        if pri_e is not None:       
            self.pmat[:self.ne, 1] = float(psi_e)
        if pse_e is not None:       
            self.pmat[:self.ne, 2] = float(pse_e)
        if psi_e is not None:       
            self.pmat[:self.ne, 3] = float(psi_e)
        if pre_i is not None:
            self.pmat[self.ne:, 0] = float(pre_i)
        if pri_i is not None:       
            self.pmat[self.ne:, 1] = float(pri_i)
        if pse_i is not None:       
            self.pmat[self.ne:, 2] = float(pse_i)
        if psi_i is not None:       
            self.pmat[self.ne:, 3] = float(psi_i)
        if poisson_file is not None:
            self.config['driving']['file']  = poisson_file
        if poisson_seed is not None:
            self.config['driving']['seed']  = str(poisson_seed)
        if T is not None:
            self.config['time']['t']        = str(T)
        if dt is not None:
            self.config['time']['dt']       = str(dt)
        if stp is not None:
            self.config['time']['stp']      = str(stp)
        if poisson_flag is not None:
            self.config['output']['poi']    = str(poisson_flag).lower()
        if v_flag is not None:
            self.config['output']['v']      = str(v_flag).lower()
        if i_flag is not None:
            self.config['output']['i']      = str(i_flag).lower()
        if ge_flag is not None:
            self.config['output']['ge']     = str(ge_flag).lower()
        if gi_flag is not None:
            self.config['output']['gi']     = str(gi_flag).lower()

    def saveconfig(self, path_to_file):
        '''
        Save config to INI file.

        '''
        with open(path_to_file, 'w') as configfile:
            self.config.write(configfile)

    def updatefiles(self, prefix='./'):
        '''
        Update parameter files.

        Including synapse file, space file, and driving file.

        '''
        self.smat[:self.ne,:self.ne] = self.mat[:self.ne,:self.ne] * self.see
        self.smat[self.ne:,:self.ne] = self.mat[self.ne:,:self.ne] * self.sie
        self.smat[:self.ne,self.ne:] = self.mat[:self.ne,self.ne:] * self.sei
        self.smat[self.ne:,self.ne:] = self.mat[self.ne:,self.ne:] * self.sii

        np.save(prefix + self.config['synapse']['file'], self.smat)
        np.save(prefix + self.config['space']['file'], self.dmat)
        np.savetxt(prefix + self.config['driving']['file'], self.pmat, delimiter = ',', fmt = '%e')
        print("config prepared!")
