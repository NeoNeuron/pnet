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
    >>> mat = np.fill_diagonal(np.ones((100,100), dtype=int), 0)
    >>> pm = {'model': 'LIF_GH', 'tref': 2.0,
    ...:  'see' : 1e-3, 'sie' : 1e-3, 'sei' : 5e-3, 'sii' : 5e-3,
    ...:  'synapse_file': 'smat.npy',
    ...:  'con_mat' : mat,
    ...:  'pre' : 1.5, 'pse' : 5e-3,
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
         0.001     0.001     0.005     0.005
    ----------------------------------------
    FFWD Poisson:
           pre       pse       pri       psi
           1.5     0.005       0.0       0.0
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
        self.ne = ne    # No. of exc neuron
        self.ni = ni    # No. of inh neuron

        self.see = 0.0  # Synaptic strength from E to E
        self.sie = 0.0  # Synaptic strength from E to I
        self.sei = 0.0  # Synaptic strength from I to E
        self.sii = 0.0  # Synaptic strength from I to I

        self.pre = 0.0  # excitatory Poisson rate, unit kHz
        self.pse = 0.0  # excitatory Poisson strength
        self.pri = 0.0  # inhibitory Poisson rate, unit kHz
        self.psi = 0.0  # inhibitory Poisson strength

        N = ne+ni       # total neuron number
        self.mat  = np.zeros((N, N), dtype=np.int32)
        self.smat = np.zeros((N, N), dtype=np.float64)
        self.grid = np.zeros((N, 2), dtype=np.float64)
        self.pmat = np.zeros((N, 4), dtype=np.float64)

        # dictionary of config
        self.config = cp.ConfigParser()
        self.config['network'] = {
                'ne' : str(ne),
                'ni' : str(ni),
                }
        self.config['neuron'] = {
                'model' : 'LIF_GH',
                'tref'  : '2.0',
                }
        self.config['synapse'] = {
                'file' : 'smat.npy',
                }
        self.config['space'] = {
                'mode'  : '-1',
                'delay' : '3.0',
                'speed' : '0.3',
                'file'  : 'coordinate.csv',
                }
        self.config['driving'] = {
                'file'  : 'PoissonSetting.csv',
                'seed'  : '3',
                'gmode' : 'true',
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
        print(('{:>10}'*4).format(self.ne, self.ni, self.mat[:,:self.ne].sum(0).mean(0), self.mat[:,self.ne:].sum(0).mean(0)))
        line(40)
        print("Synapses:")
        print(('{:>10}'*4).format('see','sie','sei','sii'))
        print(('{:>10}'*4).format(self.see, self.sie, self.sei, self.sii))
        line(40)
        print("FFWD Poisson:")
        print(('{:>10}'*4).format('pre','pse','pri','psi'))
        print(('{:>10}'*4).format(self.pre, self.pse, self.pri, self.psi))
        line(40)
        print("Afferent Connection:")
        if self.ne+self.ni >= 8:
            print(('{:>5}'*8).format(*self.mat.sum(1)[:8]))
        else:
            print(('{:>5}'*(self.ne+self.ni)).format(*self.mat.sum(1)))
        lline(40)


    def config_dict(self):
        '''
        Return dictionary of configurations.

        '''
        config_dict = {key:dict(self.config[key]) for key in self.config if key != 'DEFAULT'}
        return config_dict

    def add(self, model = None,
            tref = None,
            con_mat = None,
            see = None, sie = None, sei = None, sii = None,
            synapse_file = None,
            space_file = None,
            spatial_grid = None,
            pre = None, pse = None, pri = None, psi = None,
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
            File of spatial grid list.
        spatial_grid : array_like data of float
            Array of spatial grid.
        pre : float
            FFWD Poisson E rate.
        pse : float       
            FFWD Poisson E strength.
        pri : float       
            FFWD Poisson I rate.
        psi : float       
            FFWD Poisson I strength.
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

        if tref is not None:
            self.config['neuron']['tref']   = str(tref)
        if con_mat is not None:
            if type(con_mat) in (np.ndarray, list):
                if len(con_mat) == self.ne + self.ni and len(con_mat[0]) == self.ne + self.ni:
                    self.mat = np.array(con_mat)
                else:
                    print('Warning: invalid data. ({:d},{:d}) array_like data required'.format(self.ne + self.ni, self.ne + self.ni))
            else:
                print('Warning: invalid data. ({:d},{:d}) array like data required'.format(self.ne + self.ni, self.ne + self.ni))
            
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
        if spatial_grid is not None:
            if type(spatial_grid) in (np.ndarray, list):
                if len(spatial_grid) == self.ne + self.ni and len(spatial_grid[0]) == 2:
                    self.grid = np.array(spatial_grid)
                else:
                    print('Warning: invalid grid. ({:d},{:d}) array like data required'.format(self.ne + self.ni, 2))
            else:
                print('Warning: invalid grid. ({:d},{:d}) array like data required'.format(self.ne + self.ni, 2))

        if space_file is not None:
            self.config['space']['file']    = space_file
        #    self.config['space']['mode']     = '-1'
        #    self.config['space']['delay']    = '3.0'
        #    self.config['space']['speed']    = '0.3'
        if pre is not None:
            self.pre = float(pre)
        if pse is not None:       
            self.pse = float(pse)
        if pri is not None:       
            self.pri = float(pri)
        if psi is not None:       
            self.psi = float(psi)
        if poisson_file is not None:
            self.config['driving']['file']  = poisson_file
        if poisson_seed is not None:
            self.config['driving']['seed']  = str(poisson_seed)
        #    self.config['driving']['gmode']  = 'true'
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
            self.config['output']['ge']     = str(gi_flag).lower()

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
        np.savetxt(prefix + self.config['space']['file'], self.grid, delimiter = ',', fmt = '%f')
        self.pmat[:,0] = self.pre
        self.pmat[:,1] = self.pse
        self.pmat[:,2] = self.pri
        self.pmat[:,3] = self.psi
        np.savetxt(prefix + self.config['driving']['file'], self.pmat, delimiter = ',', fmt = '%e')
        print("config prepared!")
