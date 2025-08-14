from .MM.sc_system import *
from .NN.pgm import *

from pymbar import MBAR

## ## 

class NN_interface_helper:
    def __init__(self,):
        self.dir_results_BAR = DIR_main+'/NN/training_results/BAR/'
        self.dir_results_misc = DIR_main+'/NN/training_results/misc/'
        self.dir_results_samples = DIR_main+'/NN/training_results/samples/'
        self.dir_results_models = DIR_main+'/NN/training_results/fitted_models/'
        self.inds_rand = None

    @property
    def name_save_misc(self):
        return self.dir_results_misc + self.name + '_misc_'
    def save_misc_(self,):
        self.trainer.save_the_above_(self.name_save_misc)
    # NN_interface_sc_multimap has load

    @property
    def name_save_inds_rand(self,):
        return self.dir_results_misc + self.name + '_inds_rand_'
    def save_inds_rand_(self, key=''):
        save_pickle_(self.inds_rand, self.name_save_inds_rand+key)
    def load_inds_rand_(self, key=''):
        self.inds_rand = load_pickle_(self.name_save_inds_rand+key)

    @property
    def name_save_inv_test(self):
        return self.dir_results_misc + self.name + '_inv_test_'
    def save_inv_test_results_(self,):
        self.trainer.save_inv_test_results_(self.name_save_inv_test)
    def load_inv_test_results_(self):
        self.inv_test_results = load_pickle_(self.name_save_inv_test)

    @property
    def name_save_BAR_inputs(self):
        return self.dir_results_BAR + self.name + '_BAR_'
    # saving inside trainer
    # loading : here, solve_BAR_using_pymbar_
    
    @property
    def name_save_mBAR_inputs(self):
        return self.dir_results_BAR + self.name + '_mBAR_'
    # saving inside trainer
    # loading : in NN_interface_sc_multimap, solve_mBAR_using_pymbar_

    @property
    def name_save_samples(self):
        return self.dir_results_samples + self.name + '_samples_'
    # mNN_interface_sc_multimap has save and load

    @property
    def name_save_model(self):
        return self.dir_results_models + self.name + '_model_'
    def save_model_(self):
        self.model.save_model(self.name_save_model)
    def load_model_(self, VERSION='NEW'):
        if VERSION == 'NEW':
            self.model = self.model_class.load_model(self.name_save_model)
        else:
            # need this step to unpickle those files since working directory was moves up by one folder
            from . import NN
            sys.modules['NN'] = NN
            sys.modules['NN.util_tf'] = NN.util_tf
            sys.modules['NN.util_tf.WhitenFlow'] = NN.util_tf.WhitenFlow
            sys.modules['NN.util_tf.Static_Rotations_Layer'] = NN.util_tf.Static_Rotations_Layer
            sys.modules['NN.util_tf.FocusedHemisphere'] = NN.util_tf.FocusedHemisphere
            sys.modules['NN.util_tf.FocusedAngles'] = NN.util_tf.FocusedAngles
            sys.modules['NN.util_tf.FocusedBonds'] = NN.util_tf.FocusedBonds
            sys.modules['NN.util_tf.FocusedTorsions'] = NN.util_tf.FocusedTorsions
            sys.modules['NN.representation_layers'] =  NN.representation_layers
            sys.modules['NN.representation_layers.SingleComponent_map'] = NN.representation_layers.SingleComponent_map
            self.model = self.model_class.load_model(self.name_save_model, VERSION='OLD')

    def set_training_validation_split_(self, inds_rand=None):
        if self.inds_rand is None and inds_rand is None:
            inds_rand = None ; a=0
            while inds_rand is None:
                a+=1
                print('inds_rand attempt:',a)
                inds_rand = find_split_indices_(self.u, split_where=self.n_training, tol=0.0001)
            self.inds_rand = np.array(inds_rand)

        elif self.inds_rand is None and inds_rand is not None:
            self.inds_rand = np.array(inds_rand)
            print('inds_rand, provided, were used')

        else:
            print('inds_rand, imported earlier, were used')

        self.r_training = np.array(self.r[self.inds_rand][:self.n_training])
        self.r_validation = np.array(self.r[self.inds_rand][self.n_training:])
    
        self.u_training = np.array(self.u[self.inds_rand][:self.n_training])
        self.u_validation = np.array(self.u[self.inds_rand][self.n_training:])

    def check_PES_matching_dataset_(self, m=1000):
        print('checking that PES matches the sampled dataset:')
        errT = self.u_(self.r_training[:m]) - self.u_training[:m]
        errV = self.u_(self.r_validation[:m]) - self.u_validation[:m]
        print('errT:',errT.mean(),errT.min(),errT.max())
        print('errV:',errV.mean(),errV.min(),errV.max())

    ## ## ## ##

    def load_energies_during_training_(self, index_of_state=0):
        n_states = 1
        i = self.n_estimates-1
        name = self.name_save_BAR_inputs+'_BAR_input_'+str(i)+'_state'+str(index_of_state)+'_'+'_V'
        x = load_pickle_(name)
        if type(x) is list: _this_ = lambda x : x[0] # ...
        else: _this_ = lambda x : x
        x = _this_(x)
        m = x.shape[-1]//2
        self.MD_energies_T = np.zeros([n_states, self.n_estimates, m])
        self.MD_energies_V = np.zeros([n_states, self.n_estimates, m])
        self.BG_energies = np.zeros([n_states, self.n_estimates, m])

        for i in range(self.n_estimates):
            try:
                k = 0
                name = self.name_save_BAR_inputs+'_BAR_input_'+str(i)+'_state'+str(index_of_state)+'_'+'_V'
                x = _this_(load_pickle_(name)) ; m = x.shape[-1]//2
                self.BG_energies[k,i] = x[0,m:]
                self.MD_energies_V[k,i] = x[0,:m]
            except: print('V estimate'+str(i)+'skipped')

        for i in range(self.n_estimates):
            try:
                k = 0
                name = self.name_save_BAR_inputs+'_BAR_input_'+str(i)+'_state'+str(index_of_state)+'_'+'_T'
                x = _this_(load_pickle_(name)) ; m = x.shape[-1]//2
                self.MD_energies_T[k,i] = x[0,:m]
            except: print('T estimate'+str(i)+'skipped', end='\r')

    def plot_energies_during_training_(self, dpi=300, n_bins=80, _from=0, _range=None):
        state = 0
        density = False
        if _range is None:
            a = self.MD_energies_V.min()
            b = self.MD_energies_V.max()
            #a = #min([self.MD_energies_T.min(),self.MD_energies_V.min()])
            #b = #max([self.MD_energies_T.max(),self.MD_energies_V.max()])
            b += (b-a)
        else:
            a,b = _range
        axis = np.linspace(a,b,n_bins+1)
        axis = axis[1:] - 0.5*(axis[1]-axis[0])

        colors = plt.cm.jet(np.linspace(0,1,self.n_estimates))
        import matplotlib as mpl
        custom_cmap = mpl.colors.ListedColormap(colors)
        
        size = 12 # labels

        fig = plt.figure(dpi=dpi*0.5, facecolor="white")
        cb = mpl.colorbar.ColorbarBase(fig.add_axes([0.05, 0.10, 0.9*0.5, 0.03]),
                                    orientation='horizontal', 
                                    cmap='jet',
                                    norm=mpl.colors.Normalize(0,self.evaluation_grid[-1]+1),  # vmax and vmin
                                    #extend='both',
                                    label='batches',
                                    ticks=np.linspace(0,self.evaluation_grid[-1]+1,3)) # (self.evaluation_grid[-1]+1)//1000
        plt.show()

        fig = plt.figure(figsize=(7,2),dpi=dpi)
        for i in range(_from,self.n_estimates):
            plt.plot(axis,
                    np.histogram(self.BG_energies[state,i],range=[a,b],bins=n_bins,density=density)[0],
                    color=custom_cmap.colors[i])
            plt.plot(axis,
                    np.histogram(self.MD_energies_T[state,i],range=[a,b],bins=n_bins,density=density)[0],
                    color='black',linewidth=1)
            plt.plot(axis,
                    np.histogram(self.MD_energies_V[state,i],range=[a,b],bins=n_bins,density=density)[0],
                    color='black',linewidth=1)
            
        plt.title(self.name)
        plt.xticks([a,b,self.u_mean,self.BG_energies[state,-1].mean()])
        plt.xlim(a,b)
        plt.xlabel('potential energy [$\mathregular{k_B}$T]', size=size)
        plt.ylabel('population', labelpad=0, size=size)
        plt.xticks(rotation=45)
        #plt.show()

    def solve_BAR_using_pymbar_(self, rerun=False, index_of_state=0, key='',
                                # can check other methods
                                n_bootstraps = 0, # default in MBAR class https://pymbar.readthedocs.io/en/latest/mbar.html
                                uncertainty_method = None, # can set 'bootstrap' if n_bootstraps not None
                                save_output = True,
                                method_for_selective_evalaution_ = None,
                               ):
        # from energies that were saved during training
        n_states = 1
        try:
            estimates_BAR = load_pickle_(self.name_save_BAR_inputs+'_BAR_output'+key)
            assert self.n_estimates == estimates_BAR.shape[1]
            self.estimates_BAR = estimates_BAR
            print('found saved BAR result')
            self.set_final_result_()
            if rerun:
                print('rerun = True')
                assert 'go to'=='except'
            else: pass

        except:
            self.estimates_BAR = np.zeros([4, self.n_estimates, n_states])
            idx = 7 # 1 ; 7 closer to minimiser of than 1, faster to run mbar.
            offset = self.estimates[0,:,idx][np.where(self.estimates[0,:,idx]>-1e19)[0]].mean()

            if method_for_selective_evalaution_ is not None: 
                method_for_selective_evalaution_(obj = self,
                                                 index_of_state = index_of_state,
                                                 AVMD_V=self.estimates[0,:,1],
                                                 )
            else:
                for i in range(self.n_estimates):
                    clear_output(wait=True)
                    print(i)
    
                    name = self.name_save_BAR_inputs+'_BAR_input_'+str(i)+'_state'+str(index_of_state)+'_'+'_T'
                    try:
                        x = load_pickle_(name)
                        # change this when weights are used (x[1]), currently weights are ignored (data, model are unbiased)
                        if type(x) is list: x = x[0]
                        else: pass
                        m = x.shape[-1]//2
                        # offset = x[0,:m].mean()
                        mbar_res = MBAR(np.stack([x[0]-offset, x[1]],axis=0),
                                        np.array([m]*2),
                                        n_bootstraps=n_bootstraps,
                                        ).compute_free_energy_differences(
                                            uncertainty_method=uncertainty_method
                                            )
                        FE = mbar_res['Delta_f'][1,0] + offset
                        SE = mbar_res['dDelta_f'][1,0]
                    except:
                        FE = 1e20 ; SE = 1e20
                        print('BAR: T estimate'+str(i)+'skipped')
                    self.estimates_BAR[0,i,0] = FE
                    self.estimates_BAR[1,i,0] = SE
                    
                    name = self.name_save_BAR_inputs+'_BAR_input_'+str(i)+'_state'+str(index_of_state)+'_'+'_V'
                    try:
                        x = load_pickle_(name)
                        # change this when weights are used (x[1]), currently weights are ignored (data, model are unbiased)
                        if type(x) is list: x = x[0]
                        else: pass
                        m = x.shape[-1]//2
                        # offset = x[0,:m].mean()
                        mbar_res = MBAR(np.stack([x[0]-offset, x[1]],axis=0),
                                        np.array([m]*2),
                                        n_bootstraps=n_bootstraps,
                                        ).compute_free_energy_differences(
                                            uncertainty_method=uncertainty_method
                                            )
                        FE = mbar_res['Delta_f'][1,0] + offset
                        SE = mbar_res['dDelta_f'][1,0]
                    except:
                        FE = 1e20 ; SE = 1e20
                        print('BAR: V estimate'+str(i)+'skipped')
                    self.estimates_BAR[2,i,0] = FE
                    self.estimates_BAR[3,i,0] = SE

            self.estimates_BAR  = np.where(np.isnan(self.estimates_BAR),1e20,self.estimates_BAR)

            if save_output:
                save_pickle_(self.estimates_BAR, self.name_save_BAR_inputs+'_BAR_output'+key)
                print('saved BAR result')
            else:
                print('BAR results were recomputed but not saved')

            self.set_final_result_()

    def set_final_result_(self,):
        self.BAR_V_FEs_raw = np.array(self.estimates_BAR[2,:,0])
        self.BAR_V_SEs_raw = np.array(self.estimates_BAR[3,:,0])

        self.BAR_V_FEs = FE_of_model_curve_(self.estimates[0,:,1], self.BAR_V_FEs_raw)
        self.BAR_V_FE = self.BAR_V_FEs[-1]

        self.BAR_V_SDs = FE_of_model_curve_(self.estimates[0,:,1], (self.BAR_V_FEs_raw-self.BAR_V_FEs)**2)**0.5
        self.BAR_V_SD =  self.BAR_V_SDs[-1]

        self.BAR_V_SEs = FE_of_model_curve_(self.estimates[0,:,1]-self.BAR_V_SEs_raw, self.BAR_V_SEs_raw)
        self.BAR_V_SEs = np.max([self.BAR_V_SEs, self.BAR_V_SDs], axis=0)
        self.BAR_V_SE  = self.BAR_V_SEs[-1]

    '''
    def plot_result_(self, window=1, entropy_only=False, plot_red=True, n_mol=1, colors=['green', 'blue', 'm', 'red'], ax=None,
                     plot_raw_errors = True):
        if ax is not None: plot = ax
        else: plot = plt

        assert type(n_mol) is int and n_mol >= 1

        BAR_V = np.array(self.estimates_BAR[2,:,0])/n_mol
        BAR_V_SEs = np.array(self.estimates_BAR[3,:,0])/n_mol
        FEs = np.array(self.FEs)/n_mol

        if entropy_only:
            BAR_V = - (BAR_V - self.u_mean/n_mol)
            FEs = - (FEs - self.u_mean/n_mol)
        else: pass
        FE = FEs[-1]

        plot.plot(self.evaluation_grid, BAR_V, color=colors[0])
        plot.plot(self.evaluation_grid, self.estimates[0,:,7]/n_mol, color=colors[1], linewidth=0.3, linestyle='--')
        if plot_raw_errors:
            plot.fill_between(self.evaluation_grid, BAR_V-BAR_V_SEs, BAR_V+BAR_V_SEs, alpha=0.4, color=colors[0])
        else: pass
        
        if plot_red:
            #plot.plot([self.evaluation_grid[0],self.evaluation_grid[-1]], [FEs[-1]]*2, color='red')
            plot.plot(self.evaluation_grid, FEs, color=colors[2])
            plot.plot(self.evaluation_grid, FEs-self.SDs/n_mol, color=colors[2], linestyle='dotted', zorder=9)
            plot.plot(self.evaluation_grid, FEs+self.SDs/n_mol, color=colors[2], linestyle='dotted', zorder=9)
            plot.plot(self.evaluation_grid, self.BAR_V_FEs/n_mol-self.BAR_V_SEs/n_mol, color=colors[3], linestyle='dotted', linewidth=2, zorder=10)
            plot.plot(self.evaluation_grid, self.BAR_V_FEs/n_mol+self.BAR_V_SEs/n_mol, color=colors[3], linestyle='dotted', linewidth=2, zorder=10)
            plot.plot(self.evaluation_grid, self.BAR_V_FEs/n_mol,                      color=colors[3], linewidth=2)
        else: pass

        if ax is not None: plot.set_ylim(FE-window, FE+window)
        else:              plot.ylim(FE-window, FE+window)
        print(FE,'+/-', self.SDs[-1]/n_mol, 'final:', self.BAR_V_FE/n_mol, '+/-', self.BAR_V_SE/n_mol)
    '''
    def plot_result_(self, window=1, entropy_only=False,
                     plot_red=True, n_mol=1, colors=['green', 'blue', 'm', 'red'], ax=None,
                     plot_raw_errors = True):
        if ax is not None: plot = ax
        else: plot = plt

        assert type(n_mol) is int and n_mol >= 1

        BAR_V_FEs_raw = np.array(self.BAR_V_FEs_raw)/n_mol
        BAR_V_SEs_raw = np.array(self.BAR_V_SEs_raw)/n_mol
        BAR_V_FEs_averaged = np.array(self.BAR_V_FEs)/n_mol ; FE = float(self.BAR_V_FE)/n_mol
        BAR_V_SEs_averaged = np.array(self.BAR_V_SEs)/n_mol

        plot.plot(self.evaluation_grid, BAR_V_FEs_raw, color=colors[0])
        plot.plot(self.evaluation_grid, self.estimates[0,:,7]/n_mol, color=colors[1], linewidth=0.3, linestyle='--')

        if plot_raw_errors:
            plot.fill_between(self.evaluation_grid, BAR_V_FEs_raw-BAR_V_SEs_raw, BAR_V_FEs_raw+BAR_V_SEs_raw, alpha=0.4, color=colors[0])
        else: pass
        
        if plot_red:
            #plot.plot([self.evaluation_grid[0],self.evaluation_grid[-1]], [FEs[-1]]*2, color='red')
            plot.plot(self.evaluation_grid, self.FEs/n_mol, color=colors[2])
            plot.plot(self.evaluation_grid, self.FEs/n_mol-self.SDs/n_mol, color=colors[2], linestyle='dotted', zorder=9)
            plot.plot(self.evaluation_grid, self.FEs/n_mol+self.SDs/n_mol, color=colors[2], linestyle='dotted', zorder=9)
            plot.plot(self.evaluation_grid, BAR_V_FEs_averaged - BAR_V_SEs_averaged, color=colors[3], linestyle='dotted', linewidth=2, zorder=10)
            plot.plot(self.evaluation_grid, BAR_V_FEs_averaged + BAR_V_SEs_averaged, color=colors[3], linestyle='dotted', linewidth=2, zorder=10)
            plot.plot(self.evaluation_grid, BAR_V_FEs_averaged,                      color=colors[3], linewidth=2)
        else: pass

        if ax is not None: plot.set_ylim(FE-window, FE+window)
        else:              plot.ylim(FE-window, FE+window)

        print(f'rough grid search estimate: {self.FEs[-1]/n_mol}  +/- standard deviation = {self.SDs[-1]/n_mol} ')
        print(f'     pymbar final estimate: {self.BAR_V_FE/n_mol} +/- standard error     = {self.BAR_V_SE/n_mol}')

class NN_interface_sc(NN_interface_helper):
    '''
    ONE dataset : a metastable state

    ONE dataset is split into training and validation data.
    A single instance of ic_map_class initialised on the dataset.
    '''
    def __init__(self,
                 name : str,
                 path_dataset : str,
                 fraction_training : float = 0.8, # 0 << x < 1
                 training : bool = True,
                 ic_map_class = SingleComponent_map,
                 ):
        super().__init__()
        self.name = name + '_SC_' # ...
        self.path_dataset = path_dataset 
        self.fraction_training = fraction_training
        self.training = training
        self.ic_map_class = ic_map_class

        if not self.training:
            if type(self.path_dataset) is str:
                self.u = load_pickle_(self.path_dataset)['MD dataset']['u']
                self.u_mean = self.u.mean()
            else:
                self.u_mean = np.array(self.path_dataset)
        else:
            self.import_MD_dataset_()
 
    def import_MD_dataset_(self,):
            self.simulation_data = load_pickle_(self.path_dataset)
            self.sc = SingleComponent(**self.simulation_data['args_initialise_object'])
            self.sc.initialise_system_(**self.simulation_data['args_initialise_system'])
            self.sc.initialise_simulation_(**self.simulation_data['args_initialise_simulation'])
            assert self.sc.n_DOF in [3*(self.sc.N - 1), 3*self.sc.N]

            self.T = self.sc.T # Kelvin
            self.u_ = self.sc.u_

            self.r = self.simulation_data['MD dataset']['xyz'].astype(np.float32) #- self.simulation_data['MD dataset']['COMs']).astype(np.float32)
            b = self.simulation_data['MD dataset']['b'].astype(np.float32)
            self.b0 = b[-1]
            assert np.abs(b[0] - self.b0).max() < 0.0000001
            self.Ts = self.simulation_data['MD dataset']['T'] # temperatures : not use anywhere later
            self.u = self.simulation_data['MD dataset']['u']  # potential energies : need for FE estimates (except if using mBAR)
            self.u_mean = self.u.mean()

            assert len(self.r) == len(self.u)
            self.n_training = int(self.u.shape[0]*self.fraction_training)

            del self.simulation_data

    def truncate_data_(self, m=None):
        '''
        If needed: 
            this function makes the dataset (self.r) smaller in size
                ran as soon as NN_interface_sc initialised (at least before set_ic_map_step2)
        
        Why:
        Memory (RAM) in larger supercells overflows such that tensorflow does not work:
            - cause: tensorflow (tf) loading whole dataset into GPU memory
                - cause: any one function in any part of ic_map, during dataset initialisation, not numpy but tf
            - solutions (good): 
                change all functions involved with dataset initialisation to numpy (i.e., have a numpy version of ic_map._forward_)
                change to torch where small data batches attached to gpu at any time
                use a computer with more (V)RAM
            - solutions (bad):
                use this function to be able to train something large, but at the cost of error bars being high.
                    - cause: early overfitting = validation loss not properly minimised
                        - cause: q (PGM) not symmetry aware --> needs plenty data in a larger supercell --> memory.
        '''
        inds = np.random.choice(len(self.u),len(self.u),replace=False)
        self.r = self.r[inds][:m]
        self.u = self.u[inds][:m]
        assert len(self.r) == len(self.u)
        self.n_training = int(self.u.shape[0]*self.fraction_training)
        
        self.Ts = self.Ts[inds][:m]

    def set_ic_map_step1(self, ind_root_atom=11, option=None):
        '''
        self.sc.mol is available for this reason; to check the indices of atoms in the molecule
        once (ind_root_atom, option) pair is chosen, keep a fixed note of this for this molecule
        '''
        assert self.training
        self.ind_root_atom = ind_root_atom
        self.option = option

        PDBmol = './'+str(self.sc._single_mol_pdb_file_)
        self.ic_map = self.ic_map_class(PDB_single_mol = PDBmol)
        self.ic_map.set_ABCD_(ind_root_atom = self.ind_root_atom, option = self.option)

    def set_ic_map_step2(self,
                         inds_rand=None,
                         check_PES = True,
                         ):
        self.r = self.ic_map.remove_COM_from_data_(self.r)
        self.set_training_validation_split_(inds_rand=inds_rand)
        if check_PES: self.check_PES_matching_dataset_()
        else: pass

    def set_ic_map_step3(self,
                         n_mol_unitcell : int = 1, # !! important in this new version
                         COM_remover = WhitenFlow,
                        ):
        self.n_mol_unitcell = n_mol_unitcell

        self.ic_map.initalise_(r_dataset = self.r, b0 = self.b0,
                               n_mol_unitcell = self.n_mol_unitcell,
                               COM_remover = COM_remover,
                               )
        m = 1000
        r = np.array(self.r_validation[:m])
        x, ladJf = self.ic_map.forward_(r)
        _r, ladJi = self.ic_map.inverse_(x)
        print('ic_map inversion errors on a small random batch:')
        print('positons:', np.abs(_r-r).max())
        print('volume:', np.abs(ladJf + ladJi).max())

        try:
            value = np.abs(np.array(x)).max()
            if value > 1.000001: print('!!', value)
            else: print(value)
        except:
            values = []
            for _x in x:
                try: values.append(np.abs(np.array(_x)).max())
                except: pass
            value = max(values) # 1.0000004 , 1.0000006
            if value > 1.000001: print('!!', value)
            else: print(value)

    # set_model not here and the rest not here.

class NN_interface_sc_multimap(NN_interface_helper):
    def __init__(self,
                 name : str,
                 paths_datasets : list,
                 fraction_training : float = 0.8, # 0 << x < 1
                 running_in_notebook : bool = False,
                 training : bool = True,
                 model_class = PGMcrys_v1,
                 ic_map_class = SingleComponent_map,
                 ):
        super().__init__()
        self.name = name + '_SC_' # ...
        assert type(paths_datasets) == list
        self.paths_datasets = paths_datasets
        self.fraction_training = fraction_training
        self.running_in_notebook = running_in_notebook 
        self.training = training
        self.model_class = model_class
        self.ic_map_class = ic_map_class
        ##
        self.n_crystals = len(self.paths_datasets)

        self.nns = [NN_interface_sc(
                    name = name,
                    path_dataset = self.paths_datasets[i],
                    fraction_training = self.fraction_training,
                    training = self.training, 
                    ic_map_class = self.ic_map_class,
                    ) for i in range(self.n_crystals)]

    def save_inds_rand_(self,):
        for k in range(self.n_crystals):
            self.nns[k].save_inds_rand_(key='_crystal_index='+str(k))
        #[nn.save_inds_rand_() for nn in self.nns]

    def load_inds_rand_(self,):
        for k in range(self.n_crystals):
            self.nns[k].load_inds_rand_(key='_crystal_index='+str(k))
        #[nn.load_inds_rand_() for nn in self.nns]

    def set_ic_map_step1(self, ind_root_atom=11, option=None):
        self.ind_root_atom = ind_root_atom
        self.option = option
        [nn.set_ic_map_step1(ind_root_atom=self.ind_root_atom, option=self.option) for nn in self.nns];

    def set_ic_map_step2(self, check_PES=True):
        [nn.set_ic_map_step2(inds_rand=None, check_PES=check_PES) for nn in self.nns];

    def set_ic_map_step3(self,
                         n_mol_unitcells : list = None, # !! important in this new version
                         COM_remover = WhitenFlow,
                        ):
        if n_mol_unitcells is None:
            print(
            f'''
            !! for each of the {self.n_crystals} supercell datasets provided,
            it is important to provide here a list (n_mol_unitcells) describing 
            the number of molecules in each underlying unitcell, respectively.
            '''
            ) 
            n_mol_unitcells = [None]*self.n_crystals
        else:
            if type(n_mol_unitcells) is int and self.n_crystals == 1: n_mol_unitcells = [n_mol_unitcells]
            else: assert type(n_mol_unitcells) is list
        
        self.n_mol_unitcells = n_mol_unitcells 

        for i in range(self.n_crystals):
            self.nns[i].set_ic_map_step3(n_mol_unitcell = self.n_mol_unitcells[i],
                                         COM_remover = COM_remover,
                                        )
        # TODO: check that the later merging reaches back into each nns[i].ic_map

    def set_model(self,
                  learning_rate = 0.001,
                  evaluation_batch_size = 5000,
                  ##
                  n_layers = 4,
                  DIM_connection = 10,
                  n_att_heads = 4,
                  initialise = True, # for debugging in eager mode
                  test_inverse = True,
                  ):
        self.model = self.model_class(
                                    ic_maps = [nn.ic_map for nn in self.nns],
                                    n_layers = n_layers,
                                    optimiser_LR_decay = [learning_rate, 0.0],
                                    DIM_connection = DIM_connection,
                                    n_att_heads = n_att_heads,
                                    initialise = initialise,
                                    )
        self.model.print_model_size()
        self.evaluation_batch_size = evaluation_batch_size
        if test_inverse:
            for crystal_index in range(self.n_crystals):
                r = np2tf_(self.nns[crystal_index].r_validation[:self.evaluation_batch_size])
                inv_test_res0 = self.model.test_inverse_(r, crystal_index=crystal_index, graph =True)
                print('inv_test_res0',inv_test_res0) # TODO make interpretable
        else: pass

    def set_trainer(self, n_batches_between_evaluations = 50):
        self.trainer = TRAINER(model = self.model,
                               max_training_batches = 50000,
                               n_batches_between_evaluations = n_batches_between_evaluations,
                               running_in_notebook = self.running_in_notebook,
                               )
        
    def u_(self, r, k):
        return self.nns[k].u_(r)

    def train(self,
              n_batches = 2000,
              save_BAR = True,
              save_mBAR = False,

              save_misc = True,
              verbose = True,
              verbose_divided_by_n_mol = True,
              f_halfwindow_visualisation = 1.0, # int or list
              test_inverse = False,
              evaluate_on_training_data = False,
              #
              evaluate_main = True,
              training_batch_size = 1000, # 1000 was always used
              ):
        
        if save_BAR: name_save_BAR_inputs = str(self.name_save_BAR_inputs)
        else: name_save_BAR_inputs = None

        if save_mBAR: name_save_mBAR_inputs = str(self.name_save_mBAR_inputs)
        else: name_save_mBAR_inputs = None

        self.trainer.train(
                        n_batches = n_batches,

                        # needed, and always available: xyz coordinates of training and validation sets
                        list_r_training   = [nn.r_training for nn in self.nns],
                        list_r_validation = [nn.r_validation for nn in self.nns],

                        # needed, and always available: potential energies of training and validation sets
                        list_u_training   = [nn.u_training for nn in self.nns],
                        list_u_validation = [nn.u_validation for nn in self.nns],

                        # not needed if not relevant: [weights associated with training and validation sets]
                        list_w_training = None,
                        list_w_validation = None,

                        # if the FF is cheap can use, BAR_V estimates during training will be saved.
                        list_potential_energy_functions = [self.nns[k].u_ for k in range(self.n_crystals)],

                        # evalaution cost vs statistical significance of gradient of the loss (1000 is best)
                        training_batch_size = training_batch_size,
                        # evalaution cost vs variance of FE estimates (affects standard error from pymbar if save_BAR True)
                        evaluation_batch_size = self.evaluation_batch_size,

                        # always True, unless just quickly checking if the training works at all
                        evaluate_main = evaluate_main,
                        # pymbar will run later, but need to save those inputs during training
                        name_save_BAR_inputs = name_save_BAR_inputs,
                        name_save_mBAR_inputs = name_save_mBAR_inputs,

                        # statistical significance vs model quality. Dont need to suffle if the learned map is ideal, but it is not ideal.
                        shuffle = True,
                        # verbose: when running in jupyter notebook y_axis width of running plot that are shown during training
                        f_halfwindow_visualisation = f_halfwindow_visualisation, # f_latt / kT if verbose_divided_by_n_mol, else f_crys / kT
                        verbose = verbose, # whether to plot matplotlib plots during training
                        verbose_divided_by_n_mol = verbose_divided_by_n_mol, # plot and report lattice FEs (True) or crystal FEs (False) [lattice_FE = crystal_FE/n_mol]
                        
                        # evalaution cost:
                        evaluate_on_training_data = evaluate_on_training_data, # not needed.
                        test_inverse = test_inverse, # not needed if model ok, but useful to check sometimes. 
                            # Inversion errors must not be too biased (i.e., must include zero error). Some error is unavoidable due to model depth, and overfitting.
                            # BAR rewighting is quite robust even in the presence of large inversion errors on some sample, but only if the error distribution includes zero.
                            # If setting test_inverse True, this information is collected during training, but adds a lot of overhead cost during training.
                            # TODO: statistics collected about the quality of the inverses (inv_test) may be adjusted from what they are currently (mean, max).
                        # Disk space needed for this:
                        save_generated_configurations_anyway = False, 
                        # not tested yet but here in case the FF is expensive (self.u_ function missing, not defined, on purpose)
                        # when FF expensive, yet model is always cheap to sample, so sample model on each batch and save this along with other BAR inputs that can be used later
                        # use validation error to choose best model, and on those xyz samples from the model compute BAR (after evalauting the FF on the selected files)
                        # dont know when the model minimises validation error, so saving each evaluation batch and choose the most promising pymbar inputs afterwards.
                    )
        print('training time so far:', np.round(self.trainer.training_time, 2), 'minutes')

        if save_misc:
            self.save_misc_()
            print('misc training outputs were saved')
        else: pass
        if test_inverse:
            self.save_inv_test_results_()
            print('inv_test results were saved')
        else: pass

    def load_misc_(self,):
        self._FEs, self._SDs, self.estimates, self.evaluation_grid, self.AVMD_f_T_all, self.training_time = load_pickle_(self.name_save_misc)
        self.n_estimates = self.evaluation_grid.shape[0]
        for k in range(self.n_crystals):
            self.nns[k]._FEs = self._FEs[k]
            self.nns[k]._SDs = self._SDs[k]
            self.nns[k].estimates = self.estimates[k:k+1]
            self.nns[k].evaluation_grid = self.evaluation_grid
            self.nns[k].AVMD_f_T_all = self.AVMD_f_T_all[:,k:k+1]
            self.nns[k].n_estimates = self.n_estimates
        for nn in self.nns:
            nn.FEs = FE_of_model_curve_(nn.estimates[0,:,1], nn.estimates[0,:,7])
            nn.SDs = FE_of_model_curve_(nn.estimates[0,:,1], (nn.estimates[0,:,7]-nn.FEs)**2)**0.5

    def load_energies_during_training_(self):
        [self.nns[k].load_energies_during_training_(index_of_state=k) for k in range(self.n_crystals)];

    def plot_energies_during_training_(self, crystal_index=0, **kwargs):
        self.nns[crystal_index].plot_energies_during_training_(**kwargs)

    def solve_BAR_using_pymbar_(self, rerun=False):
        for k in range(self.n_crystals):
            self.nns[k].solve_BAR_using_pymbar_(rerun=rerun, index_of_state=k, key='_crystal_index='+str(k))

    def plot_result_(self, crystal_index=0, **kwargs):
        self.nns[crystal_index].plot_result_(**kwargs)
        


    def solve_mBAR_using_pymbar_(self, rerun=False,
                                 n_bootstraps = 0, # default in MBAR class https://pymbar.readthedocs.io/en/latest/mbar.html
                                 uncertainty_method = None, # can set 'bootstrap' if n_bootstraps not None
                                 save_output = True):
        # from energies that were saved during training
        
        n_states = int(self.n_crystals)
        try:
            self.estimates_mBAR = load_pickle_(self.name_save_mBAR_inputs+'_mBAR_output')
            print('found saved mBAR result')
            if rerun:
                print('rerun=True, so rerunning BAR calculation again.')
                assert 'go to'=='except'
            else: pass

        except:
            self.estimates_mBAR = np.zeros([4, self.n_estimates, n_states, n_states])
            HIGH = np.eye(n_states) + 1e20

            for i in range(self.n_estimates):
                clear_output(wait=True)
                print(i)

                name = self.name_save_mBAR_inputs+'_mBAR_input_'+str(i)+'_T'
                try:
                    x = load_pickle_(name) ; m = x.shape[-1]
                    mbar_res = MBAR(x.reshape([n_states, n_states*m]),
                                    np.array([m]*n_states),
                                    n_bootstraps=n_bootstraps,
                                    ).compute_free_energy_differences(uncertainty_method=uncertainty_method)
                    FE = mbar_res['Delta_f']
                    SE = mbar_res['dDelta_f']
                except:
                    FE = HIGH ; SE = HIGH
                    print('BAR: T estimate'+str(i)+'skipped')
                self.estimates_mBAR[0,i] = FE
                self.estimates_mBAR[1,i] = SE
                
                name = self.name_save_mBAR_inputs+'_mBAR_input_'+str(i)+'_V'
                try:
                    x = load_pickle_(name) ; m = x.shape[-1]
                    mbar_res = MBAR(x.reshape([n_states, n_states*m]),
                                    np.array([m]*n_states),
                                    n_bootstraps=n_bootstraps,
                                    ).compute_free_energy_differences(uncertainty_method=uncertainty_method)
                    FE = mbar_res['Delta_f']
                    SE = mbar_res['dDelta_f']
                except:
                    FE = HIGH ; SE = HIGH
                    print('BAR: V estimate'+str(i)+'skipped')
                self.estimates_mBAR[2,i] = FE
                self.estimates_mBAR[3,i] = SE

            self.estimates_mBAR  = np.where(np.isnan(self.estimates_mBAR),1e20,self.estimates_mBAR)

            if save_output:
                save_pickle_(self.estimates_mBAR, self.name_save_mBAR_inputs+'_mBAR_output')
                print('saved mBAR result')
            else: 
                print('mBAR results were recomputed but not saved')
            #self.set_final_result_()

    def sample_model_(self, m, crystal_index=0):
        n_draws = m//self.evaluation_batch_size
        return np.concatenate([self.model.sample_model(self.evaluation_batch_size, crystal_index=crystal_index)[0] for i in range(n_draws)],axis=0)

    def save_samples_(self, m:int=20000):
        for crystal_index in range(self.n_crystals):
            save_pickle_(self.sample_model_(m, crystal_index=crystal_index), self.name_save_samples+'_crystal_index='+str(crystal_index))

    def load_samples_(self, crystal_index=None):
        if crystal_index is None:
            self.samples_from_model = []
            for crystal_index in range(self.n_crystals):
                self.samples_from_model.append(load_pickle_(self.name_save_samples+'_crystal_index='+str(crystal_index)))
        else:
            self.samples_from_model = load_pickle_(self.name_save_samples+'_crystal_index='+str(crystal_index))

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

class NN_interface_sc_multimap_selective_evaluation(NN_interface_sc_multimap):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def train(self,
              n_batches = 2000,
              verbose = True,
              verbose_divided_by_n_mol = True,
              f_halfwindow_visualisation = 1.0, # int or list
              test_inverse = False,
              #
              training_batch_size = 1000, # 1000 was always used
              ):
        save_BAR = True
        save_mBAR = False
        save_misc = True
        
        if save_BAR: name_save_BAR_inputs = str(self.name_save_BAR_inputs)
        else: name_save_BAR_inputs = None

        if save_mBAR: name_save_mBAR_inputs = str(self.name_save_mBAR_inputs)
        else: name_save_mBAR_inputs = None

        self.trainer.train(
                        n_batches = n_batches,
                        list_r_training   = [nn.r_training for nn in self.nns],
                        list_r_validation = [nn.r_validation for nn in self.nns],
                        list_u_training   = [nn.u_training for nn in self.nns],
                        list_u_validation = [nn.u_validation for nn in self.nns],
                        list_w_training = None,
                        list_w_validation = None,
                        list_potential_energy_functions = [None for k in range(self.n_crystals)],
                        training_batch_size = training_batch_size,
                        evaluation_batch_size = self.evaluation_batch_size,
                        evaluate_main = True,
                        name_save_BAR_inputs = name_save_BAR_inputs,
                        name_save_mBAR_inputs = name_save_mBAR_inputs,
                        shuffle = True,
                        f_halfwindow_visualisation = f_halfwindow_visualisation,
                        verbose = verbose,
                        verbose_divided_by_n_mol = verbose_divided_by_n_mol,
                        evaluate_on_training_data = False,
                        test_inverse = test_inverse,
                        save_generated_configurations_anyway = False, 
                    )
        print('training time so far:', np.round(self.trainer.training_time, 2), 'minutes')

        if save_misc:
            self.save_misc_()
            print('misc training outputs were saved')
        else: pass
        if test_inverse:
            self.save_inv_test_results_()
            print('inv_test results were saved')
        else: pass

    def solve_BAR_using_pymbar_(self, rerun=False, n_selective_evalautions = 5):
        self.n_selective_evalautions = n_selective_evalautions 
        for k in range(self.n_crystals):
            _nn = self.nns[k]
            _nn.solve_BAR_using_pymbar_(rerun = True,
                                        index_of_state = k, 
                                        key = '_crystal_index=' + str(k), 
                                        method_for_selective_evalaution_ = self.method_for_selective_evalaution_v1_)
            self.reset_final_result_(_nn)

    def reset_final_result_(self, obj):

        AVMD_V = np.array(obj.estimates[0,:,1])

        BAR_V_FEs_raw = np.array(obj.estimates_BAR[2,:,0])
        BAR_V_SEs_raw = np.array(obj.estimates_BAR[3,:,0])

        inds_solved = np.array(self.inds_solved_current)

        BAR_V_FEs = FE_of_model_curve_(AVMD_V[inds_solved], BAR_V_FEs_raw[inds_solved])
        obj.BAR_V_FE = BAR_V_FEs[-1]

        BAR_V_SDs = FE_of_model_curve_(AVMD_V[inds_solved], (BAR_V_FEs_raw[inds_solved] - BAR_V_FEs)**2)**0.5
        obj.BAR_V_SD =  BAR_V_SDs[-1]

        BAR_V_SEs = FE_of_model_curve_(AVMD_V[inds_solved] - BAR_V_SEs_raw[inds_solved], BAR_V_SEs_raw[inds_solved])
        BAR_V_SEs = np.max([BAR_V_SEs, BAR_V_SDs], axis=0)
        obj.BAR_V_SE  = BAR_V_SEs[-1]

        nan_for_plotting  = np.ones_like(BAR_V_FEs_raw) * np.nan
        obj.BAR_V_FEs_raw = np.array(nan_for_plotting)
        obj.BAR_V_SEs_raw = np.array(nan_for_plotting)
        obj.BAR_V_FEs     = np.array(nan_for_plotting)
        obj.BAR_V_SDs     = np.array(nan_for_plotting)
        obj.BAR_V_SEs     = np.array(nan_for_plotting)

        obj.BAR_V_FEs_raw[inds_solved] = np.array(BAR_V_FEs_raw[inds_solved])
        obj.BAR_V_SEs_raw[inds_solved] = np.array(BAR_V_SEs_raw[inds_solved])
        obj.BAR_V_FEs[inds_solved]     = np.array(BAR_V_FEs)
        obj.BAR_V_SDs[inds_solved]     = np.array(BAR_V_SDs)
        obj.BAR_V_SEs[inds_solved]     = np.array(BAR_V_SEs)

    def method_for_selective_evalaution_v1_(self,
                                            obj,
                                            index_of_state,
                                            AVMD_V,
                                            ):
        validation_loss_curve = np.array( - AVMD_V )
        # chosing parts that come from model when it had the lowest validiation error
        inds_best_validation_batches = np.argsort(validation_loss_curve)[:self.n_selective_evalautions]

        offset = AVMD_V[np.where(AVMD_V>-1e19)[0]].mean()

        obj.estimates_BAR = np.ma.array(obj.estimates_BAR, mask=True)
        
        print('')
        print(f'two-state BAR evaluation in macrostate {index_of_state}:')
        for i in inds_best_validation_batches:
            path_and_name = obj.name_save_BAR_inputs+'_BAR_input_'+str(i)+'_state'+str(index_of_state) + '_'
            name = path_and_name + '_r_&_ln(q(r))'
            name_complete = path_and_name + '_V'
            name_solved = path_and_name + '_r_&_ln(q(r))_solved'

            try:
                FE, SE = load_pickle_(name_solved) #; print('found ')
            except:
                print(f'evalaution batch {i}: evaluating potential energies on model samples. The estimate will be saved.')
                r_BG, ln_q_BG, u_V, ln_q_V, w_V = load_pickle_(name)
                # TODO (in general for biased data): add way to rewight pymbar solving 2state BAR when weights (w_V) not None

                assert len(r_BG) == len(ln_q_BG)
                assert len(u_V)  == len(ln_q_V)
                n_V  = len(u_V)
                n_BG = len(ln_q_BG) 

                # the expensive step:
                u_BG = self.nns[index_of_state].u_(r_BG) # = obj.u_(r_BG)

                Q = np.stack([np.concatenate([u_V, u_BG])[...,0] - offset,
                            - np.concatenate([ln_q_V, ln_q_BG])[...,0], # important: negative sign for positive energy
                            ], axis=0)
                Ns = np.array([n_V, n_BG])
                mbar_res = MBAR(Q, Ns).compute_free_energy_differences()

                FE = mbar_res['Delta_f'][1,0] + offset
                SE = mbar_res['dDelta_f'][1,0]

                save_pickle_([FE, SE], name=name_solved, verbose=False)
                save_pickle_(np.stack([np.concatenate([u_V,       u_BG])[...,0],
                                     - np.concatenate([ln_q_V, ln_q_BG])[...,0]], axis=0), 
                             name=name_complete, verbose=False)
            
            # obj.estimates_BAR[0,i,0] = 0.0 # nothing here, not evalauting on training data
            # obj.estimates_BAR[1,i,0] = 0.0 # nothing here, not evalauting on training data
            obj.estimates_BAR[2,i,0] = FE    # FE on validation data
            obj.estimates_BAR[3,i,0] = SE    # SE on validation data
            # this turns masks to False for elements that are not zeros (batches that were done above)

        self.inds_solved_current = np.where(obj.estimates_BAR[2,:,0].mask == False)[0]

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
