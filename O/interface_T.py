from .interface import *
from .NN.pgm import *

''' f(N,V,T) for T = T_low, ..., T_high at fixed N and V(box), using M_{/mol, multimap} with only one ic_map (for temperatures rather than Forms).
    + this approach can save cost by ~ x2.5 times when all temperature states are fairly similar (same Form)

    ! despite only one ic_map in the model there are still retracing warnings (model compiled seperately for each Temperature parameter)
        - TODO: fix this cost issue

    ! temperatures are discrete states:
        Functinaloty for allowing continuous interpolation in T not yet included for two reasons:
        - to learn a smooth interpolation need to train on several temperatures (need enough MD data)
        - to compute BAR at any temperature need erogodic data (enough MD data also).
        All data available can be used for training and evalaution where descrete T is sufficent.
        
    ! thermal expansion effects neglected (box fixed):
        - TODO: add functionality to a single ic_map to toggle between discrete boxes (one per temperature)
'''

## ## 

class PGMcrys_v1_T(tf.keras.models.Model, model_helper_PGMcrys_v1, model_helper): #PGMcrys_v1):
    #''' !! : molecule should have >3 atoms (also true in ic_map) '''
    @staticmethod
    def load_model(path_and_name : str, VERSION='NEW'):
        return PGMcrys_v1_T._load_model_(path_and_name, PGMcrys_v1_T)

    def __init__(self,
                 ic_map : list,
                 n_temperatures : int,
                 n_layers : int = 4,
                 optimiser_LR_decay = [0.001,0.0],
                 DIM_connection = 10,
                 n_att_heads = 4,
                 initialise = True, # for debugging in eager mode
                 ):
        super().__init__()
        #PGMcrys_v1.__init__(self, None)
        import types
        for name, method in PGMcrys_v1.__dict__.items():
            if name in ['get_C2P_P2C_extensions_', '_forward_coupling_', '_inverse_coupling_']:
                assert callable(method)
                setattr(self, name, types.MethodType(method, self))

        self.init_args = {  'ic_map' : ic_map,
                            'n_layers' : n_layers,
                            'optimiser_LR_decay' : optimiser_LR_decay,
                            'DIM_connection' : DIM_connection,
                            'n_att_heads' : n_att_heads}
        ####

        if hasattr(ic_map, 'single_box_in_dataset'): assert ic_map.single_box_in_dataset == True
        else: ic_map.single_box_in_dataset = True

        self.ic_map = ic_map
        self.n_maps = n_temperatures
        self.n_mol = int(self.ic_map.n_mol)
        self.periodic_mask = np.array(self.ic_map.periodic_mask)

        if self.n_maps > 1:
            self.dim_crystal_encoding = 1

            #self.Tmin = self.Ts.min()
            #self.Tmax = self.Ts.max()
            #self.T_to_encoding_ = lambda T : 2.0 * (T - self.Tmin) / (self.Tmax - self.Tmin) - 1.0
            self.crystal_encodings = np2tf_(np.linspace(-1.,1.,self.n_maps))

            self.C2P_extension_shape = np2tf_(np.zeros([1,1]))
            self.P2C_extension_shape = np2tf_(np.zeros([self.n_mol,1]))
        else:
            self.dim_crystal_encoding = 0

            #self.T_to_encoding_ = lambda T : 0.0
            self.crystal_encodings = np2tf_(np.array([0]))

            self.C2P_extension_shape = np2tf_(np.zeros([1,0]))
            self.P2C_extension_shape = np2tf_(np.zeros([self.n_mol,0]))

        ####

        self.n_layers = n_layers
        self.optimiser_LR_decay = optimiser_LR_decay
        self.DIM_connection = DIM_connection
        self.n_att_heads = n_att_heads

        ##
        self.DIM_P2C_connection = self.DIM_connection  
        self.DIM_C2P_connection = self.n_mol*self.DIM_connection

        n_hidden_main = 2
        n_hidden_connection = 1
        hidden_activation = tf.nn.leaky_relu
        n_bins = 5
        print('self.n_att_heads:',self.n_att_heads)

        self.layers_P = [ POSITIONS_FLOW_LAYER(
                            n_mol = self.n_mol,
                            layer_index = i,
                            DIM_P2C_connection = self.DIM_P2C_connection,
                            DIM_C2P_connection = self.DIM_C2P_connection + self.dim_crystal_encoding,
                            name = 'POSITIONS_FLOW_LAYER',
                            n_hidden_main = n_hidden_main,
                            n_hidden_connection = n_hidden_connection,
                            hidden_activation = hidden_activation,
                            use_tfp = False,
                            n_bins = n_bins,
                            min_bin_width = 0.001,
                            knot_slope_range = [0.001, 50.0],
                            n_P2C = self.n_mol,
                        ) for i in range(self.n_layers)]

        self.layers_C = [ CONFORMER_FLOW_LAYER(
                            periodic_mask = self.periodic_mask,
                            layer_index = i,
                            DIM_P2C_connection = self.DIM_P2C_connection +  self.dim_crystal_encoding,
                            n_mol = self.n_mol,
                            DIM_C2P_connection = self.DIM_C2P_connection,
                            n_hidden_connection = n_hidden_connection,
                            half_layer_class = SPLINE_COUPLING_HALF_LAYER_AT,
                            kwargs_for_given_half_layer_class = {
                                        'flow_mask' : None,
                                        'n_mol' : self.n_mol,
                                        'n_heads' : self.n_att_heads, # 4
                                        'embedding_dim' : self.DIM_connection,
                                        'n_hidden_kqv' : [2,2,2],
                                        'hidden_activation' : hidden_activation,
                                        'one_hot_kqv' : [True]*3,
                                        'n_hidden_decode' : 1,
                                        #'new' : False,
                                        },
                            use_tfp = False,
                            n_bins = n_bins,
                            min_bin_width = 0.001,
                            knot_slope_range = [0.001, 50.0],
                            name = 'CONFORMER_FLOW_LAYER',
                        ) for i in range(self.n_layers)]

        ## p_{0}:
        self.ln_base_ = self.ic_map.ln_base_
        self.sample_base_ = self.ic_map.sample_base_
        
        ## trainability:
        self.all_parameters_trainable = True
        if initialise: self.initialise()
        else: pass

    def _forward_represenation_(self, r, crystal_index=0):
        # relevant ic_map indexed to transform r -> x
        ladJ = 0.0
        #X, ladj_rep = self.ic_maps[crystal_index].forward_(r)
        X, ladj_rep = self.ic_map.forward_(r)
        ladJ += ladj_rep
        return X, ladJ
    
    def _inverse_represenation_(self, X, crystal_index=0):
        # relevant ic_map indexed  to transform x -> r
        ladJ = 0.0
        #r, ladj_rep = self.ic_maps[crystal_index].inverse_(X)
        r, ladj_rep = self.ic_map.inverse_(X)
        ladJ += ladj_rep
        return r, ladJ

class NN_interface_sc_T(NN_interface_sc_multimap):

    # thermal expansion is ignored here in this version (fixed box for all Ts)

    def __init__(self,
                 name : str,
                 paths_datasets : list,
                 fraction_training : float = 0.8, # 0 << x < 1
                 running_in_notebook : bool = False,
                 training : bool = True,
                 ):
        self.dir_results_BAR = DIR_main+'/NN/training_results/BAR/'
        self.dir_results_misc = DIR_main+'/NN/training_results/misc/'
        self.dir_results_samples = DIR_main+'/NN/training_results/samples/'
        self.dir_results_models = DIR_main+'/NN/training_results/fitted_models/'
        self.inds_rand = None

        self.name = name + '_SC_' # ...
        assert type(paths_datasets) == list
        self.paths_datasets = paths_datasets
        self.fraction_training = fraction_training
        self.running_in_notebook = running_in_notebook 
        self.training = training
        self.model_class = PGMcrys_v1_T
        self.ic_map_class = SingleComponent_map

        ##
        self.n_crystals = len(self.paths_datasets)

        self.nns = [NN_interface_sc(
                    name = name,
                    path_dataset = self.paths_datasets[i],
                    fraction_training = self.fraction_training,
                    training = self.training, 
                    ic_map_class = self.ic_map_class,
                    ) for i in range(self.n_crystals)]
        
        if self.training:
            # sorting systems/datasets in ascending order of temperature
            self.nns = [self.nns[i] for i in np.argsort([nn.sc.T for nn in self.nns])]
            self.Ts = [nn.sc.T for nn in self.nns]

            for k in range(self.n_crystals):
                assert self.Ts[k] == self.nns[k].T
        else: pass

    def set_ic_map_step1(self, ind_root_atom=11, option=None):
        assert self.training

        self.ind_root_atom = ind_root_atom
        self.option = option
        PDBmol = './'+str(self.nns[0].sc._single_mol_pdb_file_)

        self.ic_map = self.ic_map_class(PDB_single_mol = PDBmol)
        self.ic_map.set_ABCD_(ind_root_atom = self.ind_root_atom, option = self.option)

    def set_ic_map_step2(self,
                         check_PES = True,
                         ):
        self.b0 = np.array(self.nns[0].b0)
        self.n_mol = int(self.nns[0].sc.n_mol)
        for k in range(self.n_crystals):
            check = np.abs(np.sum(self.nns[k].b0 - self.b0)) < 1e-5
            assert check, '! please use NN_interface_sc_TV instead' # adding this soon
            assert self.nns[k].sc.n_mol == self.n_mol
            
            # also the packing needs to be the same, assumed same when all datasets(T) here are ran from same initial structure
            # thermal expansion is ignored here because not sure how to interpolate between different boxes

        for k in range(self.n_crystals):
            self.nns[k].r = self.ic_map.remove_COM_from_data_(self.nns[k].r)
            self.nns[k].set_training_validation_split_()
            if check_PES: self.nns[k].check_PES_matching_dataset_()
            else: pass

    def set_ic_map_step3(self,
                         n_mol_unitcell : int = None,
                         COM_remover = WhitenFlow,
                        ):
        if n_mol_unitcell is None:
            print('!! set_ic_map_step3 : n_mol_unitcell was not provided')
            n_mol_unitcell = None
        else: pass
        self.n_mol_unitcell = n_mol_unitcell

        r_cat = []

        for k in range(self.n_crystals):
            r_cat.append(self.nns[k].r)

        r_cat = np.concatenate(r_cat, axis=0)

        self.ic_map.initalise_(r_dataset = r_cat,
                               b0 = self.b0, # same in all datasets
                               n_mol_unitcell = self.n_mol_unitcell,
                               COM_remover = COM_remover,
                               )
        del r_cat

        m = 1000
        for k in range(self.n_crystals):
            r = np.array(self.nns[k].r_validation[:m])
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
                value = max([np.abs(np.array(_x)).max() for _x in x]) # 1.0000004 , 1.0000006
                if value > 1.000001: print('!!', value)
                else: print(value)

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
                                    ic_map = self.ic_map,
                                    n_temperatures = self.n_crystals,
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
                print('T=',self.Ts[crystal_index],'inv_test_res0',inv_test_res0)
        else: pass

######################################################################################################



