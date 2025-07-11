
''' pgm.py

    c : POSITIONS_FLOW_LAYER
    c : C2P_connector_v1
    c : CONFORMER_FLOW_LAYER
    c : P3_version_SingleComponent_map_LITE
    f : load_P3_PGMcrys
    c : model_helper_PGMcrys_v1
    c : PGMcrys_v1          # good (whole P3)
    c : C2P_connector_v2
    c : C2P_connector_v2_PI # further work area
    c : PGMcrys_v2          # good (similar to PGMcrys_v1 but self.n_params/self.n_mol is smaller)

    TODO: !! add back the general model for a single molecule in vaccum
    TODO: ! add back both P2 models (ice), including the Moebius model (good in non-crystaline systems)
'''

from .model_helper import *
from .spline_layer import *
from .representation_layers import *

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

class POSITIONS_FLOW_LAYER(tf.keras.layers.Layer):
    '''
    PGMcrys_v1 only
    '''
    def __init__(self,
                 n_mol,
                 layer_index,
                 DIM_P2C_connection : int, # DIM_connection (per molecule)
                 DIM_C2P_connection : int, # n_P2C * DIM_connection ; in M_{/mol} n_P2C = n_mol
                 name = 'POSITIONS_FLOW_LAYER',
                 n_hidden_main = 2,
                 n_hidden_connection = 1,
                 hidden_activation = tf.nn.leaky_relu,
                 ##
                 use_tfp = False,
                 n_bins = 5,
                 min_bin_width = 0.001,
                 knot_slope_range = [0.001, 50.0],
                 n_P2C = None, # in M_{/mol} n_P2C = n_mol
                 ):
        super().__init__()
        self.n_mol = n_mol
        self.layer_index = layer_index
        self.DIM_P2C_connection = DIM_P2C_connection
        self.DIM_C2P_connection = DIM_C2P_connection
        self.custom_name = name
        self.n_hidden_main = n_hidden_main 
        self.n_hidden_connection = n_hidden_connection
        self.hidden_activation = hidden_activation

        if n_P2C is None: self.n_P2C = self.n_mol
        else:             self.n_P2C = n_P2C

        self.n_DOFs_pos = (self.n_mol - 1)*3
        self.periodic_mask_positons = np.array(self.n_DOFs_pos*[0])
        self.periodic_mask = self.periodic_mask_positons
        self.dim_flow = len(self.periodic_mask)
        _cond_masks = get_coupling_masks_(self.dim_flow)
        self.cond_mask = _cond_masks[np.mod(self.layer_index, len(_cond_masks))]

        self.bijector = SPLINE_COUPLING_LAYER(
                                        periodic_mask = self.periodic_mask.tolist() + [0]*self.DIM_C2P_connection,
                                        cond_mask = self.cond_mask.tolist() + [2]*self.DIM_C2P_connection,
                                        ##
                                        use_tfp = use_tfp,
                                        n_bins = n_bins,
                                        min_bin_width = min_bin_width,
                                        knot_slope_range = knot_slope_range,
                                        ##
                                        name = self.custom_name,
                                        half_layer_class = SPLINE_COUPLING_HALF_LAYER,
                                        kwargs_for_given_half_layer_class = {'n_hidden' : n_hidden_main,
                                                                             'dims_hidden' : None,
                                                                             'hidden_activation': self.hidden_activation,
                                                                            },
                                        )

        self.P2C_MLP_ =  MLP(dims_outputs = [self.DIM_P2C_connection]*self.n_P2C,
                            outputs_activations = [tf.nn.tanh]*self.n_P2C,
                            dims_hidden = [self.n_DOFs_pos]*n_hidden_connection,
                            hidden_activation = self.hidden_activation,
                            name = self.custom_name,
                            )

    def forward_(self, input, aux):
        # input  : (m, 1, 3*(n_mol-1))
        x = tf.concat([input, aux],axis=-1)
        x, ladJ = self.bijector.forward(x)
        output = x[...,:self.dim_flow]
        # output : (m, 1, 3*(n_mol-1))
        # ladJ   : (m,1)
        return output, ladJ

    def inverse_(self, input, aux):
        # input  : (m, 1, 3*(n_mol-1))
        x = tf.concat([input,aux],axis=-1)
        x, ladJ = self.bijector.inverse(x)
        output = x[...,:self.dim_flow]
        # output : (m, 1, 3*(n_mol-1))
        # ladJ   : (m,1)
        return output, ladJ
    
    def convert_to_aux_(self, input):
        # input     : (m, 1, 3n + 3(n-1))
        mlp_input = input[:,0]
        # mlp_input : (m, 3n + 3(n-1))
        mlp_outputs = self.P2C_MLP_(mlp_input)
        mlp_output = tf.stack(mlp_outputs, axis=-2)
        # output    : (m, self.n_P2C, DIM_P2C_connection)
        return mlp_output

    def convert_to_flow_(self, pos):
        # input  : (m, 3*(n_mol-1))
        # output : (m, 1, 3*(n_mol-1))
        return pos[:,tf.newaxis]
    
    def convert_from_flow_(self, pos):
        # input  : (m, 1, 3*(n_mol-1))
        # output : (m, 3*(n_mol-1))
        return pos[:,0]

class C2P_connector_v1(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        [setattr(self, key, value) for key, value in kwargs.items()]
        '''
        standard DNN. Inputs and outputs are flat. P3 method.
        '''
        self.connection_NN_ = MLP(  dims_outputs = [self.DIM_C2P_connection],
                                    outputs_activations = [tf.nn.tanh],
                                    dims_hidden = [self.dim_flow_euclidean*4]*self.n_hidden_connection,
                                    hidden_activation = self.hidden_activation,
                                    name = None, # not transferable, PGMcrys_v1 overall not transferable
                                )
    def __call__(self, input):
        # ^ input : (m, n_mol, dim_flow)
        x_P = tf.gather(input, self.inds_P, axis=-1) * PI
        x_O = tf.gather(input, self.inds_O, axis=-1)
        Input = tf.concat([tf.cos(x_P), tf.sin(x_P), x_O], axis=-1)
        # ^ Input : (m, n_mol, dim_flow_euclidean)
        Input = tf.reshape(Input, [-1, self.n_mol * self.dim_flow_euclidean])
        # ^ Input : (m, n_mol * dim_flow_euclidean)
        ouput = self.connection_NN_(Input)[0]
        # ^ output : (m, self.DIM_C2P_connection)
        Output = ouput[:, tf.newaxis]
        # ^ Output : (m, 1, self.DIM_C2P_connection)
        return Output

class CONFORMER_FLOW_LAYER(tf.keras.layers.Layer):
    ''' Notes
    '''
    def __init__(self,
                 periodic_mask,
                 layer_index,
                 n_mol,
                 DIM_P2C_connection, # per moelcule
                 DIM_C2P_connection,
                 name = 'CONFORMER_FLOW_LAYER',
                 ##
                 half_layer_class = SPLINE_COUPLING_HALF_LAYER,
                 kwargs_for_given_half_layer_class = {'n_hidden' : 2,
                                                      'dims_hidden':None,
                                                      'hidden_activation':tf.nn.leaky_relu,
                                                     },
                 ##
                 use_tfp = False,
                 n_bins = 5,
                 min_bin_width = 0.001,
                 knot_slope_range = [0.001, 50.0],
                 ##
                 custom_coupling_mask = None, 
                 n_hidden_connection = 1,
                 connector_type = C2P_connector_v1,
            ):
        super().__init__()
        self.periodic_mask = np.array(periodic_mask).flatten()
        self.dim_flow = len(self.periodic_mask)
        self.inds_P = np.where(self.periodic_mask==1)[0]
        self.inds_O = np.where(self.periodic_mask==0)[0]
        self.n_P = len(self.inds_P)
        self.n_O = len(self.inds_O)

        self.layer_index = layer_index
        if custom_coupling_mask is not None:
            self.cond_mask = custom_coupling_mask
        else:
            _cond_masks = get_coupling_masks_(self.dim_flow)
            self.cond_mask = _cond_masks[np.mod(self.layer_index, len(_cond_masks))]

        self.n_mol = n_mol
        self.DIM_P2C_connection = DIM_P2C_connection
        self.DIM_C2P_connection = DIM_C2P_connection 
            
        self.custom_name = name
        self.half_layer_class = half_layer_class
        self.kwargs_for_given_half_layer_class = kwargs_for_given_half_layer_class

        self.use_tfp = use_tfp
        self.n_bins = n_bins
        self.min_bin_width = min_bin_width
        self.knot_slope_range = knot_slope_range

        self.n_hidden_connection = n_hidden_connection
        
        ##
        self.bijector = SPLINE_COUPLING_LAYER(
                                        periodic_mask = self.periodic_mask.tolist() + [0]*self.DIM_P2C_connection,
                                        cond_mask = self.cond_mask.tolist() + [2]*self.DIM_P2C_connection,
                                        use_tfp = use_tfp,
                                        n_bins = n_bins,
                                        min_bin_width = min_bin_width,
                                        knot_slope_range = knot_slope_range,
                                        ##
                                        name = self.custom_name,
                                        half_layer_class = self.half_layer_class,
                                        kwargs_for_given_half_layer_class = self.kwargs_for_given_half_layer_class,
                                        )
        
        if self.DIM_C2P_connection is not None:
            self.dim_flow_euclidean = self.n_P*2 + self.n_O
            # check if old models still load correctly after this is in a differnt class
            self.connector = connector_type(n_mol = self.n_mol,
                                            inds_P = self.inds_P,
                                            inds_O = self.inds_O,
                                            DIM_P2C_connection = self.DIM_P2C_connection,
                                            DIM_C2P_connection = self.DIM_C2P_connection,
                                            dim_flow_euclidean = self.dim_flow_euclidean,
                                            n_hidden_connection = self.n_hidden_connection,
                                            hidden_activation = self.kwargs_for_given_half_layer_class['hidden_activation'],
                                            custom_name = self.custom_name 
                                            )
        else: pass

    def forward_(self, input, aux):
        # input : (m, n_mol, dim_flow)
        # aux   : (m, n_mol, DIM_connection)
        X = tf.concat([input,aux], axis=-1)
        Y, ladJ = self.bijector.forward(X)
        output = Y[...,:self.dim_flow]
        # output : (m, n_mol, dim_flow)
        # ladJ   : (m, 1)
        return output, ladJ
    
    def inverse_(self, input, aux):
        # input : (m, n_mol, dim_flow)
        # aux   : (m, n_mol, DIM_connection)
        Y = tf.concat([input,aux], axis=-1)
        X, ladJ = self.bijector.inverse(Y)
        output = X[...,:self.dim_flow]
        # output : (m, n_mol, dim_flow)
        # ladJ   : (m, 1)
        return output, ladJ
    
    '''
    def convert_to_aux_(self, input):
        x_P = tf.gather(input, self.inds_P, axis=-1) * PI
        x_O = tf.gather(input, self.inds_O, axis=-1)
        mlp_input = tf.concat([tf.cos(x_P), tf.sin(x_P), x_O], axis=-1)
        mlp_ouput = self.connection_MLP_A_(self.connection_MLP_(mlp_input))[0]
        return mlp_ouput # (m, n_mol, DIM_connection)
    '''

    def convert_to_aux_(self, input):
        return self.connector(input)
    
###########################

class P3_version_SingleComponent_map_LITE(SingleComponent_map):

    '''
    Allows models from the paper to be loaded,
    Only the ic_map = SingleComponent_map part of the model was changed slightly since the paper (P3)
    To be able to load all of the saved models from before,
    this class fixes the errors that would otherwise appear if loading those models with current code.
    '''
    
    def __init__(self, 
                 ic_map_OLD_version,
                 ):
        self.__dict__.update(ic_map_OLD_version.__dict__) # all attributes useful to have

        if self.PDB_single_mol[:4] == './MM':
            self.PDB_single_mol = DIR_main+'MM'+self.PDB_single_mol[4:]
        else: pass
        
        super().__init__(self.PDB_single_mol) # some methods did not change

        ''' box ticking for what is different or same:
        all local method defintions are minimum necesary for forward_ and inverse_ to match the old behaviour
        
        forward_ methods/object definition locations:
            permute_unitcell_tf_ : method defined here : depreciated in new v.
            _forward_ : method ok : SAME (no changes since previous v. ; both code and its location)
            self.WF : instance ok : SAME
            scale_shift_x_ : method ok : SAME (will work with the arrays)
            static_rotations_layer : instance ok : SAME
            Focused_Hemisphere : adjusted here, same as before
            forward_reshape_cells_tf_ : method defined here : depreciated in new v.
            inverse_reshape_cells_tf_cat_ : method defined here : depreciated in new v.
            FocusedBonds : method ok : forward_ and inverse_ match the old v. (will work with the arrays)
            FocusedAngles : method ok : forward_ and inverse_ match the old v.  (will work with the arrays)
            FocusedTorsions : method ok : forward_ and inverse_ match the old v.  (will work with the arrays)
            inverse_reshape_cells_tf_ : method defined here : depreciated in new v.
        inverse_ :
            _inverse_ : method ok : SAME
        '''
        ''' methods/arrays needed by the new v. of the model to work with old version of ic_map:
            periodic_mask : ok : SAME
            n_mol : ok now : number_of_parallel_molecules in the model deals with this
            sample_base_C_ : redefined here (no variables were harmonic in those models so it is simplified here)
            sample_base_P_ : redefiend here, only n_mol == n_mol_supercell are the same
            ln_base_C_     : redefined here (no variables were harmonic in those models so it is simplified here)
            ln_base_P_     : redefiend here
        '''
        self.b0_constant = tf.reshape(np2tf_(box_inverse_(self.h0_constant)[0]), [3,3])
        self.b0_inv_constant = np2tf_(np.linalg.inv(self.b0_constant.numpy()).astype(np.float32))
        assert self.b0_constant.shape == self.b0_inv_constant.shape == (3,3)
        self.supercell_Volume = det_3x3_(self.b0_constant, keepdims=True)
        self.ladJ_unit_box_forward = - tf.math.log(self.supercell_Volume)*(self.n_mol-1.0)
        #
        self.Focused_Hemisphere.static_rotations_layer = identity_shift()
        #
        self.ln_base_P = - np2tf_( 3*(self.n_mol-1)*np.log(2.0) )
        self.VERSION = 'P3' # paper 3 version (old)

    ## old v. reshaping
    def permute_unitcell_tf_(self,r):
        return reshape_to_atoms_tf_(    tf.gather(  reshape_to_molecules_tf_(r,
                                                    n_atoms_in_molecule=self.n_atoms_mol,
                                                    n_molecules=self.n_mol), 
                                        self.inds_permute_unitcell, axis=1),
                                    n_atoms_in_molecule=self.n_atoms_mol,
                                    n_molecules=self.n_mol)
    def unpermute_unitcell_tf_(self,r):
        return reshape_to_atoms_tf_(    tf.gather(  reshape_to_molecules_tf_(r,
                                                    n_atoms_in_molecule=self.n_atoms_mol,
                                                    n_molecules=self.n_mol), 
                                        self.inds_unpermute_unitcell, axis=1),
                                    n_atoms_in_molecule=self.n_atoms_mol,
                                    n_molecules=self.n_mol)

    def forward_reshape_cells_tf_(self, x):
        n = self.n_molecules_unitcell
        return tf.concat([x[:,i*n:(i+1)*n] for i in range(self.n_unitcells)], axis=0)
    
    def inverse_reshape_cells_tf_(self, x):
        m = x.shape[0] // self.n_unitcells
        return tf.stack([x[i*m:(i+1)*m] for i in range(self.n_unitcells)], axis=1)

    def inverse_reshape_cells_tf_cat_(self, x):
        m = x.shape[0] // self.n_unitcells
        return tf.concat([x[i*m:(i+1)*m] for i in range(self.n_unitcells)], axis=1)
    
    ##
    def ln_base_C_(self, z):
        # this number will be in the imported ic_map
        return self.ln_base_flowing
    
    def ln_base_P_(self, z):
        # same
        return self.ln_base_P

    def sample_base_C_(self, m):
        # some chosen marginal variables could be kept harmonic in the older version; depreciated 
        # !! could add back in new version for keeping bond lengths constant in constrained data
        # as in old v. this requires X_C to be split and joined into flowing and non_flowing variables past coupling layers
        return tf.clip_by_value(tf.random.uniform(shape=[m, self.n_unitcells, self.n_flowing], minval=-1.0, maxval=1.0), -1.0, 1.0)
  
    def sample_base_P_(self, m):
        # this is same
        return tf.clip_by_value(tf.random.uniform(shape=[m, 3*(self.n_mol-1)], minval=-1.0,  maxval=1.0), -1.0, 1.0)
    
    ##

    def forward_(self, r):
        # r : (m, N, 3)
        ladJ = 0.0
        r = self.permute_unitcell_tf_(r)

        X_IC, X_CB, ladJ_IC, ladJ_CB = self._forward_(r)
        ladJ += tf.reduce_sum(ladJ_IC + ladJ_CB, axis=-2) # (m,1)
        rO, q, a, d0, d1 = X_CB

        rO = tf.einsum('omi,ij->omj', rO, self.b0_inv_constant)
        ladJ += self.ladJ_unit_box_forward
        xO, ladJ_whiten = self.WF.forward(reshape_to_flat_tf_(rO, n_molecules=self.n_mol, n_atoms_in_molecule=1))
        ladJ += ladJ_whiten 
        xO, ladJ_scale_xO = scale_shift_x_(xO, physical_ranges_x = self.ranges_xO, physical_centres_x = self.centres_xO, forward = True)
        ladJ += ladJ_scale_xO

        q = self.static_rotations_layer.forward_(q)
        xq, ladJ_rotations = self.Focused_Hemisphere.forward_(q)
        ladJ += ladJ_rotations 

        X_IC = self.forward_reshape_cells_tf_(X_IC)
        a    = self.forward_reshape_cells_tf_(a)
        d0   = self.forward_reshape_cells_tf_(d0)
        d1   = self.forward_reshape_cells_tf_(d1)
        xq   = self.forward_reshape_cells_tf_(xq)

        bonds = X_IC[...,0]  ; bonds = tf.concat([d0,d1,bonds],axis=-1)
        angles = X_IC[...,1] ; angles = tf.concat([a,angles],axis=-1)
        torsions = X_IC[...,2]

        x_bonds, ladJ_scale_bonds = self.Focused_Bonds(bonds, forward=True)
        ladJ += ladJ_scale_bonds * self.n_unitcells
        x_angles, ladJ_scale_angles = self.Focused_Angles(angles, forward=True)
        ladJ += ladJ_scale_angles * self.n_unitcells
        x_torsions, ladJ_scale_torsions = self.Focused_Torsions(torsions, forward=True)
        ladJ += ladJ_scale_torsions * self.n_unitcells

        X = tf.concat([x_bonds, x_angles, x_torsions, xq], axis=-1)
        X = self.inverse_reshape_cells_tf_(X)
        X = tf.reshape(X, [-1, self.n_unitcells, self.n_molecules_unitcell * self.n_DOF_cell])

        variables = [xO , X]

        return variables, ladJ

    def inverse_(self, variables_in):
        ladJ = 0.0
        xO , X = variables_in

        X = tf.reshape(X, [-1, self.n_unitcells, self.n_molecules_unitcell , self.n_DOF_cell]) 
        X = tf.concat([X[:,i] for i in range(self.n_unitcells)], axis = 0)

        toA = self.n_atoms_IC+2
        toB = toA + self.n_atoms_IC+1
        toC = toB + self.n_atoms_IC

        x_bonds = X[...,:toA]
        x_angles = X[...,toA:toB]
        x_torsions = X[...,toB:toC]
        xq = X[...,toC:]

        torsions, ladJ_scale_torsions = self.Focused_Torsions(x_torsions, forward=False)
        ladJ += ladJ_scale_torsions * self.n_unitcells
        angles, ladJ_scale_angles = self.Focused_Angles(x_angles, forward=False)
        ladJ += ladJ_scale_angles * self.n_unitcells
        bonds, ladJ_scale_bonds = self.Focused_Bonds(x_bonds, forward=False)
        ladJ += ladJ_scale_bonds * self.n_unitcells

        d0 = bonds[...,:1]
        d1 = bonds[...,1:2]
        bonds = bonds[...,2:]
        a = angles[...,:1]
        angles = angles[...,1:]
        X_IC = tf.stack([bonds, angles, torsions], axis=-1)

        X_IC = self.inverse_reshape_cells_tf_cat_(X_IC)
        a    = self.inverse_reshape_cells_tf_cat_(a)
        d0   = self.inverse_reshape_cells_tf_cat_(d0)
        d1   = self.inverse_reshape_cells_tf_cat_(d1)
        xq    = self.inverse_reshape_cells_tf_cat_(xq)

        q, ladJ_rotations = self.Focused_Hemisphere.inverse_(xq)
        ladJ += ladJ_rotations 
        q = self.static_rotations_layer.inverse_(q)        

        xO, ladJ_scale_xO = scale_shift_x_(xO, physical_ranges_x = self.ranges_xO, physical_centres_x = self.centres_xO, forward = False)
        ladJ += ladJ_scale_xO
        rO, ladJ_whiten = self.WF.inverse(xO)
        ladJ += ladJ_whiten
        rO = reshape_to_atoms_tf_(rO, n_atoms_in_molecule=1, n_molecules=self.n_mol)
        rO = tf.einsum('omi,ij->omj', rO, self.b0_constant)
        ladJ -= self.ladJ_unit_box_forward

        X_CB = [rO, q, a, d0, d1]

        r, ladJ_IC, ladJ_CB = self._inverse_(X_IC=X_IC, X_CB=X_CB)
        ladJ += tf.reduce_sum(ladJ_IC + ladJ_CB, axis=-2)

        r = self.unpermute_unitcell_tf_(r) # r : (m, N, 3)

        return r, ladJ

def load_P3_PGMcrys(path_and_name, class_of_the_model):
    init_args, ws = load_pickle_(path_and_name)
    ic_maps_OLD_version, n_layers, optimiser_LR_decay, DIM_connection = init_args
    if str(type(ic_maps_OLD_version)) not in ["<class 'list'>","<class 'tensorflow.python.training.tracking.data_structures.ListWrapper'>"]: 
        ic_maps_OLD_version = [ic_maps_OLD_version]
    else: pass
    ic_maps = [P3_version_SingleComponent_map_LITE(x) for x in ic_maps_OLD_version]
    assert n_layers == 4
    assert DIM_connection == 10
    if 'alter' in path_and_name:
        init_kwargs = { 'ic_maps':ic_maps,
                        'n_layers':n_layers,
                        'optimiser_LR_decay':optimiser_LR_decay,
                        'DIM_connection': DIM_connection,
                        'n_att_heads':4,
                        'initialise':True,
                    }
    else:
        init_kwargs = { 'ic_maps':ic_maps,
                        'n_layers':n_layers,
                        'optimiser_LR_decay':optimiser_LR_decay,
                        'DIM_connection': DIM_connection,
                        'n_att_heads':2,
                        'initialise':True,
                    }

    model = class_of_the_model(**init_kwargs)
    for i in range(len(ws)):
        model.trainable_variables[i].assign(ws[i])
    return model

class model_helper_PGMcrys_v1:
    def __init__(self,):
        ''

    def _forward_represenation_(self, r, crystal_index=0):
        # relevant ic_map indexed to transform r -> x
        ladJ = 0.0
        X, ladj_rep = self.ic_maps[crystal_index].forward_(r)
        ladJ += ladj_rep
        return X, ladJ
    
    def forward_(self, r, crystal_index=0):
        # complete trasformation r -> z
        ladJ = 0.0
        X, ladj = self._forward_represenation_(r, crystal_index=crystal_index) ; ladJ += ladj
        Z, ladj = self._forward_coupling_(X, crystal_index=crystal_index)      ; ladJ += ladj
        return Z, ladJ

    def _inverse_represenation_(self, X, crystal_index=0):
        # relevant ic_map indexed  to transform x -> r
        ladJ = 0.0
        r, ladj_rep = self.ic_maps[crystal_index].inverse_(X)
        ladJ += ladj_rep
        return r, ladJ

    def inverse_(self, Z, crystal_index=0):
         # complete trasformation z -> r
        ladJ = 0.0
        X, ladj = self._inverse_coupling_(Z, crystal_index=crystal_index)      ; ladJ += ladj
        r, ladj = self._inverse_represenation_(X, crystal_index=crystal_index) ; ladJ += ladj
        return r, ladJ

    ## ## ## ##

    #@tf.function
    def step_ML_graph_(self, r_and_crystal_index:list=0):
        # no numpy (tf compiled graph) train one training step (maximise likelihood on MD data)
        r, crystal_index = r_and_crystal_index
        if self.all_parameters_trainable: variables = self.trainable_variables
        else: variables = self._trainable_variables
        x,  ladJrx = self._forward_represenation_(r, crystal_index=crystal_index)
        with tf.GradientTape() as tape:
            z, ladJxz = self._forward_coupling_(x, crystal_index=crystal_index)
            loss = - tf.reduce_mean(ladJxz)
        grads = tape.gradient(loss, variables)
        if no_nans_(grads): 
            ''' tensor -> bool, usually without complaints from tensorflow, but can be sometimes '''
            # grads = [tf.clip_by_value(x,-200.,200.) for x in grads]
            self.optimizer.apply_gradients(zip(grads, variables))
            no_nans = True
        else: no_nans = False
        ln_p = ladJrx + ladJxz + self.ln_base_(z)
        loss = - tf.reduce_mean(ln_p)
        return loss, no_nans

    def step_ML(self, r, crystal_index:int=0, u=None, batch_size:int=1000):
        # prepare numpy input and train one training step (maximise likelihood on MD data)
        # r : (m,n_atoms,3) : MD data
        inds_rand = np.random.choice(r.shape[0], batch_size, replace=False)
        av_neg_ln_q, no_nans = self.step_ML_graph_([np2tf_(r[inds_rand]), crystal_index])
        if no_nans: pass 
        else: print('!! nan was found in gradient, this gradient was not used.')
        av_neg_ln_q = av_neg_ln_q.numpy() # AVMD_T_s
        if u is None: AVMD_T_f = 0.0 - av_neg_ln_q
        else:         AVMD_T_f = u[inds_rand].mean() - av_neg_ln_q
        return AVMD_T_f

    #@tf.function
    def forward_graph_(self, r, crystal_index=0):
        # no numpy (tf compiled graph) forward
        return self.forward_(r, crystal_index=crystal_index)
    
    #@tf.function
    def inverse_graph_(self, z, crystal_index=0):
        # no numpy (tf compiled graph) inverse
        return self.inverse_(z, crystal_index=crystal_index)
    
    def forward(self, r, crystal_index:int = 0):
        return self.forward_graph_(np2tf_(r), crystal_index=crystal_index)

    def inverse(self, z, crystal_index:int = 0):
        return self.inverse_graph_(z, crystal_index=crystal_index)

    def ln_model(self, r, crystal_index=0):
        # evalaute ln_q(r) : normalised log probability of the datapoint (r) according to the model
        outputs, ladJrz = self.forward(np2tf_(r), crystal_index=crystal_index)
        ln_q = self.ln_base_(outputs) + ladJrz
        return np.array(ln_q)
        
    def sample_model(self, m:int, crystal_index:int = 0):
        # generate m random samples from the model
        z = self.sample_base_(m)
        r, ladJzr = self.inverse(z, crystal_index=crystal_index)
        ln_q = self.ln_base_(z) - ladJzr
        return np.array(r), np.array(ln_q)

    def test_inverse_(self, r, crystal_index=0, graph=True):
        # test invertibility of the model in both directions
        ''' same as method with the same name in model_helper but has crystal_index as arg '''

        r = np2tf_(r) 
        m = r.shape[0]

        if graph: f_ = self.forward  ; i_ = self.inverse
        else:     f_ = self.forward_ ; i_ = self.inverse_
        ##
        z, ladJrz   = f_(r, crystal_index=crystal_index)
        _r, ladJzr = i_(z, crystal_index=crystal_index)
        err_r_forward = np.abs(r - _r)
        err_r_forward = [err_r_forward.mean(), err_r_forward.max()]
        err_l_forward = np.array(ladJrz + ladJzr)
        err_l_forward = [err_l_forward.mean(), err_l_forward.min(), err_l_forward.max()]
        ##
        z = self.sample_base_(m)
        _r, ladJzr = i_(z, crystal_index=crystal_index)
        
        try: 
            _z, ladJrz = f_(_r, crystal_index=crystal_index)

            err_r_backward = [np.abs(x - _x) for x,_x in zip(z,_z)]

            err_r_backward = [[x.mean(), x.max()] for x in err_r_backward]
            err_l_backward = np.array(ladJrz + ladJzr)
            err_l_backward = [err_l_backward.mean(), err_l_backward.min(), err_l_backward.max()]

        except:
            err_r_backward = [None]
            err_l_backward = [None]

        # [2,3], [[2]*..,3] 
        return[err_r_forward, err_l_forward], [err_r_backward, err_l_backward]

class PGMcrys_v1(tf.keras.models.Model, model_helper_PGMcrys_v1, model_helper):
    #""" !! : molecule should have >3 atoms (also true in ic_map) """
    @staticmethod
    def load_model(path_and_name : str, VERSION='NEW'):
        if VERSION == 'NEW':
            return PGMcrys_v1._load_model_(path_and_name, PGMcrys_v1)
        else:
            return load_P3_PGMcrys(path_and_name, PGMcrys_v1)

    def __init__(self,
                 ic_maps : list,
                 n_layers : int = 4,
                 optimiser_LR_decay = [0.001,0.0],
                 DIM_connection = 10,
                 n_att_heads = 4,
                 initialise = True, # for debugging in eager mode
                 ):
        super().__init__()
        self.init_args = {  'ic_maps' : ic_maps,
                            'n_layers' : n_layers,
                            'optimiser_LR_decay' : optimiser_LR_decay,
                            'DIM_connection' : DIM_connection,
                            'n_att_heads' : n_att_heads}
        
        ####
        if str(type(ic_maps)) not in ["<class 'list'>","<class 'tensorflow.python.training.tracking.data_structures.ListWrapper'>"]: 
            ic_maps = [ic_maps]
        else: pass
        for ic_map in ic_maps:
            if hasattr(ic_map, 'single_box_in_dataset'): assert ic_map.single_box_in_dataset == True
            else: ic_map.single_box_in_dataset = True
        
        self.ic_maps = ic_maps
        self.n_mol = int(self.ic_maps[0].n_mol)
        assert all([ic_map.n_mol == self.n_mol for ic_map  in self.ic_maps])
        assert all([ic_map.n_atoms_mol == ic_maps[0].n_atoms_mol for ic_map in self.ic_maps])
        self.n_maps = len(self.ic_maps)

        if self.n_maps > 1:
            if self.ic_maps[0].VERSION == 'P3': pass
            else:
                print('matching ic_maps for the single model:')
                [ic_map.match_topology_(ic_maps) for ic_map in self.ic_maps]
            print('checking that ic_maps match the model:')
            assert all(np.abs(ic_map.periodic_mask - ic_maps[0].periodic_mask).sum()==0 for ic_map in self.ic_maps)
        else: pass
        self.periodic_mask = np.array(self.ic_maps[0].periodic_mask)
        assert all([np.abs(self.periodic_mask - ic_map.periodic_mask).sum() == 0 for ic_map in self.ic_maps])
        # preparing how psi_{C->P} and psi_{P->C} are extended along last axis with crystal encoding:
        if self.n_maps > 1:
            self.dim_crystal_encoding = 1
            self.crystal_encodings = np2tf_(np.linspace(-1.,1.,self.n_maps))
            self.C2P_extension_shape = np2tf_(np.zeros([1,1]))
            self.P2C_extension_shape = np2tf_(np.zeros([self.n_mol,1]))
        else:
            self.dim_crystal_encoding = 0
            self.crystal_encodings = np2tf_(np.array([0]))
            self.C2P_extension_shape = np2tf_(np.zeros([1,0]))
            self.P2C_extension_shape = np2tf_(np.zeros([self.n_mol,0]))

        ####

        self.n_layers = n_layers
        self.optimiser_LR_decay = optimiser_LR_decay
        self.DIM_connection = DIM_connection
        self.n_att_heads = n_att_heads

        ##
        if self.ic_maps[0].VERSION == 'P3': number_of_parallel_molecules = self.ic_maps[0].n_unitcells
        else: number_of_parallel_molecules = self.n_mol

        self.DIM_P2C_connection = self.DIM_connection  
        self.DIM_C2P_connection = number_of_parallel_molecules*self.DIM_connection
        self.DIM_C2C_connection = None

        n_hidden_main = 2 #2
        n_hidden_connection = 1
        hidden_activation = tf.nn.leaky_relu #tanh
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
                            n_P2C = number_of_parallel_molecules,
                        ) for i in range(self.n_layers)]

        self.layers_C = [ CONFORMER_FLOW_LAYER(
                            periodic_mask = self.periodic_mask,
                            layer_index = i,
                            DIM_P2C_connection = self.DIM_P2C_connection +  self.dim_crystal_encoding,
                            n_mol = number_of_parallel_molecules,
                            DIM_C2P_connection = self.DIM_C2P_connection,
                            n_hidden_connection = n_hidden_connection,
                            half_layer_class = SPLINE_COUPLING_HALF_LAYER_AT,
                            kwargs_for_given_half_layer_class = {
                                        'flow_mask' : None,
                                        'n_mol' : number_of_parallel_molecules,
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
        self.ln_base_ = self.ic_maps[0].ln_base_
        self.sample_base_ = self.ic_maps[0].sample_base_
        
        ## trainability:
        self.all_parameters_trainable = True
        if initialise: self.initialise()
        else: pass

    def get_C2P_P2C_extensions_(self, m, crystal_index):
        '''
        psi_{C->P}, psi_{P->C} are extended along last axis by 1 additional dimension
            This extra dimensions is called 'crystal encoding'.
            To be able to concateneate this crystal encoding,
            the batch size axes needs to match.
            The following steps adjust the batch axis:

        TODO: move this to layer, because this does not match the other types of layer.
            The other types of layer will use self.P2C_extension_shape for both P and C.
        '''

        number = self.crystal_encodings[crystal_index]

        C2P_extension = self.C2P_extension_shape + number   # (1, 1)
        C2P_extension = tf.stack([C2P_extension]*m, axis=0) # (m, 1, 1)

        P2C_extension = self.P2C_extension_shape + number   # (n_mol, 1)
        P2C_extension = tf.stack([P2C_extension]*m, axis=0) # (m, n_mol, 1)

        return [C2P_extension, P2C_extension] # crystal embeddings, zero dimensional if training on just 1 state

    ##

    def _forward_coupling_(self, X, crystal_index=0):
        # trainable trasformation x -> z, conditioned on crystal_index
        ladJ = 0.0
        x_P, X_C = X
        X_P = self.layers_P[0].convert_to_flow_(x_P)
        C2P_extension, P2C_extension = self.get_C2P_P2C_extensions_(m=x_P.shape[0], crystal_index=crystal_index)

        for i in range(self.n_layers):

            aux_C2P   = self.layers_C[i].convert_to_aux_(X_C)
            aux_C2P   = tf.concat([aux_C2P, C2P_extension], axis=-1)
            X_P, ladj = self.layers_P[i].forward_(X_P, aux=aux_C2P) ; ladJ += ladj

            aux_P2C   = self.layers_P[i].convert_to_aux_(X_P)
            aux_P2C   = tf.concat([aux_P2C, P2C_extension], axis=-1)
            X_C, ladj = self.layers_C[i].forward_(X_C, aux=aux_P2C) ; ladJ += ladj

        x_P = self.layers_P[-1].convert_from_flow_(X_P)
        Z = [x_P, X_C]
        return Z, ladJ
    
    def _inverse_coupling_(self, Z, crystal_index=0):
         # trainable trasformation z -> x, conditioned on crystal_index
        ladJ = 0.0
        x_P, X_C = Z
        X_P = self.layers_P[-1].convert_to_flow_(x_P)
        C2P_extension, P2C_extension = self.get_C2P_P2C_extensions_(m=x_P.shape[0], crystal_index=crystal_index)

        for i in reversed(range(self.n_layers)):

            aux_P2C   = self.layers_P[i].convert_to_aux_(X_P)
            aux_P2C   = tf.concat([aux_P2C, P2C_extension], axis=-1)
            X_C, ladj = self.layers_C[i].inverse_(X_C, aux=aux_P2C) ; ladJ += ladj

            aux_C2P   = self.layers_C[i].convert_to_aux_(X_C)
            aux_C2P   = tf.concat([aux_C2P, C2P_extension], axis=-1)
            X_P, ladj = self.layers_P[i].inverse_(X_P, aux=aux_C2P) ; ladJ += ladj

        x_P = self.layers_P[-1].convert_from_flow_(X_P)
        X = [x_P, X_C]
        return X, ladJ

"""

####################################################################################################

# attempts at PGMcrys_v2:

# Aims:
# smaller model that does comparably to PGMcrys_v1
# make transferable between different sizes of a supercell : trying with C2P_connector_v2_PI (method not yet found)

####################################################################################################

class C2P_connector_v2(tf.keras.layers.Layer):
    def __init__(self,  **kwargs):
        super().__init__()
        [setattr(self, key, value) for key, value in kwargs.items()]
        '''
        standard DNN. Inputs and outputs are flat. Not invariant to n_mol.
        Output reshaped differently (shaped as molecules).
        '''
        self.connection_NN_ = MLP( dims_outputs = [self.n_mol * self.DIM_P2C_connection],
                                    outputs_activations = [tf.nn.tanh],
                                    dims_hidden = [self.dim_flow_euclidean*4]*self.n_hidden_connection,
                                    hidden_activation = self.hidden_activation,
                                    name = None, # dont need name, not transferable
                                ) # output[0] because list
        
    def __call__(self, input):
        # ^ input : (m, n_mol, dim_flow)
        x_P = tf.gather(input, self.inds_P, axis=-1) * PI
        x_O = tf.gather(input, self.inds_O, axis=-1)
        Input = tf.concat([tf.cos(x_P), tf.sin(x_P), x_O], axis=-1)
        # ^ Input : (m, n_mol, dim_flow_euclidean)
        Input = tf.reshape(Input, [-1, self.n_mol * self.dim_flow_euclidean])
        # ^ Input : (m, n_mol * dim_flow_euclidean)
        ouput = self.connection_NN_(Input)[0]
        # ^ output : (m, n_mol * self.DIM_P2C_connection])
        Output = tf.reshape(ouput, [-1, self.n_mol, self.DIM_P2C_connection])
        # ^ Output : (m, n_mol, self.DIM_P2C_connection)
        return Output

class C2P_connector_v2_PI(tf.keras.layers.Layer):
    # permutatiannly invariant (PI) : all molecules treated alike, so can pass any n_mol.
    def __init__(self,  **kwargs):
        super().__init__()
        [setattr(self, key, value) for key, value in kwargs.items()]
        '''
        self.connection_NN_ = MLP(  dims_outputs = [self.DIM_P2C_connection],
                                    outputs_activations = [tf.nn.tanh],
                                    dims_hidden = [self.dim_flow_euclidean*4]*self.n_hidden_connection,
                                    hidden_activation = self.hidden_activation,
                                    name = self.custom_name, 
                                ) # output[0] because list
        '''
        #'''
        self.connection_NN_A_ = AT_MLP(
                                    n_mol = self.n_mol,                          # this
                                    n_heads = 10,                                 # any
                                    embedding_dim = self.DIM_P2C_connection*2,   # any
                                    output_dim = self.DIM_P2C_connection,        # this
                                    n_hidden_kqv = [1,1,1],                      # any
                                    hidden_activation =  self.hidden_activation,  
                                    one_hot_kqv = [False]*3,                     # False to keep PI
                                    name =  self.custom_name,                    # provide name
                                    mask_self = False,                           # this
                                    )
        self.connection_NN_ = lambda x : [tf.nn.tanh(self.connection_NN_A_(x))] # output[0] because list
        #'''
        '''
        self.connection_NN_A_ = AT_MLP(
                                    n_mol = self.n_mol,                          # this
                                    n_heads = 1,                                 # any
                                    embedding_dim = self.DIM_P2C_connection*3,   # any
                                    output_dim = self.dim_flow_euclidean,        # this
                                    n_hidden_kqv = [1,1,1],                      # any
                                    hidden_activation =  self.hidden_activation,
                                    one_hot_kqv = [False]*3,                     # False to keep PI
                                    name =  self.custom_name,                    # provide name
                                    mask_self = True,                            # True ok if some kind of res. connection
                                    )
        self.connection_NN_B_ = MLP(dims_outputs = [self.DIM_P2C_connection],
                                    outputs_activations = [tf.nn.tanh],
                                    dims_hidden = [self.dim_flow_euclidean]*1,
                                    hidden_activation = self.hidden_activation,
                                    name = self.custom_name,                     # provide name
                                    )
        self.connection_NN_ = lambda x : self.connection_NN_B_(self.connection_NN_A_(x) + x) # output[0] because list
        '''
    def __call__(self, input):
        # ^ input : (m, n_mol, dim_flow)
        x_P = tf.gather(input, self.inds_P, axis=-1) * PI
        x_O = tf.gather(input, self.inds_O, axis=-1)
        Input = tf.concat([tf.cos(x_P), tf.sin(x_P), x_O], axis=-1)
        # ^ Input : (m, n_mol, dim_flow_euclidean)
        Output = self.connection_NN_(Input)[0]
        # ^ Output : (m, n_mol ,DIM_P2C_connection])
        return Output

class PGMcrys_v2(tf.keras.models.Model, model_helper_PGMcrys_v1, model_helper):
    #''' !! : molecule should have >3 atoms (also true in ic_map) '''
    @staticmethod
    def load_model(path_and_name : str, VERSION='NEW'):
        return PGMcrys_v2._load_model_(path_and_name, PGMcrys_v2)

    def __init__(self,
                 ic_maps : list,
                 n_layers : int = 4,
                 optimiser_LR_decay = [0.001,0.0],
                 DIM_connection = 10,
                 n_att_heads = 4,
                 initialise = True, # for debugging in eager mode
                 ):
        super().__init__()
        self.init_args = {  'ic_maps' : ic_maps,
                            'n_layers' : n_layers,
                            'optimiser_LR_decay' : optimiser_LR_decay,
                            'DIM_connection' : DIM_connection,
                            'n_att_heads' : n_att_heads}
        
        ####
        if str(type(ic_maps)) not in ["<class 'list'>","<class 'tensorflow.python.training.tracking.data_structures.ListWrapper'>"]: 
            ic_maps = [ic_maps]
        else: pass
        self.ic_maps = ic_maps
        self.n_mol = int(self.ic_maps[0].n_mol)
        assert all([ic_map.n_mol == self.n_mol for ic_map  in self.ic_maps])
        assert all([ic_map.n_atoms_mol == ic_maps[0].n_atoms_mol for ic_map in self.ic_maps])
        self.n_maps = len(self.ic_maps)

        if self.n_maps > 1:
            print('matching ic_maps for the single model:')
            [ic_map.match_topology_(ic_maps) for ic_map in self.ic_maps]
            print('checking that ic_maps match the model:')
            assert all(np.abs(ic_map.periodic_mask - ic_maps[0].periodic_mask).sum()==0 for ic_map in self.ic_maps)
        else: pass
        self.periodic_mask = np.array(self.ic_maps[0].periodic_mask)
        assert all([np.abs(self.periodic_mask - ic_map.periodic_mask).sum() == 0 for ic_map in self.ic_maps])
        # preparing how psi_{C->P} and psi_{P->C} are extended along last axis with crystal encoding:
        if self.n_maps > 1:
            self.dim_crystal_encoding = 1
            self.crystal_encodings = np2tf_(np.linspace(-1.,1.,self.n_maps))
            self.extension_shape = np2tf_(np.zeros([self.n_mol,1]))
        else:
            self.dim_crystal_encoding = 0
            self.crystal_encodings = np2tf_(np.array([0]))
            self.extension_shape = np2tf_(np.zeros([self.n_mol,0]))

        ####
        self.n_layers = n_layers
        self.optimiser_LR_decay = optimiser_LR_decay
        self.DIM_connection = DIM_connection
        self.n_att_heads = n_att_heads

        connector_type = C2P_connector_v2
        #connector_type = C2P_connector_v2_PI # doesn't work well enough

        n_hidden_kqv = [2,2,2] # n_hidden_main
        n_hidden_decode = 1
        embedding_dim = self.DIM_connection
        n_hidden_connection = 1
        hidden_activation = tf.nn.leaky_relu
        n_bins = 5
        ##
        print('self.n_att_heads:', self.n_att_heads)
        print('connector_type:',   connector_type)

        self.layers_C = [ CONFORMER_FLOW_LAYER(
                            periodic_mask = self.periodic_mask,
                            layer_index = i,
                            n_mol = self.n_mol,
                            DIM_P2C_connection = self.DIM_connection + self.dim_crystal_encoding,
                            DIM_C2P_connection = self.DIM_connection,
                            n_hidden_connection = n_hidden_connection,
                            half_layer_class = SPLINE_COUPLING_HALF_LAYER_AT,
                            kwargs_for_given_half_layer_class = {
                                        'flow_mask' : None,
                                        'n_mol' : self.n_mol,
                                        'n_heads' : self.n_att_heads,
                                        'embedding_dim' : embedding_dim,
                                        'n_hidden_kqv' : n_hidden_kqv,
                                        'hidden_activation' : hidden_activation,
                                        'one_hot_kqv' : [False]*3,
                                        'n_hidden_decode' : n_hidden_decode,
                                        },
                            use_tfp = False,
                            n_bins = n_bins,
                            min_bin_width = 0.001,
                            knot_slope_range = [0.001, 50.0],
                            name = 'TRANSFERABLE_FLOW_LAYER',
                            connector_type = connector_type
                        ) for i in range(self.n_layers)]

        self.layers_P = [ CONFORMER_FLOW_LAYER(
                            periodic_mask = [0,0,0],
                            layer_index = i,
                            n_mol = self.n_mol,
                            DIM_P2C_connection = self.DIM_connection + self.dim_crystal_encoding,
                            DIM_C2P_connection = self.DIM_connection,
                            n_hidden_connection = n_hidden_connection,
                            half_layer_class = SPLINE_COUPLING_HALF_LAYER_AT,
                            kwargs_for_given_half_layer_class = {
                                        'flow_mask' : self.ic_maps[0].flow_mask_xO,
                                        'n_mol' : self.n_mol,
                                        'n_heads' : self.n_att_heads,
                                        'embedding_dim' : embedding_dim,
                                        'n_hidden_kqv' : n_hidden_kqv,
                                        'hidden_activation' : hidden_activation,
                                        'one_hot_kqv' : [False]*3,
                                        'n_hidden_decode' : n_hidden_decode,
                                        },
                            use_tfp = False,
                            n_bins = n_bins,
                            min_bin_width = 0.001,
                            knot_slope_range = [0.001, 50.0],
                            name = 'TRANSFERABLE_FLOW_LAYER',
                            connector_type = connector_type
                        ) for i in range(self.n_layers)]

        ## p_{0}:
        self.ln_base_ = self.ic_maps[0].ln_base_
        self.sample_base_ = self.ic_maps[0].sample_base_
        
        ## trainability:
        self.all_parameters_trainable = True
        if initialise: self.initialise()
        else: pass

    def get_extension_(self, m, crystal_index):
        number = self.crystal_encodings[crystal_index]
        extension = self.extension_shape + number   # (n_mol, 1)
        extension = tf.stack([extension]*m, axis=0) # (m, n_mol, 1)
        return extension # crystal embedding, zero dimensional if training on just 1 state

    ##

    def _forward_coupling_(self, X, crystal_index=0):
        # trainable trasformation x -> z, conditioned on crystal_index
        ladJ = 0.0
        x_P, X_C = X
        X_P = self.ic_maps[0].xO_reshape_(x_P, forward=True)
        extension = self.get_extension_(m=x_P.shape[0], crystal_index=crystal_index)

        for i in range(self.n_layers):

            aux_C2P   = self.layers_C[i].convert_to_aux_(X_C)
            aux_C2P   = tf.concat([aux_C2P, extension], axis=-1)
            X_P, ladj = self.layers_P[i].forward_(X_P, aux=aux_C2P) ; ladJ += ladj

            aux_P2C   = self.layers_P[i].convert_to_aux_(X_P)
            aux_P2C   = tf.concat([aux_P2C, extension], axis=-1)
            X_C, ladj = self.layers_C[i].forward_(X_C, aux=aux_P2C) ; ladJ += ladj

        x_P = self.ic_maps[0].xO_reshape_(X_P, forward=False)
        Z = [x_P, X_C]
        return Z, ladJ
    

    def _inverse_coupling_(self, Z, crystal_index=0):
         # trainable trasformation z -> x, conditioned on crystal_index
        ladJ = 0.0
        x_P, X_C = Z
        X_P = self.ic_maps[0].xO_reshape_(x_P, forward=True)
        extension = self.get_extension_(m=x_P.shape[0], crystal_index=crystal_index)

        for i in reversed(range(self.n_layers)):

            aux_P2C   = self.layers_P[i].convert_to_aux_(X_P)
            aux_P2C   = tf.concat([aux_P2C, extension], axis=-1)
            X_C, ladj = self.layers_C[i].inverse_(X_C, aux=aux_P2C) ; ladJ += ladj

            aux_C2P   = self.layers_C[i].convert_to_aux_(X_C)
            aux_C2P   = tf.concat([aux_C2P, extension], axis=-1)
            X_P, ladj = self.layers_P[i].inverse_(X_P, aux=aux_C2P) ; ladJ += ladj

        x_P = self.ic_maps[0].xO_reshape_(X_P, forward=False)
        X = [x_P, X_C]
        return X, ladJ

    ##

"""

####################################################################################################


