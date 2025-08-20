from .representation_layers import *
from .pgm import *

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

class SingleComponent_monatomic_map(SingleComponent_map):
    '''IC_map for monatomic crystal with one type of atom
       implemented as simplifaction of methods inherited from SingleComponent_map 
    '''
    def __init__(self,
                 PDB_single_mol: str,   
                ):
        super().__init__(PDB_single_mol)
        self.VERSION = 'NEW'
        self.fixed_one_atom = None

    def set_ABCD_(self, ind_root_atom = None, option=None):
        self.inds_unpermute_atoms = 'not used'
        self.ABCD_IC_inverse = 'not used'
        self.n_atoms_IC = 'not used'
        self.inds_atoms_IC = 'not used'
        self.ABCD_IC = 'not used'
        self.ABCD = 'not used'
        self.inds_atoms_CB = 'not used'

        assert self.n_atoms_mol == 1
        print('set_ABCD_ : nothing was defined')

    def _first_times_crystal_(self, shape):
        assert len(shape) == 3
        self.n_mol = shape[1] // self.n_atoms_mol
        assert self.n_mol == shape[1] / self.n_atoms_mol
        self.n_atoms_crys = self.n_atoms_mol * self.n_mol
        self.n_DOF_crys = self.n_atoms_crys*3 - 3
        self.n_mol_supercell = self.n_mol
        
    ## ## ## ##

    def _forward_(self, r):
        rO = reshape_to_atoms_tf_(r, n_molecules=self.n_mol, n_atoms_in_molecule=self.n_atoms_mol)
        # (m,m_mol,3) is already rO
        X_IC = None ; ladJ_IC = 0.0
        X_CB = [rO] ; ladJ_CB = 0.0
        return X_IC, X_CB, ladJ_IC, ladJ_CB

    def _inverse_(self, X_IC, X_CB):
        r = X_CB[0]
        ladJ_IC = 0.0
        ladJ_CB = 0.0
        return r, ladJ_IC, ladJ_CB

    ## ## ## ##
    
    def _forward_init_(self, r, batch_size = 10000):
        shape = r.shape ;  m = shape[0]
        self._first_times_crystal_(shape)
        rO = []

        X_IC = None
        ladJ_IC = 0.0
        ladJ_CB = 0.0
        for i in range(m//batch_size):
            _from = i*batch_size
            _to = (i+1)*batch_size
            x_IC, x_CB, ladj_IC, ladj_CB = self._forward_(r[_from:_to])
            rO.append(x_CB[0])

        x_IC, x_CB, ladj_IC, ladj_CB = self._forward_(r[_to:])
        rO.append(x_CB[0])

        rO = tf.concat(rO,axis=0)

        print('initialising on',rO.shape[0],'datapoints provided')
        X_CB = [rO]

        return X_IC, X_CB, ladJ_IC, ladJ_CB

    def remove_COM_from_data_(self, r):

        # Whitening happens later. Model linked to dataset
        # Not generalisable to other MD data (for model that is not invariant to permuations)

        assert self.fixed_one_atom in [None, False]

        self.COM_remover = WhitenFlow
        self.fixed_one_atom = False

        # carefull with np.float32 when averaging a constant
        r = np.array(r).astype(np.float32)
        self._first_times_crystal_(r.shape)
        rO = reshape_to_atoms_np_(r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=self.n_mol)
        print('COM removed from data, without taking into account PBC of the box')
        r_com = rO.mean(-2,keepdims=True)

        return r-r_com

    def fix_one_atom_in_data_(self, r, b0):

        # new; not in parent class, alternative to remove_COM_from_data_. 
        # May be generalisable to other MD data (if model invariant to permuations).

        assert self.fixed_one_atom in [None, True]
        assert b0.shape == (3,3)

        self.COM_remover = NotWhitenFlow
        self.fixed_one_atom = True

        r = np.array(r).astype(np.float32)
        self._first_times_crystal_(r.shape)
        rO = reshape_to_atoms_np_(r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=self.n_mol)
        print('fixing one atom in data, without taking into account PBC of the box')
        #r_com = rO[...,:1,:]

        sO = np.einsum('...i,ij->...j', rO, np.linalg.inv(b0))
        sO -= sO[...,:1,:]
        sO = np.mod(sO, 1.0)
        output = np.einsum('...i,ij->...j', sO, b0)

        return output

    def reshape_to_unitcells_(self, *args, **kwargs):
        print(' reshape_to_unitcells_: not used')
        return 'not used'

    def initalise_(self,
                   r_dataset,
                   b0,
                   batch_size=10000,
                   n_mol_unitcell = 'not used',
                   COM_remover = 'not used',
                   focused = 'not used',
                   ):
        none, X_CB = self._forward_init_(r_dataset, batch_size=batch_size)[:2]
        rO = X_CB[0].numpy()
        assert rO.shape[1] == self.n_atoms_crys
        assert rO.shape[1] == self.n_mol

        if b0.shape == (3,3):
            self.single_box_in_dataset = True
            print('SCmap.initalise_ : a single box provided -> Cartesian transformed by SCmap.')
            self.b0_constant = np2tf_(np.array(b0).astype(np.float32)) # reshape just to catch error
            self.supercell_Volume = det_3x3_(self.b0_constant, keepdims=True)
            self.ladJ_unit_box_forward = - tf.math.log(self.supercell_Volume)*(self.n_mol-1.0)
            self.b0_inv_constant  = np2tf_(np.linalg.inv(b0).astype(np.float32)) 
            self.h0_constant = box_forward_(self.b0_constant[tf.newaxis,...])[0][0]
            rO = np.einsum('...i,ij->...j', rO, self.b0_inv_constant.numpy())
        else:
            assert b0.shape == (rO.shape[0], 3,3)
            self.single_box_in_dataset = False
            print('SCmap.initalise_ : >1 boxes provided -> Transform Cartesian outside SCmap.')
            # (m,N,3) -> (m,n_mol,n_atoms_mol,3) -> (m,n_mol,3,3) -> (m,n_mol,3)
            rO = np.einsum('omi,oij->omj', rO, np.linalg.inv(b0))

        rO_flat = reshape_to_flat_np_(rO, self.n_mol, 1) # (m, 3*n_mol)

        if self.COM_remover == NotWhitenFlow:
            self.WF = self.COM_remover(rO_flat, whiten_anyway=False)
        else:
            'checking for molecules not to be jumping more than half box length'
            check = get_ranges_centres_(rO_flat, axis=[0])
            assert all([x < 0.5 for x in check[0]])
            self.WF = self.COM_remover(rO_flat, removedims=3)

        xO_flat = self.WF._forward_np(rO_flat)                # (m, 3*n_mol-3)

        if isinstance(self.WF, NotWhitenFlow):
            # one atom fixed, can do periodic model
            self.periodic_mask_rO = np.array([1,1,1]) # new; not in parent class
            self.wrap_to_01_or_no_wrap_ = self.wrap_to_01_ # used after WF, volume change 0 because translation
            self.ranges_xO = np2tf_(np.ones(3*(self.n_mol-1)))      # 1
            self.centres_xO = np2tf_(np.ones(3*(self.n_mol-1))*0.5) # 0.5
        else:
            # COM removed by Euclidean whitening, pbc not known anymore, non-periodic
            self.periodic_mask_rO = np.array([0,0,0]) # new; not in parent class
            self.wrap_to_01_or_no_wrap_ = self.no_wrap_    # dummy function
            self.ranges_xO, self.centres_xO = get_ranges_centres_(xO_flat, axis=[0]) # (3*n_mol-3,)

        self.focused = focused

        self.n_DOF_mol = None
        self.ln_base_P = - np2tf_( 3*(self.n_mol-1)*np.log(2.0) )
        self.ln_base_C = 0.0
        
        self.periodic_mask = 'not used' # the only one in the parent class, dont use.
        self.periodic_mask = np.array([0,0])

    def wrap_to_01_(self, x):
        return tf.math.floormod(x, 1.0)
    
    def no_wrap_(self, x):
        return x
    
    def set_periodic_mask_(self,):
        print(' set_periodic_mask_ : not used')

    @property
    def flexible_torsions(self,):
        print(' flexible_torsions : not used')
        return 'not used'
    
    @property
    def current_masks_periodic_torsions_and_Phi(self,):
        print('current_masks_periodic_torsions_and_Phi : not used')
        return 'not used'

    def match_topology_(self, ic_maps:list):
        print(' match_topology_ : not used')
    
    def sample_base_P_(self, m):
        # p_{0} (base distribution) positions:
        return tf.clip_by_value(tf.random.uniform(shape=[m, self.n_mol_supercell-1, 3], minval=-1.0, maxval=1.0), -1.0, 1.0)

    def sample_base_C_(self, m):
        return 'not used'
        #return tf.clip_by_value(tf.random.uniform(shape=[m, self.n_mol_supercell-1, 2], minval=-1.0, maxval=1.0), -1.0, 1.0)
  
    # def sample_base_(self, m):  same as inherited

    # def ln_base_(self, inputs): same as inherited
        
    def forward_(self, r):
        # r : (m, N, 3) = (m, n_mol, 3)
        ladJ = 0.0
        rO = self._forward_(r)[1][0]

        if self.single_box_in_dataset:
            ''' Cartesian forward '''
            rO = tf.einsum('...i,ij->...j', rO, self.b0_inv_constant)
            ladJ += self.ladJ_unit_box_forward
            # rO                  : (m, n_mol, 3)
            # self.ladJ_box       : (1,1)

            xO, ladJ_whiten = self.WF.forward(reshape_to_flat_tf_(rO, n_molecules=self.n_mol, n_atoms_in_molecule=1))
            ladJ += ladJ_whiten 
            # xO                  : (m, 3*(n_mol-1))
            # ladJ_whiten         : (m,1)

            xO = self.wrap_to_01_or_no_wrap_(xO) # volume change 0 because translation (if any)

            # still needed if wrap_to_01_ above, to rescale : [0,1) -> [-1,1)
            xO, ladJ_scale_xO = scale_shift_x_(xO, physical_ranges_x = self.ranges_xO, physical_centres_x = self.centres_xO, forward = True)
            ladJ += ladJ_scale_xO
            # xO                  : (m, 3*(n_mol-1))
            # ladJ_scale_xO       : (,)
        else: 
            xO = rO # (m, n_mol, 3)

        xO = tf.reshape(xO, [-1, self.n_mol-1, 3])

        variables = [xO, 'none']
        return variables, ladJ

    def sample_nvt_h_(self, m):
        print('sample_nvt_h_ : not')

    def inverse_(self, variables_in):
        # variables_in:
        # xO : (m, 3*(n_mol-1))
        ladJ = 0.0
        ######################################################

        xO, none = variables_in

        xO = tf.reshape(xO, [-1, (self.n_mol-1)*3])
        
        if self.single_box_in_dataset:
            ''' Cartesian inverse '''
            xO, ladJ_scale_xO = scale_shift_x_(xO, physical_ranges_x = self.ranges_xO, physical_centres_x = self.centres_xO, forward = False)
            ladJ += ladJ_scale_xO
            rO, ladJ_whiten = self.WF.inverse(xO)
            ladJ += ladJ_whiten 
            rO = reshape_to_atoms_tf_(rO, n_atoms_in_molecule=1, n_molecules=self.n_mol)
            rO = tf.einsum('...i,ij->...j', rO, self.b0_constant)
            ladJ -= self.ladJ_unit_box_forward
        else:
            rO = xO # (m, n_mol, 3)

        X_CB = [rO]

        r = self._inverse_(X_IC=None, X_CB=X_CB)[0]
        # r : (m, N, 3)

        # just to match remove_COM_from_data_
        #r -= tf.reduce_mean(r, axis=-2, keepdims=True)
        
        return r, ladJ

def get_pos_encoding_tf_(pos, C=6, dim_embedding=3):
    '''ABW
    Lambda replacement
    '''
    return cast_32_([tf.sin(pos/(C**(2*i/dim_embedding))) for i in range(1,dim_embedding+1)] + [tf.cos(pos/(C**(2*i/dim_embedding))) for i in range(1,dim_embedding+1)])

#get_pos_encoding_tf_ = lambda pos, C=6, dim_embedding=3 : cast_32_([tf.sin(pos/(C**(2*i/dim_embedding))) for i in range(1,dim_embedding+1)] + [tf.cos(pos/(C**(2*i/dim_embedding))) for i in range(1,dim_embedding+1)])

class CUSTOM_LAYER_LJ(tf.keras.layers.Layer):
    def __init__(self,
                 periodic,
                 name = 'CUSTOM_LAYER_LJ',
                 ##
                 n_bins = 5,
                 min_bin_width = 0.001,
                 knot_slope_range = [0.001, 50.0],
                 ##
                 K = 100,
            ):
        super().__init__()
        self.K = K

        self.periodic = periodic
        if periodic: 
            self.periodic_mask = [1]
            d = 2
        else:        
            self.periodic_mask = [0]
            d = 1

        hidden_activation = tf.nn.leaky_relu


        Z = self.K // 2
        self.Z = Z

        self.AUG_0_MLP_ = MLP(
                            dims_outputs = [K*Z],
                            dims_hidden = [K*Z]*2, # try reduce number of hidden layer from 2 (current) to 1
                            hidden_activation = hidden_activation,
                            outputs_activations = None,
                            name =  name+'aug_0_MLP',
                            )
        self.AUG_1_MLP_ = MLP(
                            dims_outputs = [K*Z],
                            dims_hidden = [K*Z]*2, # try reduce number of hidden layer from 2 (current) to 1
                            hidden_activation = hidden_activation,
                            outputs_activations = None,
                            name =  name+'aug_1_MLP',
                            )

        self.SPLINE =  SPLINE_COUPLING_HALF_LAYER(

                 periodic_mask = self.periodic_mask + [0]*(d*2+K),
                 cond_mask = [1] + [0]*(d*2+K),
                 use_tfp = False,
                 n_bins = n_bins,
                 min_bin_width = min_bin_width,
                 knot_slope_range = knot_slope_range,
                 ##
                 dims_hidden = None,
                 n_hidden = 2, # also, maybe 1 enough
                 hidden_activation = hidden_activation,
                 ##
                 name = name,
        )

    def pos_encoding_extend_(self, x, n_mol):
        return tf.stack([tf.concat([x[:,i,:],[get_pos_encoding_tf_(i)]*x.shape[0]], axis=-1) for i in range(n_mol)],axis=-2)

    def transform_(self, x, A, B, forward=True):

        # N here = N-1 because one particle missing out of the total number of particles

        # x (m,N,d)
        # A (m,N,d)
        # B (m,N,d)

        # d = 1

        N = x.shape[1]

        if self.periodic:
            _A = tf.concat([tf.cos(A*PI), tf.sin(A*PI)], axis=-1) # (m,N,1) -> (m,N,2)
            _B = tf.concat([tf.cos(B*PI), tf.sin(B*PI)], axis=-1) # (m,N,1) -> (m,N,2)
            # d := 2
            _A_B = tf.concat([_A,_B], axis=-1) # (m,N,2*d)
            _A_B_with_pos_encoding = _A_B      # (m,N,2*d + dim_pos_encoding) ; dim_pos_encoding = 0
        
        else:
            _A = A
            _B = B
            # d == 1
            _A_B = tf.concat([_A,_B], axis=-1) # (m,N,2*d)
            _A_B_with_pos_encoding = self.pos_encoding_extend_(_A_B, N) # (m,N,2*d + dim_pos_encoding) ; dim_pos_encoding = 6?
            # pos_encoding prevents invariance of model to permutaions of the molecules (LJ particles)

        ## ## 

        psi_0 = self.AUG_0_MLP_( _A_B_with_pos_encoding )[0]
        psi_1 = self.AUG_1_MLP_( _A_B_with_pos_encoding )[0]

        psi_0 = tf.reshape(psi_0, [-1, N, self.K, self.Z])
        psi_1 = tf.reshape(psi_1, [-1, N, self.K, self.Z])
        
        ''' a bit like attention but cheaper then AT_MLP. 
            Can try different things here to maybe couple the molecules (LJ particles) better,
            Even this current simplified version is expensive due to quadratic cost scaling of the following dot product with n_mol (=N+1)
        '''
        psi = tf.einsum('oikz,ojkz->oijk', psi_0, psi_1) # (m,N,N,K) # expensive part of training in high enough n_mol ()>= 100)

        psi = tf.reduce_mean(tf.nn.tanh(psi), axis=-2) # (m,N,K)

        psi = tf.concat([_A_B, psi], axis=-1) # (m,N,2*d) | (m,N,K) -> (m,N,d*2+K)
        psi = [ psi ]

        # psi last axis are all cond_mask = [0]: not transformed
        Inputs = tf.concat([x]+psi, axis=-1) # (m,N,1) | (m,N,d*2+K) -> (m, N, 1 + d*2+K)
    
        Outputs, ladJ = self.SPLINE.transform_(Inputs, forward=forward) 
        y = Outputs[...,:1]

        return y, ladJ

class PGMcrys_LJ(tf.keras.models.Model, model_helper_PGMcrys_v1, model_helper):
    @staticmethod
    def load_model(path_and_name : str, VERSION='NEW'):
        return PGMcrys_LJ._load_model_(path_and_name, PGMcrys_LJ)

    def __init__(self,
                 ic_maps : list,
                 n_layers : int = 4,
                 optimiser_LR_decay = [0.001,0.0],
                 n_att_heads = 'not used',
                 initialise = True, # for debugging in eager mode
                 ):
        super().__init__()
        self.DIM_connection = None
        self.init_args = {  'ic_maps' : ic_maps,
                            'n_layers' : n_layers,
                            'optimiser_LR_decay' : optimiser_LR_decay,
                        }

        if str(type(ic_maps)) not in ["<class 'list'>","<class 'tensorflow.python.training.tracking.data_structures.ListWrapper'>"]: 
            ic_maps = [ic_maps]
        else: pass
        self.ic_maps = ic_maps

        assert all([ic_map.n_atoms_mol == 1 for ic_map in self.ic_maps])

        self.fixed_one_atom = self.ic_maps[0].fixed_one_atom
        assert all([ic_map.fixed_one_atom is self.fixed_one_atom for ic_map  in self.ic_maps])

        if self.fixed_one_atom:
            self.periodic_mask = [1]
            self.periodic = True
        else:
            self.periodic_mask = [0]
            self.periodic = False

        self.n_maps = len(self.ic_maps)

        self.dim_crystal_encoding = 1
        self.crystal_encodings = np2tf_(np.linspace(-1.,1.,self.n_maps))

        ####
        self.n_layers = n_layers
        self.optimiser_LR_decay = optimiser_LR_decay

        n_bins = 5             # ok
        ##

        self.flow_masks = [self.ic_maps[crystal_index].flow_mask_xO for crystal_index in range(self.n_maps)]

        aug_dimension = 50
        self.layers_X = [CUSTOM_LAYER_LJ(
                            periodic = self.periodic,
                            name = 'CUSTOM_LAYER_LJ_X',
                            ##
                            n_bins = n_bins,
                            min_bin_width = 0.001,
                            knot_slope_range = [0.001, 50.0],
                            ##
                            K = aug_dimension,
                        ) for i in range(self.n_layers)]
        self.layers_Y = [CUSTOM_LAYER_LJ(
                            periodic = self.periodic,
                            name = 'CUSTOM_LAYER_LJ_Y',
                            ##
                            n_bins = n_bins,
                            min_bin_width = 0.001,
                            knot_slope_range = [0.001, 50.0],
                            ##
                            K = aug_dimension,
                        ) for i in range(self.n_layers)]
        self.layers_Z = [CUSTOM_LAYER_LJ(
                            periodic = self.periodic,
                            name = 'CUSTOM_LAYER_LJ_Z',
                            ##
                            n_bins = n_bins,
                            min_bin_width = 0.001,
                            knot_slope_range = [0.001, 50.0],
                            ##
                            K = aug_dimension,
                        ) for i in range(self.n_layers)]


        ## p_{0}:
        self.ln_base_ = self.ic_maps[0].ln_base_
        self.sample_base_ = self.ic_maps[0].sample_base_

        self.n_mol = self.ic_maps[0].n_mol # only for the TRAINER to plot lattice FE, but will not be correct if different n_mol with multimap
        
        ## trainability:
        self.all_parameters_trainable = True
        if initialise: self.initialise()
        else: pass
    
    ##

    def _forward_coupling_(self, X, crystal_index=0):
        '''
        transform the 3 variables in each particle one by one conditioned on the other two variables
        '''
        ladJ = 0.0
        x_P, none = X

        _X = x_P[...,:1]
        _Y = x_P[...,1:2]
        _Z = x_P[...,2:]

        for i in range(self.n_layers):

            _X, ladj = self.layers_X[i].transform_(_X, A=_Y, B=_Z, forward=True) ; ladJ += ladj
            _Y, ladj = self.layers_X[i].transform_(_Y, A=_X, B=_Z, forward=True) ; ladJ += ladj
            _Z, ladj = self.layers_X[i].transform_(_Z, A=_X, B=_Y, forward=True) ; ladJ += ladj

        x_P = tf.concat([_X, _Y, _Z], axis=-1)

        Z = [x_P, 'none']
        return Z, ladJ
    
    def _inverse_coupling_(self, Z, crystal_index=0):
        '''
        transform the 3 variables in each particle one by one conditioned on the other two variables
        '''
        ladJ = 0.0
        x_P, none = Z

        _X = x_P[...,:1]
        _Y = x_P[...,1:2]
        _Z = x_P[...,2:]

        for i in reversed(range(self.n_layers)):

            _Z, ladj = self.layers_X[i].transform_(_Z, A=_X, B=_Y, forward=False) ; ladJ += ladj
            _Y, ladj = self.layers_X[i].transform_(_Y, A=_X, B=_Z, forward=False) ; ladJ += ladj
            _X, ladj = self.layers_X[i].transform_(_X, A=_Y, B=_Z, forward=False) ; ladJ += ladj

        x_P = tf.concat([_X, _Y, _Z], axis=-1)

        X = [x_P, 'none']

        return X, ladJ

    def test_inverse_(self, r, crystal_index=0, graph=True):
        # test invertibility of the model in both directions
        ''' same as inherited, except for z[0] - _z[0] is the only part here '''

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

            err_r_backward = [np.abs(z[0] - _z[0])]

            err_r_backward = [[x.mean(), x.max()] for x in err_r_backward]
            err_l_backward = np.array(ladJrz + ladJzr)
            err_l_backward = [err_l_backward.mean(), err_l_backward.min(), err_l_backward.max()]

        except:
            err_r_backward = [None]
            err_l_backward = [None]

        # [2,3], [[2]*..,3] 
        return [err_r_forward, err_l_forward], [err_r_backward, err_l_backward]

################################################################################################################

from ..interface import *

class NN_interface_sc_LJ(NN_interface_helper):
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
                 ic_map_class = SingleComponent_monatomic_map,
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

    def set_ic_map_step1(self,):
        '''
        self.sc.mol is available for this reason; to check the indices of atoms in the molecule
        once (ind_root_atom, option) pair is chosen, keep a fixed note of this for this molecule
        '''
        assert self.training

        PDBmol = './'+str(self.sc._single_mol_pdb_file_)
        self.ic_map = self.ic_map_class(PDB_single_mol = PDBmol)
        self.ic_map.set_ABCD_()

    def set_ic_map_step2(self,
                         inds_rand=None,
                         check_PES = True,
                         fixed_atom = True,
                         ):
        self.fixed_atom = fixed_atom

        if self.fixed_atom: self.r = self.ic_map.fix_one_atom_in_data_(self.r, b0=self.b0)
        else:               self.r = self.ic_map.remove_COM_from_data_(self.r)

        self.set_training_validation_split_(inds_rand=inds_rand)
        if check_PES: self.check_PES_matching_dataset_()
        else: pass

    def set_ic_map_step3(self,*args,**kwargs):
        self.n_mol_unitcell = 'not used'

        self.ic_map.initalise_(r_dataset = self.r, b0 = self.b0)

        m = 1000
        r = np.array(self.r_validation[:m])
        x, ladJf = self.ic_map.forward_(r)
        _r, ladJi = self.ic_map.inverse_(x)
        print('ic_map inversion errors on a small random batch:')
        print('positons:', np.abs(_r-r).max())
        print('volume:', np.abs(ladJf + ladJi).max())

        value = max([np.abs(np.array(_x)).max() for _x in x[:1]]) # 1.0000004 , 1.0000006
        if value > 1.000001: print('!!', value)
        else: print(value)

    # set_model not here and the rest not here.

class NN_interface_sc_LJ_multimap(NN_interface_helper):
    def __init__(self,
                 name : str,
                 paths_datasets : list,
                 fraction_training : float = 0.8, # 0 << x < 1
                 running_in_notebook : bool = False,
                 training : bool = True,
                 model_class = PGMcrys_LJ,
                 ic_map_class = SingleComponent_monatomic_map,
                 fixed_atom = True,
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
        self.fixed_atom = fixed_atom
        ##
        self.n_crystals = len(self.paths_datasets)

        self.nns = [NN_interface_sc_LJ(
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
        [nn.set_ic_map_step1() for nn in self.nns];

    def set_ic_map_step2(self, check_PES=True):
        [nn.set_ic_map_step2(inds_rand=None, check_PES=check_PES, fixed_atom=self.fixed_atom) for nn in self.nns];

    def set_ic_map_step3(self,*args,**kwargs):
        self.n_mol_unitcells ='not used'

        for i in range(self.n_crystals):
            self.nns[i].set_ic_map_step3()
        # TODO: check that the later merging reaches back into each nns[i].ic_map

    def set_model(self,
                  learning_rate = 0.001,
                  evaluation_batch_size = 5000,
                  ##
                  n_layers = 4,
                  n_att_heads = 4,
                  initialise = True, # for debugging in eager mode
                  test_inverse = True,
                  ):
        self.model = self.model_class(
                                    ic_maps = [nn.ic_map for nn in self.nns],
                                    n_layers = n_layers,
                                    optimiser_LR_decay = [learning_rate, 0.0],
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
              training_batch_size = 500,

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

                        # evalaution cost vs statistical significance of gradient of the loss (fixing to 1000, but can change)
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
