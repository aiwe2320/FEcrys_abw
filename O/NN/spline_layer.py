
''' spline_layer.py
    c : MLP
    f : get_pos_encoding_
    c : AT_MLP
    c : SPLINE_COUPLING_helper
    c : SPLINE_COUPLING_HALF_LAYER
    c : SPLINE_COUPLING_HALF_LAYER_AT
    c : SPLINE_COUPLING_LAYER
'''

import numpy as np
import tensorflow as tf

from .rqs import *

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

class MLP(tf.keras.layers.Layer):
    def __init__(self,
                 
                 dims_outputs        : list,
                 dims_hidden         : list,
                 hidden_activation   = tf.nn.silu,
                 outputs_activations : list = None,

                 **kwargs):
        super().__init__(**kwargs)
        ''' Multilayer Perceptron (MLP)
        '''
        n_hidden_layers = len(dims_hidden)
        n_output_layers = len(dims_outputs)

        if outputs_activations is None: outputs_activations = ['linear']*n_output_layers
        else: assert len(outputs_activations) == n_output_layers

        self.hidden_layers = [tf.keras.layers.Dense(dims_hidden[i], activation = hidden_activation) for i in range(n_hidden_layers)]
        self.output_layers = [tf.keras.layers.Dense(dims_outputs[j], activation = outputs_activations[j]) for j  in range(n_output_layers)]

    def call(self,
             x,
             ):
        '''
        Input:   x : (..., d) ; d = dimensionality of input
        Outputs: ys : list of outputs with shapes (m, dims_outputs[i]) for every output layer i.
        '''
        for layer in self.hidden_layers:
            x = layer(x)
        #if drop_rate > 0.0: x = tf.keras.layers.Dropout(rate = drop_rate)(x, training=True)
        #else: pass
        ys = [layer(x) for layer in self.output_layers]
        return ys # ! always a list, even if one element

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

get_pos_encoding_ = lambda pos, C=6, dim_embedding=3 : np.array([np.sin(pos/(C**(2*i/dim_embedding))) for i in range(1,dim_embedding+1)] + [np.cos(pos/(C**(2*i/dim_embedding))) for i in range(1,dim_embedding+1)])

class AT_MLP(tf.keras.layers.Layer):
    '''
    self-attention block
    '''
    def __init__(self,
                 n_mol : int,                     # number of molecules (this layer treats molecules in parallel)
                 n_heads : int,                   # C : number of seperate attentions to average over
                 embedding_dim : int,             # A : dimensinality that the information between molecules may have
                 output_dim : int,                # B : arbitrary dimensionality needed for the output
                 n_hidden_kqv : list = [1,1,1],   # number of hidden layers in the _k_, _q_ _v_ DNNs respectively
                 hidden_activation = tf.nn.silu,  # activation function same all hidden layers in the _k_, _q_ _v_ DNNs
                 one_hot_kqv = [True]*3,          # whether to add a positional encoding to the inputs
                 name = 'AT_MLP',                 # name passed down to trainable weights for indexing later (optional)
                 mask_self = True,                # if True only pairs of different molecules are coupled
                ):
        super().__init__()

        A = embedding_dim
        B = output_dim
        C = n_heads

        self.flow_mask_unmasked = tf.ones([n_mol]) # (n_mol,)

        if mask_self and n_mol > 1: self.mask_self = 1.0 - tf.eye(n_mol)     # (n_mol, m_mol) ones with zero diagonal
        else:                       self.mask_self = tf.ones([n_mol, n_mol]) # (n_mol, n_mol) ones

        ## input shape (m, n_mol, dim_input) ; m = batch size
        ## DNNs _k_,_q_,_v_ computations are parallel over the n_mol axis
        ## Unless distinguishable encodings are added with shape (n_mol, dim_encoding) to the input,
        ## or masking is provided to the call function (default self.flow_mask_unmasked),
        ## the output is unaffected by permutations along the n_mol axis of the input.

        self._k_ = MLP(
                        dims_outputs = [C*A],
                        dims_hidden = [C*A]*n_hidden_kqv[0],
                        hidden_activation = hidden_activation,
                        outputs_activations = None,
                        name =  name,
                    )

        self._q_ = MLP(
                        dims_outputs = [C*A],
                        dims_hidden = [C*A]*n_hidden_kqv[1],
                        hidden_activation = hidden_activation,
                        outputs_activations = None,
                        name =  name,
                    )
        self._v_ = MLP(
                        dims_outputs = [C*B],
                        dims_hidden = [C*B]*n_hidden_kqv[2],
                        hidden_activation = hidden_activation,
                        outputs_activations = None,
                        name =  name,
                    )

        self.A = A
        self.B = B
        self.C = C
        self.n_mol = n_mol

        self.normalisation = 1.0 / (A**0.5)

        self.pos_emb = [cast_32_(get_pos_encoding_(i)) for i in range(self.n_mol)] # default [(6,), ..., (6,)]

        if one_hot_kqv[0]: self.k_ = lambda x : self._k_(self.one_hot_extend_(x))[0]
        else:              self.k_ = lambda x : self._k_(x)[0]

        if one_hot_kqv[1]: self.q_ = lambda x : self._q_(self.one_hot_extend_(x))[0]
        else:              self.q_ = lambda x : self._q_(x)[0]

        if one_hot_kqv[2]: self.v_ = lambda x : self._v_(self.one_hot_extend_(x))[0]
        else:              self.v_ = lambda x : self._v_(x)[0]

    def one_hot_extend_(self, x):
        return tf.stack([tf.concat([x[:,i,:],[self.pos_emb[i]]*x.shape[0]], axis=-1) for i in range(self.n_mol)],axis=-2)

    def call(self,
             x,    # input with shape (m, n_mol, dim_input) ; m is batch_size
             flow_mask = None,
             ):
        if flow_mask is None: flow_mask = self.flow_mask_unmasked # (n_mol,)
        else: pass # not tested

        k = self.k_(x) # (m, n_mol, CA)
        q = self.q_(x) # (m, n_mol, CA)
        v = self.v_(x) # (m, n_mol, CB)

        k = tf.reshape(k, [-1,self.n_mol,self.C,self.A])
        q = tf.reshape(q, [-1,self.n_mol,self.C,self.A])
        v = tf.reshape(v, [-1,self.n_mol,self.C,self.B])

        arg = tf.einsum('i,j,...ica,...jca->...ijc', flow_mask, flow_mask, k, q) * self.normalisation
        
        w = tf.math.softplus(arg) + 1e-5
        w = tf.einsum('...ijc,ij->...ijc', w, self.mask_self)
        w /= tf.reduce_sum(w, axis=-2, keepdims=True)

        y = tf.einsum('j,...ijc,...jcb->...ib', flow_mask, w, v)

        return y  # output with shape (m, n_mol, output_dim) # can be used as an input to another MLP or AT_MLP

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

class SPLINE_COUPLING_helper:
    def __init__(self,):
        ''
    def init_(self,
              periodic_mask : list,
              cond_mask : list,
              flow_mask : tf.constant = None,
              use_tfp : bool = False,
              n_bins : int = 4,
              min_bin_width : float = 0.001,
              knot_slope_range : list = [0.001, 50.0],
              nk_for_periodic_encoding : int = 1,
             ):
        '''
        Inputs:
            periodic_mask  : (n_variables,)
                1 if variable is periodic,
                0 otherwise.
            cond_mask      : (n_variables,) ; conditioning mask
                1 if variable should be transformed, 
                0 if variable should help condition transformations of the former.
            flow_mask      : (1, n_mol, n_variables) ; default is None (no masking)
                1.0 if variable should be transformed,
                0.0 if variable should be kept constant.
                    This is mostly useful if the variables that should not be transformed
                    are different depending on the axis that is between the last axis, 
                    and the batch axis (i.e., first axis) of the input.
                        NB: the computations of the splines are still done on the 0.0 variables, but
                        these variables are returned back to original, with 0.0 set for log volume change.
                        This is why this mask is flaot, as it is actually multipled to the variables.
            use_tfp        : whether to the use tensorflow_probability version of rational quadratic spline (both are in rqs.py)
            n_bins         : number of bins for each spline (a 'bins' is an area between two neighboring knots)
            min_bin_width  : mimimum vertical/horizontal distance between any pair of neighboring knots
            knot_slope_range : minimum and maximum slopes allowed at any knot
            nk_for_periodic_encoding : number of orthogonal sines and cosines for encoding each periodic marginal variable
                Such encoding used before forwarding any periodic variable into a DNN (DNN can only deal with Euclidean variables)
        '''
        ## 0,1
        periodic_mask = np.array(periodic_mask).flatten().astype(np.int32)
        cond_mask = np.array(cond_mask).flatten().astype(np.int32)
        self.periodic_mask = periodic_mask 
        self.cond_mask = cond_mask
        # lengths of both masks should be equal
        assert len(self.cond_mask) == len(self.periodic_mask)
        self.n_variables = len(periodic_mask)

        self.inds_flowing_Periodic = np.where((cond_mask==1)&(periodic_mask==1))[0]
        self.inds_flowing_Ordinary = np.where((cond_mask==1)&(periodic_mask!=1))[0]
        self.inds_conditioning_Periodic = np.where((cond_mask==0)&(periodic_mask==1))[0]
        self.inds_conditioning_Ordinary = np.where((cond_mask==0)&(periodic_mask!=1))[0]

        cat_inds = np.concatenate([self.inds_flowing_Periodic,
                                   self.inds_flowing_Ordinary,
                                   self.inds_conditioning_Periodic,
                                   self.inds_conditioning_Ordinary,
                                   ])
        self.inds_unpermute = np.array([np.where(cat_inds == i)[0][0] for i in range(len(cat_inds))])

        self.n_flowing_Periodic = len(self.inds_flowing_Periodic)
        self.n_flowing_Ordinary = len(self.inds_flowing_Ordinary)
        self.n_conditioning_Periodic = len(self.inds_conditioning_Periodic)
        self.n_conditioning_Ordinary = len(self.inds_conditioning_Ordinary)
        self.dim_MLP_input = self.n_conditioning_Periodic*2 + self.n_conditioning_Ordinary
        
        ## 2
    
        self.ignore_sort_ladj_Periodic_ = lambda ladj : tf.reduce_sum(ladj, axis=(-1,-2))[...,tf.newaxis]
        self.ignore_sort_ladj_Ordinary_ = self.ignore_sort_ladj_Periodic_
        self.ignore_sort_variable_Periodic_ = lambda input, output : output
        self.ignore_sort_variable_Ordinary_ = self.ignore_sort_variable_Periodic_
    
        if flow_mask is not None:
            self.mask_flowing  = cast_32_(flow_mask)  # (1, n_mol, n_variables) 
            self.mask_constant = 1.0 - self.mask_flowing
            
            self.mask_flowing_P  = tf.gather(self.mask_flowing,  self.inds_flowing_Periodic, axis=-1) # (1, n_mol, self.n_flowing_Periodic)
            self.mask_flowing_O  = tf.gather(self.mask_flowing,  self.inds_flowing_Ordinary, axis=-1) # (1, n_mol, self.n_flowing_Ordinary)
            self.mask_constant_P = tf.gather(self.mask_constant, self.inds_flowing_Periodic, axis=-1) # (1, n_mol, self.n_flowing_Periodic)
            self.mask_constant_O = tf.gather(self.mask_constant, self.inds_flowing_Ordinary, axis=-1) # (1, n_mol, self.n_flowing_Ordinary)

            self.sort_ladj_Periodic_ = lambda ladj : tf.reduce_sum(ladj*self.mask_flowing_P, axis=(-1,-2))[...,tf.newaxis]
            self.sort_ladj_Ordinary_ = lambda ladj : tf.reduce_sum(ladj*self.mask_flowing_O, axis=(-1,-2))[...,tf.newaxis]
            self.sort_variable_Periodic_ = lambda input, output : output*self.mask_flowing_P + input*self.mask_constant_P
            self.sort_variable_Ordinary_ = lambda input, output : output*self.mask_flowing_O + input*self.mask_constant_O

        else:
            self.sort_ladj_Periodic_ = self.ignore_sort_ladj_Periodic_
            self.sort_ladj_Ordinary_ = self.ignore_sort_ladj_Ordinary_
            self.sort_variable_Periodic_ = self.ignore_sort_variable_Periodic_
            self.sort_variable_Ordinary_ = self.ignore_sort_variable_Ordinary_
        
        ## 3,4,5,6
        self.use_tfp = use_tfp
        self.n_bins = n_bins
        self.min_bin_width = min_bin_width
        self.knot_slope_range = knot_slope_range
        self.interval = [-1.0, 1.0]

        if self.use_tfp:
            self.rqs_Periodic_ = periodic_rqs_tfp_
            self.rqs_Ordinary_ = rqs_tfp_
            self.n_slopes = self.n_bins - 1
        else:
            self.rqs_Periodic_ = periodic_rqs_
            self.rqs_Ordinary_ = rqs_
            self.n_slopes = self.n_bins + 1

        self.dims_MLP_outputs = []

        nP = self.n_flowing_Periodic
        nO = self.n_flowing_Ordinary
        if nP > 0:
            # 2 * (w, h, s, shift)
            self.dims_MLP_outputs.append( 2 * (nP*self.n_bins + nP*self.n_bins + nP*self.n_slopes + nP) ) 
            if nO > 0:
                # 1 * (w, h, s)
                self.dims_MLP_outputs.append( nO*self.n_bins + nO*self.n_bins + nO*self.n_slopes )
                #self.transform_ = self.PO_
            else: 
                self.dims_MLP_outputs.append( 0 )
                #self.transform_ = self.P_
        else:
            self.dims_MLP_outputs.append( 0 )
            self.dims_MLP_outputs.append( nO*self.n_bins + nO*self.n_bins + nO*self.n_slopes )
            #self.transform_ = self.O_

        self.ind_MLP_ouputs_P_split = self.dims_MLP_outputs[0]//2
        assert self.ind_MLP_ouputs_P_split == self.dims_MLP_outputs[0]/2

        ## 7
        self.nk_for_periodic_encoding = nk_for_periodic_encoding

        PI = np.pi

        def cos_sin_(x, nk:int=1):
            x*=PI
            output = []
            for k in range(1,nk+1):
                output.append(tf.cos(k*x))
                output.append(tf.sin(k*x))
            return tf.concat(output, axis = -1)

        def cos_sin_1_(x):
            x*=PI
            return tf.concat( [tf.cos(x), tf.sin(x)], axis = -1)

        if self.nk_for_periodic_encoding == 1:
            self.cos_sin_ = cos_sin_1_
        else:
            self.cos_sin_ = lambda x : cos_sin_(x, nk=self.nk_for_periodic_encoding)
        
    def transform_(self, x, forward : bool, dont_mask=False):
        '''
        Inputs:
            x         : (m, n_mol, n_variables) input variables
                          ; where n_mol >= 1 (i.e., cannot have a missing axis for molecules)
            forward   : if True transfromation is in the forward direction, False for inverse
            dont_mask : if True, self.flow_mask is ignored as if it was None (default)
        Outputs:
            y         : (m, n_mol, n_variables) transformed variables
            ladJ      : (m, 1) elementwise sum_{i,j} log(dy[...,i,j]/dx[...,i,j])
        
        '''
        # variables that are flowing (unless masked out by force later)
        xP = tf.gather(x, self.inds_flowing_Periodic, axis=-1)
        xO = tf.gather(x, self.inds_flowing_Ordinary, axis=-1)

        # variables that will be used for conditioning the transfromatons of the variables above.
        xcP = tf.gather(x, self.inds_conditioning_Periodic, axis=-1)
        xcO = tf.gather(x, self.inds_conditioning_Ordinary, axis=-1)
        xc = tf.concat([self.cos_sin_(xcP), xcO], axis=-1) # ready for DNN

        pP, pO = self.MLP_(xc) # DNN or sequence of any computations of choice evalauted on xc
        # pP are parameters for splines dealing with xP
        # pO are parameters for splines dealing with xO

        ladJ = 0.0

        if self.n_flowing_Periodic > 0:
            ## Periodic:
            '''
            NB: there are two seperate transformations done to each periodic variable (i.e., two splines used)
            Although more expensive, this ensures that the entire domain can be transformed in this layer.
                One spline cannot transform the edges of the domain well, because the positions of the edge slopes are fixed.
                Two splines can cover the whole domain but the data needs to be shifted (under PBCs; floor_mod).
                Each of the two splines has its own shifts, with both shifts trainable (list_shifts below output from MLP).
                After each spline, the data is shifted back to original position. Any shifts are volume-preserving.
            '''
            pP = tf.stack([pP[...,:self.ind_MLP_ouputs_P_split],
                           pP[...,self.ind_MLP_ouputs_P_split:]],
                           axis=0)
            # since the parameters are flat need to separate them into 'widths', 'heights', 'slopes', 'shifts'
            n = self.n_flowing_Periodic * self.n_bins
            list_w = pP[...,:n]
            list_h = pP[...,n:2*n]
            list_shifts = pP[...,2*n:2*n+self.n_flowing_Periodic]
            list_s = pP[...,2*n+self.n_flowing_Periodic:]
            yP, ladj = self.rqs_Periodic_(x = xP,
                                          list_w = list_w,
                                          list_h = list_h,
                                          list_s = list_s,
                                          list_shifts = list_shifts,
                                          interval = self.interval,
                                          forward = forward,
                                          min_bin_width = self.min_bin_width,
                                          knot_slope_range = self.knot_slope_range,
                                         )
            '''
            'masking by force':
                if flow_mask was provided, and some entries are zeros, doing
                y = y*mask + x*(1-mask) leaves transformed only the entries with ones,
                while the entries with zeros recive an identity transformation.
                Similarly, ladj := ladj*mask, sets the log volumes of the identity transformations to zero.
                No gradient of the loss passes through the zeros.

                dont_mask : ignores any self.flow_mask, as if it is ones.

                sort_ladj_Periodic_ adds up the marginal log volumes to give ladJ has shape (m,1) ; m = batch_size
            '''
            if dont_mask:
                ladJ += self.ignore_sort_ladj_Periodic_(ladj) # (m,1)
                yP = self.ignore_sort_variable_Periodic_(input=xP, output=yP)
            else:
                ladJ += self.sort_ladj_Periodic_(ladj) # (m,1)
                yP = self.sort_variable_Periodic_(input=xP, output=yP)
        else:
            '''
            # incase there are no periodic marginals in the input of this layer, need a name yP later:
            '''
            yP = xP

        if self.n_flowing_Ordinary > 0:
            ## Ordinary (non-periodic variables):
            # same as above but no shifts are needed, and only 1 spline used
            m = self.n_flowing_Ordinary * self.n_bins
            w = pO[...,:m]
            h = pO[...,m:2*m]
            s = pO[...,2*m:]
            yO, ladj = self.rqs_Ordinary_(x = xO,
                                          w = w,
                                          h = h,
                                          s = s,
                                          interval = self.interval,
                                          forward = forward,
                                          min_bin_width = self.min_bin_width,
                                          knot_slope_range = self.knot_slope_range,
                                         )
            if dont_mask:
                ladJ += self.ignore_sort_ladj_Ordinary_(ladj) # (m,1)
                yO = self.ignore_sort_variable_Ordinary_(input=xO, output=yO)
            else:
                ladJ += self.sort_ladj_Ordinary_(ladj) # (m,1)
                yO = self.sort_variable_Ordinary_(input=xO, output=yO)
        else:
            yO = xO

        '''
        # variables are collected back along the last axis where only [yP, yO] were transformed
        # the varables that were not trasformed but need to be transformed ([xcP, xcO]),
        # require the same sort of pass through the above steps (using 1 - cond_mask as the next cond_mask)
        # since cond_mask is fixed in this object, two instances of this class are used to couple 
        # [yP, yO] with [xcP, xcO] in both directions.
        '''
        cat = tf.concat([yP, yO, xcP, xcO], axis=-1)
        y = tf.gather(cat, self.inds_unpermute, axis=-1)

        return y, ladJ
    
    def forward_(self, x, dont_mask=False):
        return self.transform_(x, forward=True, dont_mask=dont_mask)
    
    def inverse_(self, x, dont_mask=False):
        return self.transform_(x, forward=False, dont_mask=dont_mask)

class SPLINE_COUPLING_HALF_LAYER(tf.keras.layers.Layer,
                                 SPLINE_COUPLING_helper,
                                    ):
    '''
    SPLINE_COUPLING_helper with standard DNN used for parametrising the splines
    '''

    ''' self.init_ inputs template:
    periodic_mask,           
    cond_mask,               
    flow_mask = None,
    use_tfp = False,
    n_bins = 4,
    min_bin_width = 0.001,
    knot_slope_range = [0.001, 50.0],
    '''
    def __init__(self,
                 periodic_mask,
                 cond_mask,
                 use_tfp = False,
                 n_bins = 4,
                 min_bin_width = 0.001,
                 knot_slope_range = [0.001, 50.0],
                 ##
                 dims_hidden = None,
                 n_hidden = 2,
                 hidden_activation = tf.nn.silu,
                 ##
                 name = None,
                ):
        super().__init__()
        self.init_(
        periodic_mask = periodic_mask,         
        cond_mask = cond_mask,               
        flow_mask = None,
        use_tfp = use_tfp,
        n_bins = n_bins,
        min_bin_width = min_bin_width,
        knot_slope_range = knot_slope_range,
        )
        '''
        SPLINE_COUPLING_helper is ready,
        now only need to define the self.MLP_ method (this can be any method with dims_output = self.dims_MLP_outputs)
        In here, a standard DNN is used (i.e., class MLP).

        In self.transform_, pP and pO are the necesary outputs.
        If either pP or pO is not needed in that function, it is replaced with None output, defined here: 

        Reason why it is done this way is to prevent a warning message:
            tf.GradientTape gives a warning if output dimension is zero, although it still works fine after the warning.
        '''
        self.dims_hidden = dims_hidden 
        self.n_hidden = n_hidden
        self.hidden_activation = hidden_activation
        self.custom_name = name

        if self.dims_hidden is None:
            self.dims_hidden = [sum(self.dims_MLP_outputs)] * self.n_hidden
        else: pass

        if 0 in self.dims_MLP_outputs:
            self._MLP_ = MLP(
                            dims_outputs = [sum(self.dims_MLP_outputs)],
                            outputs_activations = None,
                            dims_hidden = self.dims_hidden,
                            hidden_activation = self.hidden_activation,
                            name = self.custom_name,
                            )
            if np.where(np.array(self.dims_MLP_outputs)==0)[0][0] == 0:
                self.MLP_ = lambda x : [None] + self._MLP_(x)
            else:
                self.MLP_ = lambda x : self._MLP_(x) + [None]
        else:
            self.MLP_ = MLP(
                            dims_outputs = self.dims_MLP_outputs,
                            outputs_activations = None,
                            dims_hidden = self.dims_hidden,
                            hidden_activation = self.hidden_activation,
                            name = self.custom_name,
                            )

class SPLINE_COUPLING_HALF_LAYER_AT(tf.keras.layers.Layer,
                                    SPLINE_COUPLING_helper,
                                   ):
    
    '''
    SPLINE_COUPLING_helper with AT_MLP involved for parametrising the splines
    '''
    
    def __init__(self,
                 periodic_mask,
                 cond_mask,
                 use_tfp = False,
                 n_bins = 4,
                 min_bin_width = 0.001,
                 knot_slope_range = [0.001, 50.0],

                 ##
                 flow_mask = None,
                 n_mol = 1,
                 n_heads = 2,
                 embedding_dim = 20,
                 n_hidden_kqv = [1,1,1],
                 hidden_activation = tf.nn.silu,
                 one_hot_kqv = [False]*3,
                 n_hidden_decode = 2,
                 add_residual = False,
                 ##
                 #new = False,

                 name = None,
                ):
        super().__init__()
        self.init_(
        periodic_mask = periodic_mask,         
        cond_mask = cond_mask,               
        flow_mask = flow_mask,
        use_tfp = use_tfp,
        n_bins = n_bins,
        min_bin_width = min_bin_width,
        knot_slope_range = knot_slope_range,
        )
        self.n_hidden_decode = n_hidden_decode
        self.hidden_activation = hidden_activation
        self.custom_name = name

        self.n_mol = n_mol
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.n_hidden_kqv = n_hidden_kqv
        self.hidden_activation = hidden_activation
        self.one_hot_kqv= one_hot_kqv

        self.add_residual = add_residual

        self.set_MLPs_default_()

        #if new: self.set_MLPs_new_()
        #else: self.set_MLPs_default_()
    
    def set_MLPs_default_(self,):
        '''
        Anything that works is allowed in this part of the model.
        Finding a combinatoon that works better is always useful.

        The settings that were used so far, are:

            y_flowing = splines(x_flowing, params = MLP( \psi_conditioning )

            Since the output dimensionality of the MLP is very high (spline params are many),
            if does not matter too much if:
                \psi_conditioning = [AT_MLP(x_conditioning), x_conditioning] # add_residual False
            OR; 
                \psi_conditioning = AT_MLP(x_conditioning) + x_conditioning  # add_residual True
        
        '''
        if 0 in self.dims_MLP_outputs:
            self._MLP1_ = MLP(
                        dims_outputs = [sum(self.dims_MLP_outputs)],
                        outputs_activations = None,
                        dims_hidden =  [sum(self.dims_MLP_outputs)] * self.n_hidden_decode,
                        hidden_activation = self.hidden_activation,
                        name = self.custom_name,
                        )
            if np.where(np.array(self.dims_MLP_outputs)==0)[0][0] == 0:
                self.MLP1_ = lambda x : [None] + self._MLP1_(x)
            else:
                self.MLP1_ = lambda x : self._MLP1_(x) + [None]
        else:
            self.MLP1_ = MLP(
                        dims_outputs = self.dims_MLP_outputs,
                        outputs_activations = None,
                        dims_hidden =  [sum(self.dims_MLP_outputs)] * self.n_hidden_decode,
                        hidden_activation = self.hidden_activation,
                        name = self.custom_name,
                        )

        if self.n_mol > 1:
            self.MLP0_ = AT_MLP(
                        n_mol = self.n_mol,
                        n_heads = self.n_heads,
                        embedding_dim = self.embedding_dim,
                        #output_dim = sum(self.dims_MLP_outputs),
                        output_dim = self.dim_MLP_input,
                        n_hidden_kqv = self.n_hidden_kqv,
                        hidden_activation = self.hidden_activation,
                        one_hot_kqv = self.one_hot_kqv,
                        name = self.custom_name,
                        )

            if self.add_residual:
                self.MLP_ = lambda x : self.MLP1_(self.MLP0_(x) + x)
            else:
                self.MLP_ = lambda x : self.MLP1_(tf.concat([self.MLP0_(x), x],axis=-1))
        else:
            self.MLP_ = lambda x : self.MLP1_(x) 

class SPLINE_COUPLING_LAYER(tf.keras.layers.Layer):
    '''
    General coupling layer wrapper, where can choose half layer type (half_layer_class arg).

    There are currenty two options for the half_layer_class:
        - SPLINE_COUPLING_HALF_LAYER
        - SPLINE_COUPLING_HALF_LAYER_AT

    kwargs_for_given_half_layer_class:
        dictionary of settings that are relevant to the chosen half_layer_class
    
    Template examples for choosing half_layer_class and kwargs_for_given_half_layer_class:

    """ standard coupling layer:
    half_layer_class = SPLINE_COUPLING_HALF_LAYER,
    kwargs_for_given_half_layer_class = {
                                        'n_hidden' : 2,
                                        'dims_hidden':None,
                                        'hidden_activation':tf.nn.silu,
                                        }
    """

    """ coupling layer involving self-attention:
    half_layer_class = SPLINE_COUPLING_HALF_LAYER_AT,
    kwargs_for_given_half_layer_class = {
                                        'flow_mask' : None,
                                        'n_mol' : 1,
                                        'n_heads' : 2,
                                        'embedding_dim' : 20,
                                        'n_hidden_kqv' : [1,1,1],
                                        'hidden_activation' : tf.nn.silu,
                                        'one_hot_kqv' : [False]*3,
                                        'n_hidden_decode' : 2,
                                        }
    """

    '''
    def __init__(self,
                periodic_mask,
                cond_mask,
                use_tfp = False,
                n_bins = 4,
                min_bin_width = 0.001,
                knot_slope_range = [0.001, 50.0],
                ##
                name = None,
                half_layer_class = SPLINE_COUPLING_HALF_LAYER,
                kwargs_for_given_half_layer_class = {'n_hidden' : 2,
                                                     'dims_hidden':None,
                                                     'hidden_activation':tf.nn.silu,
                                                    }
                ):
        super().__init__()
        self.custom_name = name 
        # name passed all the way down to any MLPs,
        # to be able to later index trainable weights of this whole layer.
        # If a unique name is given, trainable weights of this exact layer can be located in any_model(tf.keras.models.Model).
        ''' 
        The rule for setting elements of the cond_mask:
            1 : flowing in first half_layer, conditioning in second
            0 : conditioning in first half_layer, flowing in second
            2 : conditioning only (auxilary)
        '''
        cond_mask_A = np.array(cond_mask).flatten().astype(np.int32)
        ## in A:
        # 1 ->  1 = 1   -> flowing
        # 0 ->  0 < 0.5 -> conditioning
        # 2 ->  2 > 1.5 -> conditioning
        self.cond_mask_A = np.where((cond_mask_A>1.5)|(cond_mask_A<0.5),0,cond_mask_A)

        cond_mask_B = np.array(1 - cond_mask_A) 
        ## in B:
        # 1 ->  0 < 0.5 -> conditioning
        # 0 ->  1 = 1   -> flowing
        # 2 -> -1 < 0.5 -> conditioning
        self.cond_mask_B = np.where((cond_mask_B>1.5)|(cond_mask_B<0.5),0,cond_mask_B)

        self.layer_A = half_layer_class(
                        periodic_mask = periodic_mask,
                        cond_mask = self.cond_mask_A,
                        use_tfp = use_tfp,
                        n_bins = n_bins ,
                        min_bin_width = min_bin_width,
                        knot_slope_range = knot_slope_range ,
                        name = name,
                        **kwargs_for_given_half_layer_class,
                        )
        
        self.layer_B = half_layer_class(
                        periodic_mask = periodic_mask,
                        cond_mask = self.cond_mask_B,
                        use_tfp = use_tfp,
                        n_bins = n_bins ,
                        min_bin_width = min_bin_width,
                        knot_slope_range = knot_slope_range ,
                        name = name,
                        **kwargs_for_given_half_layer_class,
                        )
        
    def forward(self, x, dont_mask=False):
        ladJ = 0.0
        x, ladj = self.layer_A.forward_(x, dont_mask=dont_mask) ; ladJ += ladj
        x, ladj = self.layer_B.forward_(x, dont_mask=dont_mask) ; ladJ += ladj
        return x, ladJ

    def inverse(self, x, dont_mask=False):
        ladJ = 0.0
        x, ladj = self.layer_B.inverse_(x, dont_mask=dont_mask) ; ladJ += ladj
        x, ladj = self.layer_A.inverse_(x, dont_mask=dont_mask) ; ladJ += ladj
        return x, ladJ

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 