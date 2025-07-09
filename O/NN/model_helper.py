
''' model_helper.py
    f : no_nans_
    c : model_helper
    f : pool_
    f : tolBAR_
    f : get_FE_estimates_
    f : find_split_indices_
    f : FE_of_model_
    f : FE_of_model_curve_
    f : get_phi_ij_
    c : TRAINER 
    
    TODO: confirm TRAINER compatible with single molecule in vaccum in the same way as before
'''

from ..util_np import *
from ..plotting import *
from .util_tf import *

## ## ## ##

def no_nans_(grads:list):
    _grads = grads # [x for x in grads if x is not None]
    if tf.reduce_sum([tf.reduce_sum(1.0 - tf.cast(tf.math.is_finite(x),dtype=tf.float32)) for x in _grads]) > 0.0:
        return False
    else: return True

class model_helper:
    def __init__(self,):
        """the class which inherits these methods should have the following methods:
        
        sample_base_ : m   -> z         ; z : list or tensors
        ln_base_     : z   -> ln_p0     ; tensor which shape (m,1)

        forward_     : xyz -> z, ladJ   ; z : list, ladJ : tensor which shape (m,1)
        inverse_     : z   -> xyz, ladJ ; zyz : list, ladJ : tensor which shape (m,1)
        
        """

    def initialise_weights_(self,):
        # pass one sample through the model to initialise the sizes of all the trainable weight arrays
        self.all_parameters_trainable = True
        self.inverse(self.sample_base_(1))
        self.n_trainable_tensors = len(self.trainable_weights)

    def reset_optimiser_(self, optimiser_LR_decay:list):
        # reset to a fresh instance of Adam
        self.learning_rate, self.rate_decay = optimiser_LR_decay
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate,) #decay = self.rate_decay)
        
    def ln_model_for_step_ML_(self, inputs:list):
        # no numpy version of ln_model_
        outputs, ladJ = self.forward_(inputs)
        return outputs, self.ln_base_(outputs) + ladJ

    #@tf.function
    def step_ML_graph_(self, inputs_batch:list):
        # no numpy (tf compiled graph) train one training step (maximise likelihood on MD data)
        if self.all_parameters_trainable: variables = self.trainable_variables
        else: variables = self._trainable_variables
        with tf.GradientTape() as tape:
            ln_p = self.ln_model_for_step_ML_(inputs_batch)[-1]
            loss = - tf.reduce_mean(ln_p)
        grads = tape.gradient(loss, variables)
        if no_nans_(grads): 
            ''' tensor -> bool, is 99% of time without complaints from tensorflow '''
            # grads = [tf.clip_by_value(x,-200.,200.) for x in grads]
            self.optimizer.apply_gradients(zip(grads, variables))
            no_nans = True
        else: no_nans = False
        return loss, no_nans

    #@tf.function
    def forward_graph_(self, inputs:list):
        # no numpy (tf compiled graph) forward
        return self.forward_(inputs)
    #@tf.function
    def inverse_graph_(self, inputs:list):
        # no numpy (tf compiled graph) inverse
        return self.inverse_(inputs)
    
    def initialise_graphs_(self, re_initialise:bool=False):
        # faster than 'eager mode' when implemented in a way that does not involve retracing too many times
        # to reduce retracing:
        #     the shapes of inputs need to remain fixed
        #     ! indexing internal methods via input, seems to crease seperate graphs for each permutation
        #         (MBAR rewighting using a multimap model; unsolved effeciency issue)
        if re_initialise:
            del self.forward_graph_
            del self.inverse_graph_
            del self.step_ML_graph_
        else: pass
        self.forward_graph_ = tf.function(self.forward_graph_)
        self.inverse_graph_ = tf.function(self.inverse_graph_)
        self.step_ML_graph_ = tf.function(self.step_ML_graph_)

    ############ numpy interface:

    def forward(self, r):
        return self.forward_graph_(np2tf_(r))

    def inverse(self, inputs:list):
        return self.inverse_graph_(inputs)
    
    def step_ML(self, r, u:np.ndarray=None, batch_size:int=1000):
        # prepare numpy input and train one training step (maximise likelihood on MD data)
        # r : (m,n_atoms,3) : MD data
        inds_rand = np.random.choice(r.shape[0], batch_size, replace=False)
        AVMD_T_s, no_nans = self.step_ML_graph_(np2tf_(r[inds_rand]))
        if no_nans: pass 
        else: print('!! nan was found in gradient, this gradient was not used.')
        AVMD_T_s = AVMD_T_s.numpy()
        if u is None: return AVMD_T_s, AVMD_T_s
        else: 
            AVMD_T_f = u[inds_rand].mean() - AVMD_T_s
            return AVMD_T_f, AVMD_T_s

    def ln_model(self, r:list):
        # evalaute ln_q(r) : normalised log probability of the datapoint (r) according to the model
        outputs, ladJrz = self.forward(np2tf_(r))
        ln_q = self.ln_base_(outputs) + ladJrz
        return np.array(ln_q)
        
    def sample_model(self, m:int):
        # generate m random samples from the model
        z = self.sample_base_(m)
        r, ladJzr = self.inverse(z)
        ln_q = self.ln_base_(z) - ladJzr
        return np.array(r), np.array(ln_q)
    
    def print_model_size(self):
        # for information; to check size of the model (# trainable parameters)
        ws = self.trainable_weights
        self.n_params = sum([np.prod(ws[i].shape) if 0 not in ws[i].shape else np.sum(ws[i].shape) for i in range(len(ws))])
        print('There are',self.n_params,'trainable parameters in this model, among', len(ws),'trainable weights.' )
        shapes = [tuple(x.shape) for x in ws]
        shapes_str = ['W: '+str(shapes[i*2])+' b: '+str(shapes[2*i+1])+' ' for i in range(len(shapes)//2)]
        self.shapes_trainable_weights = [''.join([(' ' * (8 - len(y))) + y for y in [x.split(' ')  for x in shapes_str][i]]) for i in range(len(shapes)//2)]
        print('[To see dimensionalities of the trainable weights print(list(self.shapes_trainable_weights)).] ')

    def save_model(self, path_and_name : str):
        # saves initialisation variables for the model, and the model's weights
        # this includes a instance of ic_map (first initialisation argument)
        save_pickle_([self.init_args, self.trainable_variables], path_and_name)

    @staticmethod
    def _load_model_(path_and_name : str, class_of_the_model):
        ''' in each class_of_the_model have a method:
        @staticmethod
        def load_model(path_and_name : str):
            return class_of_the_model.load_model_(path_and_name, class_of_the_model)
        '''
        init_args, ws = load_pickle_(path_and_name)
        model = (lambda f, args : f(**args))(class_of_the_model, init_args)
        for i in range(len(ws)):
            model.trainable_variables[i].assign(ws[i])
        return model

    def test_inverse_(self, r, graph=True):
        # r : Cartesian coordinates of a random validation batch
        
        # this function tests invertibility error of the model in both directions
        # TODO: maybe add more useful statistics in the output

        r = np2tf_(r) 
        m = r.shape[0]

        if graph: f_ = self.forward  ; i_ = self.inverse
        else:     f_ = self.forward_ ; i_ = self.inverse_
        ##
        z, ladJrz   = f_(r)
        _r, ladJzr = i_(z)
        # absolute error in r coordinates when mapping validation batch forward and then back
        err_r_forward = np.abs(r - _r)
        # average and maximum of the above error, over the batch
        err_r_forward = [err_r_forward.mean(), err_r_forward.max()]
        # error in log volume when mapping validation batch forward and then back
        err_l_forward = np.array(ladJrz + ladJzr)
        # average, minimum and maximum, of the above error, over the batch
        err_l_forward = [err_l_forward.mean(), err_l_forward.min(), err_l_forward.max()]
        ##
        z = self.sample_base_(m)
        _r, ladJzr = i_(z)
        
        try: 
            _z, ladJrz = f_(_r)
            # absolute error in the z coordinates, when mapping samples from the base distribution backward and then forward
            err_r_backward = [np.abs(x - _x) for x,_x in zip(z,_z)]
            # average and maximum of the above error, over the batch
            err_r_backward = [[x.mean(), x.max()] for x in err_r_backward]
            # error in log volume, when mapping samples from the base distribution backward and then forward
            err_l_backward = np.array(ladJrz + ladJzr)
            # average, minimum and maximum, of the above error, over the batch
            err_l_backward = [err_l_backward.mean(), err_l_backward.min(), err_l_backward.max()]

        except:
            # should not be reached here
            err_r_backward = [None]
            err_l_backward = [None]

        # [2,3], [[2]*..,3] 
        return[err_r_forward, err_l_forward], [err_r_backward, err_l_backward]

    ###############

    def index_transferable_parameters_(self,):
        self.inds_transferable_parameters = []
        for i in range(self.n_trainable_tensors):
            if 'TRANSFERABLE' in self.trainable_variables[i].name:
                self.inds_transferable_parameters.append(i)
            else: pass
        self.init_args_to_match_CM = ['TRANSFERABLE', self.n_layers, self.DIM_connection]

    def save_transferable_parameters_(self, path_and_name):
        ws_CM = [self.trainable_variables[j] for j in self.inds_transferable_parameters]
        save_pickle_([self.init_args_to_match_CM, ws_CM], path_and_name)

    def load_transferable_parameters_(self, path_and_name):
        args_to_match, ws_CM = load_pickle_(path_and_name)
        matching = all([x==y for x,y in zip(args_to_match, self.init_args_to_match_CM)])
        try: assert matching 
        except: raise ImportError('!! the transferable parameters being loaded does not match this model')
        a = 0
        for j in self.inds_transferable_parameters:
            self.trainable_variables[j].assign(ws_CM[a])
            a +=1
        self.initialise_graphs_()

    def set_transferable_parameters_fixed_(self,):
        self.all_parameters_trainable = False
        self.inds_trainable_variables = list(set(np.arange(self.n_trainable_tensors)) - set(self.inds_transferable_parameters))
        self._trainable_variables = [self.trainable_variables[k] for k in self.inds_trainable_variables]
        self.reset_optimiser_(self.optimiser_LR_decay)
        self.initialise_graphs_(re_initialise=True)
        print('the transferable parameters were set: fixed')
        
    def set_transferable_parameters_trainable_(self,):
        self.all_parameters_trainable = True
        try: del self._trainable_variables
        except: pass
        self.inds_trainable_variables = np.arange(self.n_trainable_tensors).tolist()
        self.reset_optimiser_(self.optimiser_LR_decay)
        self.initialise_graphs_(re_initialise=True)
        print('trainable parameters of the transferable parameters were set: trainable')

    ###########

    def initialise(self, ):
        self.initialise_weights_()
        self.reset_optimiser_(self.optimiser_LR_decay)
        self.initialise_graphs_()
        self.index_transferable_parameters_()

########################################################################

def pool_(x, ws=None):
    ''' Output: weighted average of x '''
    if ws is None: return x.mean()
    else:          return (x*ws).sum() / ws.sum()

def tolBAR_(incuA, incuB,
            wsA=None, wsB=None,
            f_window_grain = [-60000.,60000.,10000],
            tol=0.00001,
            return_errs=False,
            ):
    
    ''' 
    This is only used as a double check. Final FE results are always solved using pymbar

    local implementation of 2-state BAR, finding f (via grid search) that best fits the BAR equality A = B below.
        'tol' is tolerance of the best fit (absolute error |A-B|)

    No standard error method is included here, therefore cannot be taken as a final result.

    Inputs as used in current work:
        f is absolute FE

        incuA : (m,1) shaped array = \phi(r) = u(r) + ln(p(r)) ; r ~ \mu
            \mu = \mu(r) = exp(-u(r)) / Z, is the MD distribution of NVT data with 
                unknown normalisation constant Z, but we have m ergodic samples r ~ \mu
                Want output f to be as close as possible to underlying FE -ln(Z).

        incuB : (m,1) shaped array = \phi(r) = u(r) + ln(p(r)) ; r ~ p
            p is any normalised distribution that is similar to \mu,
            Should be able to evaluate ln(p) exactly, and sample m ergodic samples from it r ~ p

        wsA : was the sampling from \mu (no bias)? YES: wsA = None, NO : wsA = weights to reweight the bias.
        wsB : is the sampling from p ergodic? Yes by default: wsB = None.
        f_window_grain : list [float, float, int] = [a ,b, grain]
            a = minimum f that may be valid in this system
            b = maximum f that may be valid in this system
            grain = how many values on a grid to try between a and b (probably does not make much difference given tol)
    Output:
        f : scalar : estimate of -ln(Z)

    '''
    def _BAR_(grid_f, incuA, incuB, wsA=None, wsB=None):
        sigma_ = lambda x : (1.0+np.exp(-x))**(-1)
        errs = []
        for f in grid_f:
            A = pool_(sigma_( (f - incuA) ), ws=wsA)
            B = pool_(sigma_( (incuB  - f) ), ws=wsB)
            errs.append(np.abs(A-B))
        errs = np.array(errs)
        ind_min = np.argmin(errs)
        f = grid_f[ind_min]
        if f in [grid_f[0], grid_f[-1]]: return f, False, ind_min, errs
        else:                            return f, True, ind_min, errs
            
    bar_ = lambda _grid : _BAR_(_grid, incuA=incuA, incuB=incuB, wsA=wsA, wsB=wsB)
    
    log_grids = [] ; log_errs = []
    a, b, grain = f_window_grain
    grid = np.linspace(a, b, grain) ; log_grids.append(grid)
    f, ok, ind_min, errs = bar_(grid) ; log_errs.append(errs)
    
    if not ok:
        print('!! warning: BAR_ : grid not adjusted properly')
        pass
    else:
        old_f = np.array(f+1000.)
        while np.abs(old_f-f) > tol:
            old_f = np.array(f)
            grid = np.linspace(grid[ind_min-1], grid[ind_min+1], grain) ; log_grids.append(grid)
            f, ok, ind_min, errs = bar_(grid) ; log_errs.append(errs)
            assert ok
    if return_errs:
        return f, [[x,y] for x,y in zip(log_grids, log_errs)], ok # [plt.plot(*x) for x in errs[1:2]]
    else:
        return f, ok

'''
def get_SE_(u_kn, N_k, f_k, fast=True):
    """ analytical standard error on FE estimates
        not using this, just using pymbar
    """
    def get_W_(u_kn, N_k, f_k):
        # u_kn : (K, N_tot) = 'Q'
        # N_k : (K,)        = 'Ns' 
        # f_k : (K,) # pymbar output MBAR(u_kn, N_k).compute_free_energy_differences()['Delta_f'][0] 
        K, N_tot = u_kn.shape
        N_k = np.array(N_k).reshape([K,1])
        f_k = np.array(f_k).reshape([K,1])
        #W = np.exp(f_k - u_kn) / np.sum(np.exp(f_k - u_kn + np.log(N_k)), axis=0, keepdims=True)
        logW = f_k - u_kn - sp.special.logsumexp(f_k - u_kn, b=N_k, axis=0)
        #print(  ((W*N_k).sum(0)/N_k.sum()).sum() , W.sum(1)  ) # 1, [1,...,1] ; should be.
        W = np.exp(logW)
        return W #, logW # (K, N_tot)
    W = get_W_(u_kn, N_k, f_k).T # (N_tot, K)
    if not fast:
        K, N_tot = u_kn.shape
        I_N_tot = np.eye(N_tot)
        I_K = np.eye(K)
        effective_normalisation = np.linalg.pinv(I_N_tot - W.dot(I_K*N_k.flatten()).dot(W.T))
        C = W.T.dot(effective_normalisation).dot(W) # Theta_hat_{ij} in the paper arXiv:0801.1426v3
    else:
        from pymbar import MBAR
        C = MBAR(u_kn, N_k, solver_protocol="robust")._computeAsymptoticCovarianceMatrix(W, N_k)
    Cii = np.diag(C)
    var = (Cii[np.newaxis,:] + Cii[:,np.newaxis]) - 2.0*C
    SE = var**0.5
    return SE
'''

def get_FE_estimates_(
                      model,

                      r_training,
                      r_validation,

                      name_save_BAR_inputs : str,

                      u_training,
                      u_validation,

                      u_function_ = None,

                      w_training = None,
                      w_validation = None,

                      crystal_index : int = 0,

                      evaluation_batch_size = 5000,
                      shuffle = True,

                      test_inverse = False,
                      evaluate_on_training_data = False,
                      save_generated_configurations_anyway = False
                      ):
    ''' FE evalaution from the model:

    During training every so often this function is ran on each macrostate being trained.

    The overall cost of this function therefore can be high, but 
    can this be reduced to just a validation loss evalaution (does not require a FF).
    If the FF is cheap to evaluate, and want FEs as effeciently as possible:
        use only the top 7 to 9 arguments, with other arguments left as default.

    Inputs (arguments):
        model         : instace of PGM model with methods: ln_model, sample_model, and test_inverse_
        r_training    : (m_T,N,3) : Cartesian coordiantes of the training data (dataset contains m_T configurations)
        r_validation  : (m_V,N,3) : Cartesian coordiantes of the validation data (dataset contains m_V configurations)
        name_save_BAR_inputs : path and name where to save pymbar inputs for BAR 
            (saved with suffix added at the end of the name: '_V' ; validation, '_T' ; training)
        u_training    : (m_T,1)   : potential energies of the training data
        u_validation  : (m_T,1)   : potential energies of the validation data
        u_function_   : potential energy function of the current macrostate
        w_training    : (m,1)     : weights of training data (only if the MD data sampled from a biased ensemble)
        w_validation  : (m,1)     : weights of validation data (only if the MD data sampled from a biased ensemble)
        crystal_index : in multimap model, index of the current macrostate. (! placeholder is 0, no warnings)
        evaluation_batch_size : number of samples to include for all types of FE estiamates
        shuffle       : default True, otherwise evaluating only on the first evaluation_batch_size configurations
        test_inverse  : extra cost if True, but useful when testing a model (allows plottin inversion accuracy during training)
        evaluate_on_training_data : extra cost if True, default False (not necesary for general use)
        save_generated_configurations_anyway : optional, default is False (saving only the small energy arrays, saving disk space)
    Outputs:
        running_estimates : abbreviations defined in doi/10.1021/acs.jctc.4c00520 (*P1)
            all elements of this output can be undestood by refering to the short lines they were defined
        inv_test_result   : [None] if test_inverse False, otherwise depends the outputs from model.test_inverse_

    '''

    m_T = len(r_training)
    m_V = len(r_validation)
    if evaluation_batch_size > m_T:
        print('!! get_FE_estimates_ : evaluation_batch_size (',evaluation_batch_size,') > training_dataset_size (',m_T,')')
    else: pass
    if evaluation_batch_size > m_V:
        print('!! get_FE_estimates_ : evaluation_batch_size (',evaluation_batch_size,') > training_dataset_size (',m_V,')')
    else: pass

    if shuffle:
        inds_rand_T = np.random.choice(m_T, evaluation_batch_size, replace=False)
        r_T = np.array(r_training[inds_rand_T])
        u_T = np.array(u_training[inds_rand_T])

        inds_rand_V = np.random.choice(m_V, evaluation_batch_size, replace=False)
        r_V = np.array(r_validation[inds_rand_V])
        u_V = np.array(u_validation[inds_rand_V])

    else:
        r_T = np.array(r_training[:evaluation_batch_size])
        u_T = np.array(u_training[:evaluation_batch_size])

        r_V = np.array(r_validation[:evaluation_batch_size])
        u_V = np.array(u_validation[:evaluation_batch_size])

    # keeping this incase the MD sampling was accelerated (Alex, MACE)
    # since *P1 this was not tested, but expected work well if there are no errors below.
    if w_training   is None: w_T = None
    else: 
        if shuffle:          w_T = np.array(w_training[inds_rand_T])
        else:                w_T = np.array(w_training[:evaluation_batch_size])
    if w_validation is None: w_V = None
    else:
        if shuffle:          w_V = np.array(w_validation[inds_rand_V])
        else:                w_V = np.array(w_validation[:evaluation_batch_size])

    negS_V = model.ln_model(r_V, crystal_index=crystal_index)
    offset = np.median(u_V + negS_V) # any offset ~ f_T or f_V
    f_V = u_V + negS_V - offset

    if evaluate_on_training_data:
        negS_T = model.ln_model(r_T, crystal_index=crystal_index)
        f_T = u_T + negS_T - offset

        AVMD_T = pool_(f_T, ws=w_T) + offset
        EXPMD_T = np.log(pool_(np.exp(f_T), ws=w_T)) + offset
    else:
        AVMD_T = 0.0
        EXPMD_T = 0.0
        
    AVMD_V = pool_(f_V, ws=w_V) + offset
    EXPMD_V = np.log(pool_(np.exp(f_V), ws=w_V)) + offset

    r_BG, negS_BG = model.sample_model(evaluation_batch_size, crystal_index=crystal_index)

    if u_function_ is not None:
        u_BG = u_function_(r_BG)
        f_BG = u_BG + negS_BG - offset

        #
        AVBG   = pool_(f_BG, ws=None) + offset
        EXPBG  = - np.log(pool_(np.exp(-f_BG), ws=None)) + offset
        #EXPBG = - sp.special.logsumexp(-(f_BG+offset))
        '''
        _C = -sp.special.logsumexp(-(f_BG+offset))
        EXPBG  = -np.log( pool_( np.exp(-(f_BG+offset-_C))  , ws=None) ) + _C
        '''
        # 

        ## miscellaneous info: reweighted average potential energy using EXP_BG
        AV_u_EXPBG = pool_(u_BG, ws=np.exp(-f_BG))
        ##
        
        if evaluate_on_training_data:
            BAR_T, ok = tolBAR_(incuA=f_T + offset, incuB=f_BG + offset, wsA=w_T)
        else:
            BAR_T = 0.0
            
        BAR_V, ok = tolBAR_(incuA=f_V + offset, incuB=f_BG + offset, wsA=w_V)
        if not ok: AVMD_V = -1e20
        else: pass

        ## miscellaneous info: reweighted average potential energy using BAR_V: [TODO: double check]
        sigma_ = lambda x : (1.0+np.exp(-x))**(-1)
        AV_u_BAR_V = pool_(u_BG*sigma_(BAR_V - f_BG-offset), ws=None) + pool_(u_V*sigma_(BAR_V - f_V-offset), ws=w_V)
        ##

        if name_save_BAR_inputs is not None:
            if evaluate_on_training_data:
                first_row = np.concatenate([u_T, u_BG], axis=0)
                second_row = np.concatenate([u_T-(f_T+offset), u_BG-(f_BG+offset)], axis=0)
                mBAR_inputs_T = np.stack([first_row, second_row],axis=0)[:,:,0]
                save_pickle_([mBAR_inputs_T, w_T], name_save_BAR_inputs + '_T',)
            else: pass

            first_row = np.concatenate([u_V, u_BG], axis=0)
            second_row = np.concatenate([u_V-(f_V+offset), u_BG-(f_BG+offset)], axis=0)
            mBAR_inputs_V = np.stack([first_row, second_row],axis=0)[:,:,0]
            save_pickle_([mBAR_inputs_V, w_V], name_save_BAR_inputs + '_V')

            if save_generated_configurations_anyway:
                save_pickle_([r_BG, negS_BG, u_V, negS_V, w_V], name_save_BAR_inputs + '_r_&_ln(q(r))',)
            else: pass 
        else: pass

        u_BG_mean = u_BG.mean()
    else:
        AVBG = 0.0
        EXPBG = 0.0
        AV_u_EXPBG = 0.0
        BAR_T = 0.0
        BAR_V = 0.0
        AV_u_BAR_V = 0.0
        u_BG_mean = 0.0

        if name_save_BAR_inputs is not None:
                save_pickle_([r_BG, negS_BG, u_V, negS_V, w_V], name_save_BAR_inputs + '_r_&_ln(q(r))',)
        else: pass

    if test_inverse: inv_test_result = [model.test_inverse_(r_V, graph=True)]
    else: inv_test_result = [None]

    running_estimates = np.array([  AVMD_T,      # 0
                                    AVMD_V,      # 1
                                    AVBG,        # 2
                                    EXPMD_T,     # 3
                                    EXPMD_V,     # 4
                                    EXPBG,       # 5
                                    BAR_T,       # 6
                                    BAR_V,       # 7
                                    u_T.mean(),  # 8 
                                    u_V.mean(),  # 9
                                    u_BG_mean,   # 10 
                                    AV_u_BAR_V,  # 11
                                    AV_u_EXPBG,  # 12
                                ])
    return running_estimates, inv_test_result

def find_split_indices_(u, split_where:int, tol=0.00001):
    ''' training : validation split where both sets have same average potential energy within tol
    Inputs:
        u : (m,1) array of potential energies during MD sampling
        split_where : int, how many samples wanted in the training set
        tol : how similar should average energy of training set be to the average energy of the validation set
    Outputs:
        inds_rand or None: run multiple times until returns not None, or increase tol
            inds_rand : permutation of u (i.e., u[inds_rand]),
            where the first split_where points/samples are belong to the training set,
            and the rest of the array (i.e., u[inds_rand][split_where:]) validation set.
            Use this permuation on any other array relevant for training: r, u, w, b
    '''
    u = np.array(u)
    n = u.shape[0]
    target = u.mean()
    for i in range(1000):
        inds_rand = np.random.choice(n,n,replace=False)
        randomised = np.array(u[inds_rand])
        if np.abs(randomised[:split_where].mean() - target) < tol and np.abs(randomised[split_where:].mean() - target) < tol:
            print('found !')
            return inds_rand
        else: pass
    print('! not found')
    return None

def FE_of_model_(AVMD_V, BAR_V):
    ''' weighted average of raw BAR_V FE estimates 
        the weights = np.exp(AVMD_V) ; higher is better (lower validation error)
            motivation for heuristic : since AVMD_V is a type of FE estimate, exponentiating it a type of weight >= 0
        REF: doi/10.1021/acs.jctc.4c01612 (P2, defined also in P3)

        Inputs:
            AVMD_V : (n_evaluation_batches, 1) array of raw FE estimates based on AVMD_V
            BAR_V  : (n_evaluation_batches, 1) array of raw estimates based on BAR_V
        Output:
            av_BAR_V : scalar value : averaged BAR_V FE from the training run
    '''
    import scipy as sp
    ws = np.exp(AVMD_V - sp.special.logsumexp(AVMD_V))
    ws /= ws.sum()
    av_BAR_V = (BAR_V*ws).sum()
    return av_BAR_V

def FE_of_model_curve_(AVMD_V, BAR_V):
    ''' cumulative weighted average using FE_of_model_
        Inputs: same as in FE_of_model_
        Output: (n_evaluation_batches, ) array of *averaged BAR_V FEs from the training run
            *(averaged up to each evaluation batch)
    '''
    n = len(AVMD_V) # n_evaluation_batches
    assert n == len(BAR_V)
    output = np.zeros([n])
    for _to in range(1,n+1):
        output[_to-1] = FE_of_model_(AVMD_V[:_to], BAR_V[:_to])
    return output

def get_phi_ij_(model,
                list_r,
                list_potential_energy_functions,
                evalation_sample_size = 5000,
                shuffle = True):
    
    ''' composition of maps r^{[i]} -> z -> r^{[j]} ; states i,j = 0,...,n_states

        n_states = len(list_r) = len(list_potential_energy_functions)

    Inputs:
        model : instace of PGM model with methods: forward, inverse
            both methods need to have crystal_index : int as argument
        list_r : list of MD datasets ergodically sampled in n_states different states
        list_potential_energy_functions : list of corresponding potential energy functions
        evalation_sample_size : number of samples to evalaute (here this is the same number in each direction)
        shuffle : default True, otherwise evaluating only on the first evaluation_batch_size configurations
    Outputs:
        phi_ij : MBAR input for pymbar to compute FE differences between all pairs of states

    '''
    n_states = len(list_r)
    phi_ij = np.zeros([n_states,n_states,evalation_sample_size])
    
    for i in range(n_states):

        ri = list_r[i]

        if shuffle: ri = np.array(ri[np.random.choice(len(ri), evalation_sample_size,replace=False)])
        else:       ri = np.array(ri[:evalation_sample_size])

        #z, ladJi = model.forward_(np2tf_(ri), crystal_index=i # graph mode False
        z, ladJi = model.forward(np2tf_(ri), crystal_index=i) # graph mode True
        for j in range(n_states):
            
            #rj, ladJj = model.inverse_(z, crystal_index=j) # graph mode False
            rj, ladJj = model.inverse(z, crystal_index=j) # graph mode True

            phi_ij[i,j] = list_potential_energy_functions[j](tf2np_(rj))[:,0] - tf2np_(ladJi + ladJj)[:,0]
            
    phi_ij = np.einsum('ijk->jik',phi_ij)

    return phi_ij # (n_states,n_states,n)

class TRAINER:
    def __init__( self,
                  model,
                  max_training_batches : int = 50000,
                  n_batches_between_evaluations : int = 50,
                  running_in_notebook : bool = False,
                 ):
        ''' takes the model + MD data, to train, and evaluate the model during training
            Inputs:
                model : instace of PGM model with methods: 
                    step_ML, forward, inverse, ln_model, sample_model, and test_inverse_
                max_training_batches : any large number 
                    to allocate long enough arrays into which output numbers are stored during training
                n_batches_between_evaluations : evaluation stride
                    number of training batches between which model is evaluated for FE estimate(s)
                running_in_notebook : True is better if running the training in a jupyter notebook
            Output:
                None
                    NB: if running_in_notebook, the cell in which TRAINER is initialised (here)
                    is the cell where progress of training, and FE estiamte(s) printed using text
        '''
        self.model = model
        self.n_states = self.model.n_maps
        self.n_mol = self.model.n_mol
        
        ##

        self.n_main_estimates = 13
        self.max_training_batches = max_training_batches
        self.n_batches_between_evaluations = n_batches_between_evaluations

        ##

        self._evaluation_grid = np.arange(self.n_batches_between_evaluations,
                                          self.max_training_batches+self.n_batches_between_evaluations,
                                          self.n_batches_between_evaluations)-1
        
        self._AVMD_T_f = np.zeros([self.max_training_batches, self.n_states])

        self._estimates = np.zeros([self.n_states,len(self._evaluation_grid),self.n_main_estimates])
        self.count = 0
        self.count_strided = 0

        self.FEs = dict(zip(np.arange(self.n_states),[[0.0]]*self.n_states))
        self.SEs = dict(zip(np.arange(self.n_states),[[0.0]]*self.n_states))
        self.AVMD_Vs = dict(zip(np.arange(self.n_states),[[0.0]]*self.n_states))
        self.inv_test_results = dict(zip(np.arange(self.n_states),[[]]*self.n_states))

        ## verbose (minimum):
        self.running_in_notebook = running_in_notebook
        if self.running_in_notebook: self.dh = display('', display_id=True)
        else : pass

        self.training_time = 0.0 # add up time for all batches in minutes

        print('')

    def print_(self, text):
        if not self.running_in_notebook: print(text, end='\r')
        else: self.dh.update(text)

    def train(self,
                n_batches : int, # < max_training_batches 

                # needed and always available:
                list_r_training : list,
                list_r_validation : list,

                list_u_training : list,
                list_u_validation : list,

                # not needed if not relevant:
                list_w_training : list = None,
                list_w_validation : list = None,

                # if the FF is cheap can use, BAR_V estiamtes during training will be saved.
                list_potential_energy_functions : list = None,

                # evalaution cost vs statistical significance of gradient of the loss:
                training_batch_size = 1000,
                # evalaution cost vs variance of FE estimates (inc. standard error if name_save_BAR_inputs ):
                evaluation_batch_size = 5000,

                # always True, unless just quickly checking if the training works at all:
                evaluate_main = True,
                # pymbar will run later, safe inputs during training:
                name_save_BAR_inputs = None,
                name_save_mBAR_inputs = None,

                # statistical significance vs model quality:
                shuffle = True,
                # verbose: y_axis width of running plot if not None, or running not in notebook : None.
                f_halfwindow_visualisation = 0.5, # f_latt / kT if verbose_divided_by_n_mol, else f_crys / kT
                verbose = True, # only for plotting, while the text is printed every batch anyway
                verbose_divided_by_n_mol = True, # if True, both text and plots show quantities that were standardised by the number of molecules
                
                # evalaution cost:
                evaluate_on_training_data = False, # not needed.
                test_inverse = False, # not needed if model ok.
                # disk space:
                save_generated_configurations_anyway = False
              ):
        if verbose_divided_by_n_mol: n_mol = int(self.n_mol)
        else: n_mol = 1
        if type(f_halfwindow_visualisation) in [int, float]: f_halfwindow_visualisation = [f_halfwindow_visualisation]*2
        else: pass

        if list_w_training is None: list_w_training = [None]*self.n_states
        else: assert len(list_w_training) == self.n_states
        if list_w_validation is None: list_w_validation = [None]*self.n_states
        else: assert len(list_w_validation) == self.n_states

        assert len(list_r_training) == len(list_r_validation) == len(list_u_training) == len(list_u_validation) == self.n_states

        if list_potential_energy_functions is None:
            list_potential_energy_functions = [None]*self.n_states
            if name_save_BAR_inputs is not None:
                print('! no potential energy function(s) provided, the BAR inputs will be saved as configurations to evaluate later')
            else: pass
            if name_save_mBAR_inputs is not None:
                print('potential energy functions not provided')
                print('!! MBAR in this version cannot be ran without potential energy functions evaluated during training')
                name_save_mBAR_inputs = None
            else: pass
        else:
            assert len(list_potential_energy_functions) == self.n_states

        for i in range(n_batches):
            t0 = time.time()
            for k in range(self.n_states):
                self._AVMD_T_f[self.count, k] = self.model.step_ML(
                                                    r = list_r_training[k],
                                                    crystal_index = k,
                                                    u = list_u_training[k],
                                                    batch_size = training_batch_size,
                                                    )
            self.print_(str(np.round(self.training_time, 2))                 # training_time in minutes
                        + ' '        + str(i)                                # index training batch, counting since last call of self.train
                        + ' '        + str(self.count)                       # index overall training batch, counting since TRAINER was initialised (even if self.train was paused)
                        + ' '        + str(self.count_strided)               # index overall evaluation batch, counting since TRAINER was initialised (even if self.train was paused)
                        + ' AVMD_T:' + str(self._AVMD_T_f[self.count]/n_mol) # new every training batch
                        + ' AVMD_V:' + str([self.AVMD_Vs[k][-1]/n_mol for k in range(self.n_states)]) # new every evaluation batch (list of states)
                        + ' ||'                                                                       # related to running BAR_V estimates after ||
                        + ' FE:'     + str([self.FEs[k][-1]/n_mol for k in range(self.n_states)])     # running BAR_V absolute FE, new every evalaution batch (list of states)
                        + ' SD:'     + str([self.SEs[k][-1]/n_mol for k in range(self.n_states)]),    # running BAR_V absolute SE, new every evalaution batch (list of states)
                        )
            if self.count in self._evaluation_grid:
                if name_save_mBAR_inputs is None: pass
                else:
                    name_mBAR = name_save_mBAR_inputs+'_mBAR_input_'+str(self.count_strided)
                    if evaluate_on_training_data:
                        PHIij_T = get_phi_ij_(model = self.model,
                                              list_r = list_r_training,
                                              list_potential_energy_functions = list_potential_energy_functions,
                                              evalation_sample_size = evaluation_batch_size,
                                              shuffle = shuffle)
                        save_pickle_(PHIij_T, name_mBAR + '_T')
                    else: pass
                    PHIij_V = get_phi_ij_(model = self.model,
                                          list_r = list_r_validation,
                                          list_potential_energy_functions = list_potential_energy_functions,
                                          evalation_sample_size = evaluation_batch_size,
                                          shuffle = shuffle)
                    save_pickle_(PHIij_V, name_mBAR + '_V')

                if evaluate_main:
                    for k in range(self.n_states):
                        if name_save_BAR_inputs is None:
                            name_BAR = None
                        else: 
                            name_BAR = name_save_BAR_inputs+'_BAR_input_'+str(self.count_strided)+'_state'+str(k)+'_'
                        estimates_kc, inv_test = get_FE_estimates_(
                            
                                                    model = self.model,

                                                    r_training = list_r_training[k],
                                                    r_validation = list_r_validation[k],

                                                    name_save_BAR_inputs = name_BAR,

                                                    u_training = list_u_training[k],
                                                    u_validation = list_u_validation[k],

                                                    u_function_ = list_potential_energy_functions[k],

                                                    w_training = list_w_training[k],
                                                    w_validation = list_w_validation[k],

                                                    crystal_index = k,

                                                    evaluation_batch_size = evaluation_batch_size,
                                                    shuffle = shuffle,

                                                    test_inverse = test_inverse,
                                                    evaluate_on_training_data = evaluate_on_training_data,
                                                    save_generated_configurations_anyway = save_generated_configurations_anyway,

                                                                    )
                        
                        self._estimates[k,self.count_strided] = estimates_kc
                        if test_inverse:
                            self.inv_test_results[k].append([self.count_strided,k]+inv_test)
                        else: pass
                else: pass

                self.count_strided += 1

                if evaluate_main:
                    for k in range(self.n_states):
                        self.AVMD_Vs[k].append(self.estimates[k,-1,1])
                        self.FEs[k] = FE_of_model_curve_(self.estimates[k,:,1], self.estimates[k,:,7])
                        self.SEs[k] = FE_of_model_curve_(self.estimates[k,:,1], (self.estimates[k,:,7]-self.FEs[k])**2)**0.5
                else: pass

            self.count += 1

            '''
            plotting (if verbose AND running in notebook)

            three types of plots depending on setting used
            the plots are labeled with a title

            TODO: test if all three options work

            '''

            if self.count_strided>0 and self.running_in_notebook and verbose and evaluate_main and list_potential_energy_functions[0] is not None:
                def plot_this_():
                    fig,ax = plt.subplots(1,2, figsize=(10,5))

                    show_AVMD_T_f = np.array(self.AVMD_T_f)/n_mol
                    x_ax_AVMD_T_f = np.arange(len(show_AVMD_T_f))

                    est = np.array(self.estimates)/n_mol
                    x_ax = np.array(self.evaluation_grid)

                    av_FEs = [np.array(self.FEs[k])/n_mol for k in range(self.n_states)]
                    av_SEs = [np.array(self.SEs[k])/n_mol for k in range(self.n_states)]

                    y_range_0 = [min([est[k,-1,1] for k in range(self.n_states)])-f_halfwindow_visualisation[0],
                                 max([est[k,-1,1] for k in range(self.n_states)])+f_halfwindow_visualisation[0]]
                    y_range_1 = [min([x[-1] for x in av_FEs])-f_halfwindow_visualisation[1],
                                 max([x[-1] for x in av_FEs])+f_halfwindow_visualisation[1]]

                    for k in range(self.n_states):
                        [ax[0].plot(x_ax, est[k,:,x], color=color, alpha=alpha, linewidth=1) for x,color,alpha in zip([0,1],['black','black'],[0.5,1.0])]

                    ax[0].plot(x_ax_AVMD_T_f, show_AVMD_T_f, color='black', alpha=0.25, linewidth=0.5)

                    for k in range(self.n_states):
                        [ax[1].plot(x_ax, est[k,:,x], color=color, zorder=0, linewidth=1) for x, color in zip([7],['black'])]
                        ax[1].fill_between(x_ax, av_FEs[k]-av_SEs[k],av_FEs[k]+av_SEs[k], alpha=0.5, color='C'+str(k), linewidth=1, zorder=1)
                        ax[1].plot(x_ax, av_FEs[k], color='C'+str(k), linewidth=2, label='state'+str(k))

                    ax[0].set_title('AVMD_T (grey), AVMD_V (black)')
                    ax[1].set_title('BAR_V (colours : states)')
                    ax[0].set_ylim(*y_range_0)
                    ax[1].set_ylim(*y_range_1)
                    plt.tight_layout()
                    plt.show()

                clear_output(wait=True)
                plot_this_()

            elif self.count_strided>0 and self.running_in_notebook and verbose and evaluate_main and list_potential_energy_functions[0] is None:
                plots = [[plot, color] for plot, color in zip([1,4],['black','blue'])]

                def plot_this_():
                    fig,ax = plt.subplots(1,1, figsize=(5,5))

                    show_AVMD_T_f = np.array(self.AVMD_T_f)/n_mol
                    x_ax_AVMD_T_f = np.arange(len(show_AVMD_T_f))

                    est = np.array(self.estimates)/n_mol
                    x_ax = np.array(self.evaluation_grid)

                    y_range_0 = [min([est[k,-1,1] for k in range(self.n_states)])-f_halfwindow_visualisation[0],
                                 max([est[k,-1,1] for k in range(self.n_states)])+f_halfwindow_visualisation[0]]

                    for k in range(self.n_states):
                        [ax.plot(x_ax, est[k,:,x], color=color, alpha=alpha, linewidth=1) for x,color,alpha in zip([0,1],['black','black'],[0.5,1.0])]

                    ax.plot(x_ax_AVMD_T_f, show_AVMD_T_f, color='black', alpha=0.25, linewidth=0.5)

                    ax.set_title('AVMD_T (grey), AVMD_V (black)')
                    ax.set_ylim(*y_range_0)
                    plt.tight_layout()
                    plt.show()

                clear_output(wait=True)
                plot_this_()

            elif self.running_in_notebook and verbose:

                def plot_this_():
                    fig,ax = plt.subplots(1,1, figsize=(5,5))

                    show_AVMD_T_f = np.array(self.AVMD_T_f)/n_mol
                    x_ax_AVMD_T_f = np.arange(len(show_AVMD_T_f))

                    y_range_0 = [min([show_AVMD_T_f[-1,k] for k in range(self.n_states)])-f_halfwindow_visualisation[0],
                                 max([show_AVMD_T_f[-1,k] for k in range(self.n_states)])+f_halfwindow_visualisation[0]]

                    ax.plot(x_ax_AVMD_T_f, show_AVMD_T_f, color='black', alpha=0.25, linewidth=0.5)

                    ax.set_title('AVMD_T (grey)')
                    ax.set_ylim(*y_range_0)
                    plt.tight_layout()
                    plt.show()

                clear_output(wait=True)
                plot_this_()

            else: 
                # nothing plotted because verbose False and/or not in JN where this can be plotted
                pass
            
            t1 = time.time()
            self.training_time += (t1 - t0)/60.0

    @property
    def estimates(self,):
        return np.array(self._estimates[:,:self.count_strided])
    
    @property
    def evaluation_grid(self,):
        return np.array(self._evaluation_grid[:self.count_strided])
    
    @property
    def AVMD_T_f(self,):
        return np.array(self._AVMD_T_f[:self.count])
    
    def save_the_above_(self, name : str):
        save_pickle_([
                      self.FEs,
                      self.SEs,
                      self.estimates,
                      self.evaluation_grid,
                      self.AVMD_T_f,
                      self.training_time,
                      ], name)
        
    def save_inv_test_results_(self, name:str):
        save_pickle_(self.inv_test_results, name)

## ##

def plot_inv_test_res_(inv_test_result, mean_range=[True,True], forward_inverse=[True,True], plot_during_training=True):
    if mean_range[0]:
        if forward_inverse[0]:
            # [2] results
            # [0] forward (r->z->r)
            # [1] ladJ
            # [0] average
            show = np.array([x[2][0][1][0] for x in inv_test_result])
            if plot_during_training:
                plt.plot(show, color='red')
            print('F MEAN : forward (r->z->r) : mean ladJ error [averages of (first, last) 10 batches]:               ',show[:10].mean().round(3), show[-10:].mean().round(3))
        else: pass

        if forward_inverse[1]:
            # [2] results
            # [1] inverse (z->r->z)
            # [1] ladJ
            # [0] average
            show = np.array([x[2][1][1][0] for x in inv_test_result])
            print('        I MEAN : inverse (z->r->z) : mean ladJ error [averages of (first, last) 10 batches]:                    ',show[:10].mean().round(3), show[-10:].mean().round(3))
            if plot_during_training:
                plt.plot(show, color='black')
        else: pass
    else: pass

    if mean_range[1]:
        if forward_inverse[0]:
            show = np.array([x[2][0][1][1] for x in inv_test_result])
            # [2] results
            # [0] forward (r->z->r)
            # [1] ladJ
            # [1] minimum
            if plot_during_training:
                plt.plot(show, linestyle='dotted', color='red')
            print('F MIN  : forward (r->z->r) : minimum ladJ deviation [averages of (first, last) 10 batches]:       ',show[:10].mean().round(3), show[-10:].mean().round(3))

            show = np.array([x[2][0][1][2] for x in inv_test_result])
            if plot_during_training:
                plt.plot(show, linestyle='dotted', color='red')
            print('F MAX  : forward (r->z->r) : Maxiumum ladJ deviation [averages of (first, last) 10 batches]:       ',show[:10].mean().round(3), show[-10:].mean().round(3))
            
        else: pass
        if forward_inverse[1]:
            show = np.array([x[2][1][1][1] for x in inv_test_result])
            # [2] results
            # [1] inverse (z->r->z)
            # [1] ladJ
            # [1] minimum
            if plot_during_training:
                plt.plot(show, linestyle='dotted', color='black')
            print('        I MIN  : inverse (z->r->z) : minimum ladJ deviation [averages of (first, last) 10 batches]:             ',show[:10].mean().round(3), show[-10:].mean().round(3))
            
            show = np.array([x[2][1][1][2] for x in inv_test_result])
            if plot_during_training:
                plt.plot(show, linestyle='dotted', color='black')
            print('        I MAX  : inverse (z->r->z) : Maxiumum ladJ deviation [averages of (first, last) 10 batches]:             ',show[:10].mean().round(3), show[-10:].mean().round(3))
            
        else: pass
    else: pass
