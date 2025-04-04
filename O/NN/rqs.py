
''' rqs.py
    f : cast_64_
    f : cast_32_
    f : clip_by_value_preserve_gradient
    f : get_grid_
    f : softplus_
    f : soft_cap_
    f : softplus_with_a_softcap_
    f : normalize_knot_slopes_
    f : rqs_bin_
    f : rqs_
    f : shift_
    f : periodic_rqs_
    f : test_periodic_rqs_
    f : test_periodic_rqs_parallel_
    f : bin_positons_
    f : knot_slopes_
    f : rqs_tfp_
    f : periodic_rqs_tfp_
    f : test_periodic_rqs_tfp_
    f : test_periodic_rqs_tfp_parallel_
'''

import tensorflow as tf
import warnings

'''
docstrings not updated since the ice paper (P2)
'''

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# 
    # The ice paper was done using float32.

    # The float64 (current) is included because
    # it reduces the inversion error (near any knots), 
    # there is no significant cost increase from using better resolution

    # flaot64 seems essential for a better inversion accuracy 
#
    # about rqs_(...,identity_BCs=False):
    #   For non-periodic variables, edge slopes
    #   should be 1.0 (i.e., identity_BCs=True). 
    #   This was not the case in the ice paper,
    #   and is another source of inversion error.
    #   This was, and it, kept because with only 4 or 5 bins, 
    #   having all slopes trainable helps expressivity.

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

def cast_64_(x):
    return tf.cast(x,dtype=tf.float64) # can change to tf.float32 here, to recover old behaviour (e.g., error near any knots when slopes on grid change suddently)
def cast_32_(x):
    return tf.cast(x,dtype=tf.float32)

def clip_by_value_preserve_gradient(t, clip_value_min, clip_value_max,
                                    name=None):
  ''' copy from tensorflow_probability.math.clip_by_value_preserve_gradient '''
  with tf.name_scope(name or 'clip_by_value_preserve_gradient'):
    t = tf.convert_to_tensor(t, name='t')
    clip_t = tf.clip_by_value(t, clip_value_min, clip_value_max)
    return t + tf.stop_gradient(clip_t - t)

def get_grid_(MLP_output,
              dim,
              n_bins,
              min_bin_w, 
              interval,
              shape_parallel,
             ):
    a, b = interval ; L = b - a
    w = tf.reshape(MLP_output, [-1] + shape_parallel + [dim, n_bins]) # (..., dim, n_bins)
    w = tf.nn.softmax(w, axis=-1)                                     # real -> [0,1]
    w = min_bin_w + (1.0 - min_bin_w * n_bins) * w
    c_w = tf.math.cumsum(w, axis=-1)                                  # (..., dim, n_bins)
    c_w = tf.concat([tf.zeros_like(c_w[...,:1]),c_w],axis=-1)         # pad with zero on the left side of last axis
    c_w = L * c_w + a # [0,1] -> interval of chocice (these are bin edge positions in ascending order, i.e., a grid)
    return c_w # (m, dim, n_bins+1)

def softplus_(x):
    return tf.math.log(1.0 + tf.exp(x))

def soft_cap_(x, a, s):
    a = cast_64_(a)
    s = cast_64_(s)
    exp_a = tf.exp(a)
    c = s*(1.0+exp_a)/exp_a
    return s*tf.math.log( (x/c) + 1.0 - (a/c) ) + tf.math.log(1.0 + exp_a)

def softplus_with_a_softcap_(x, a=9.0, s=0.2):
    return tf.where(x<=a, softplus_(x), soft_cap_(x, a=a, s=s))

def normalize_knot_slopes_(x, # elementwise
                           knot_slope_range, # list
                           ):
    ''' 
    in this .py file, knot slopes that are learned cannot be too high or too low.
    This is a filter to allow this:
    '''
    min_knot_slope, max_knot_slope = knot_slope_range 
    return softplus_with_a_softcap_(x, a = max_knot_slope) + min_knot_slope

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

ZERO_64 = cast_64_(0.0)
def rqs_bin_(x, xA, xB, yA, yB, sA=1.0, sB=1.0, forward=True):
    ''' The main function! [REF : arXiv:1906.04032]
    Inputs:
        x : 1D input variable 
        xA : x coordianate of the lower knot
        xB : x coordianate of the upper knot
        yA : y coordianate of the lower knot
        yB : y coordianate of the upper knot
        sA : slope at lower knot 
        sB : slope at upper knot
        forward : going from x->y is forward, else y->x (inverting : forward=False)
    Ouputs:
        y : 1D output variable
        ladJ : ln( dy/dx ) : log of gradient of output with respect to input
    '''
    d = (yB-yA) / (xB-xA)
    if forward:
        r = (x-xA) / (xB - xA)
        one_minus_r = 1.0 - r
        r_one_minues_r = r * one_minus_r
        r_squared = r**2
        num = (yB-yA) * (d*r_squared + sA*r_one_minues_r)
        den = d + (sB + sA - 2.0*d) * r_one_minues_r
        y = yA + num/den
        num_der = (d**2) * ( sB*r_squared + 2.0*d*r_one_minues_r + sA*(one_minus_r**2) )
        ladJ = tf.math.log(num_der) - 2. * tf.math.log(den)
        return  y, ladJ 
    else:
        o = sA + sB - 2.0*d
        _o = (x - yA)*o
        a = (yB - yA)*(d - sA) + _o
        b = (yB - yA)*sA       - _o
        c = d*(yA - x)
        discriminant = b**2 - 4. * a * c
        if tf.math.reduce_any(discriminant<ZERO_64):
            discriminant = tf.where(discriminant<ZERO_64, ZERO_64, discriminant)
        else: pass
        r = - 2. * c / (b + discriminant**0.5)
        y = r * (xB - xA) + xA
        #return y, -f_RQ_(y, xA, xB, yA, yB, sA=sA, sB=sB, forward=True)[-1]
        one_minus_r = 1.0 - r
        r_one_minues_r = r * one_minus_r
        r_squared = r**2
        den = d + o * r_one_minues_r
        num_der = (d**2) * ( sB*r_squared + 2.0*d*r_one_minues_r + sA*(one_minus_r**2) )
        ladJ =  2. * tf.math.log(den) - tf.math.log(num_der)
        return y, ladJ

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

def rqs_(   x, # (..., dim           ) all elements inside the interval specified below
            w, # (..., dim*n_bins    ) unconstrained real numbers
            h, # (..., dim*n_bins    ) unconstrained real numbers
            s, # (..., dim*(n_bins+1)) unconstrained real numbers
            interval = [-1.,1.],
            identity_BCs = False,
            periodic_BCs = False,
            forward = True,
            min_bin_width = 1e-4,
            knot_slope_range = [1e-4, 50.0], # the upper bound is logarithmic (soft)
            eps = 1e-6,
        ):
    x = cast_64_(x)
    w = cast_64_(w)
    h = cast_64_(h)
    s = cast_64_(s)

    eps = 1e-6 # for float32, fixing just in case.
    """ elementwise 1D rational quadratic spline on an interval = [a,b] ; b>a ; L=b-a

    REF: arXiv:1906.04032
    
    dim = number of variables
    n_bins = hyperparameter

    Inputs:
        x : array with shape (..., dim            ) # all numbers in [a,b]
        w : array with shape (..., dim*n_bins     ) # any real numbers
        h : array with shape (..., dim*n_bins     ) # any real numbers
        s : array with shape (..., dim*(n_bins+1) ) # any real numbers
        interval : list = [a,b], same for input and output variables
        identity_BCs : if True both edge slopes are fixed to ones (2*dim elements of s per datapoint are neglected; no gradient flows through them)
        periodic_BCs : if True and identity_BCs False, the slopes at the outermost edges of the interval are averages of the 2 elements of s in each dim.
                       if False and identity_BCs False, the slopes at the outermost edges of the interval are independent
                       [ all slopes are greater than knot_slope_range[0] and capped smoothly near knot_slope_range[1] ]
        forward : False when inverting (keeping w,h,s fixed in either case)
        min_bin_width : float : not too high, depends on n_bins (warning if problem)
        knot_slope_range : dont want near zero or very steep slopes, as both can create spikes in training.
                         : the cap at knot_slope_range[1] is soft (not exactly at the number provided)
                         : can plot normalize_knot_slopes_(linspace(-100,100,1000), knot_slope_range) to visualise this filter
        eps : depends on DTYPE, needs to be sufficently positive.
            : can be adjusted with x = a*ones([high,high]) and x = b*ones([high,high]), 
            # using random whs, looking for the lowest error in terms of invertibility.
            # [identity_BCs True, is best against such round off issues, periodic second best.]

    Outputs:
        y    : (..., dim) : transformed variables
        ladJ : (..., dim) : ln( dy/dx ) : log of gradient of output with respect to input
    """

    dim = x.shape[-1]
    shape_parallel = [_ for _ in x.shape[1:-1]] # list of ints, empty if ... = batch size
    n_axes = len(shape_parallel) + 2
    n_bins = w.shape[-1] // dim
    a, b = interval ; L = b - a
    
    if min_bin_width * n_bins > L: warnings.warn("!! minimal bin width too large for the number of bins")
    else: pass
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    
    x = clip_by_value_preserve_gradient(x, a+eps, b-eps) # needed
    
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

    grid_x = get_grid_(w, dim=dim, n_bins=n_bins, min_bin_w=min_bin_width, interval=interval, shape_parallel=shape_parallel)
    grid_y = get_grid_(h, dim=dim, n_bins=n_bins, min_bin_w=min_bin_width, interval=interval, shape_parallel=shape_parallel)

    grid_x = clip_by_value_preserve_gradient(grid_x, a, b) # better
    grid_y = clip_by_value_preserve_gradient(grid_y, a, b) # better

    if forward:
        diff_grid_x = grid_x - tf.expand_dims(x, axis=-1)
        ind_right_grid = tf.argmin(tf.where(diff_grid_x>=0.0, diff_grid_x, L+1.0), axis=-1)[...,tf.newaxis]
    else:
        diff_grid_y = grid_y - tf.expand_dims(x, axis=-1)
        ind_right_grid = tf.argmin(tf.where(diff_grid_y>=0.0, diff_grid_y, L+1.0), axis=-1)[...,tf.newaxis]

    xA = tf.gather_nd(grid_x, ind_right_grid-1, batch_dims=n_axes) # (m,dim)
    xB = tf.gather_nd(grid_x, ind_right_grid,   batch_dims=n_axes) # (m,dim)
    yA = tf.gather_nd(grid_y, ind_right_grid-1, batch_dims=n_axes) # (m,dim)
    yB = tf.gather_nd(grid_y, ind_right_grid,   batch_dims=n_axes) # (m,dim)
    
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

    grid_slopes = normalize_knot_slopes_(tf.reshape(s, [-1]+shape_parallel+[dim, n_bins+1]), knot_slope_range=knot_slope_range)

    if identity_BCs: 
        ones = tf.ones_like(grid_slopes[...,:1])
        grid_slopes = tf.concat([ones, grid_slopes[...,1:-1], ones], axis=-1)
    else:
        if periodic_BCs:
            edge_slope = 0.5*(grid_slopes[...,:1] + grid_slopes[...,-1:])
            grid_slopes = tf.concat([edge_slope, grid_slopes[...,1:-1], edge_slope], axis=-1)
        else: pass
        
    sA = tf.gather_nd(grid_slopes, ind_right_grid-1, batch_dims=n_axes) # (m,dim)
    sB = tf.gather_nd(grid_slopes, ind_right_grid,   batch_dims=n_axes) # (m,dim)
        
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    
    y, ladJ = rqs_bin_(x, xA, xB, yA, yB, sA=sA, sB=sB, forward=forward) # A, B refer to the two knots.
    
    y = cast_32_(y)
    ladJ = cast_32_(ladJ)

    return y, ladJ # (m,dim), (m,dim)

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

def shift_(x,                    # (..., dim)
           shifts,               # (..., dim), or a scalar
           interval = [-1.0,1.0],
           forward = True,
           ):
    a, b = interval ; L = b-a
    if forward: return tf.math.floormod(x-a + shifts, L) + a
    else: return       tf.math.floormod(x-a - shifts, L) + a

def periodic_rqs_(  x,
                    list_w,               # 2 * [(..., n_bins * dim)]
                    list_h,               # 2 * [(..., n_bins * dim)]
                    list_s,               # 2 * [(..., (n_bins+1) * dim)]
                    list_shifts,          # 2 * [(..., dim)]
                    interval = [-1.0, 1.0],
                    forward = True,
                    min_bin_width = 0.001,
                    knot_slope_range = [0.001, 50.0],
                    eps = 1e-6,
                ):
    # rqs with periodic shifts

    n_transforms = list_h.shape[0] # len(list_h) # =2 in practice

    #if None in list_shifts: a, b = interval ; L = b - a ; list_shifts = ([0.0, 0.5*L]*n_transforms)[:n_transforms]
    #else: pass 

    if forward: inds_list = [i for i in range(n_transforms)]
    else:       inds_list = [n_transforms-1-i for i in range(n_transforms)]

    ladJ = 0.0
    for i in inds_list:
        x = shift_(x, list_shifts[i], interval=interval, forward=True)
        x, ladj = rqs_( x=x,
                        w=list_w[i],
                        h=list_h[i],
                        s=list_s[i],
                        interval = interval,
                        identity_BCs = False,
                        periodic_BCs = True,
                        forward = forward,
                        min_bin_width = min_bin_width,
                        knot_slope_range = knot_slope_range,
                        eps = eps,
                        )
        ladJ += ladj
        x = shift_(x, list_shifts[i], interval=interval, forward=False)

    return x, ladJ # (..., dim), (..., dim)

############################################################################################
# test local version:

def test_periodic_rqs_(n_bins = 8,
                       n_transforms = 2,
                       min_bin_width = 0.001,
                       knot_slope_range = [0.001,50.0],
                       ):
    m = 1000
    import numpy as np
    import matplotlib.pyplot as plt

    x = cast_32_(np.linspace(-1.0, 1.0, m)[:,np.newaxis]) ; dim = 1
    w = cast_32_([np.concatenate([np.random.randn(1,dim*n_bins)]*m, axis=0) for i in range(n_transforms)])
    h = cast_32_([np.concatenate([np.random.randn(1,dim*n_bins)]*m, axis=0) for i in range(n_transforms)])
    s = cast_32_([np.concatenate([np.random.randn(1,dim*(n_bins+1))]*m, axis=0) for i in range(n_transforms)])

    list_shifts = ([0.0, 1.0]*n_transforms)[:n_transforms]

    y, ladJxy = periodic_rqs_(x, list_w=w, list_h=h, list_s=s,
                              list_shifts = list_shifts,
                              forward = True,
                              min_bin_width = min_bin_width, knot_slope_range = knot_slope_range,
                            )

    _x, ladJyx = periodic_rqs_(y, list_w=w, list_h=h, list_s=s,
                               list_shifts = list_shifts,
                               forward = False,
                               min_bin_width = min_bin_width, knot_slope_range = knot_slope_range,
                            )
    
    _err_volume = np.abs(ladJxy+ladJyx)
    print('inversion error (volume):',_err_volume.mean(), _err_volume.max())

    fig = plt.figure(figsize=(5,4.5))
    plt.scatter(x,y,s=1)  # foward
    plt.scatter(y,_x,s=1) # relection of foward along the line
    plt.scatter(x,_x,s=1) # the line
    plt.show()
    fig = plt.figure(figsize=(5,4.5))
    plt.plot(ladJxy)
    plt.plot(ladJyx) # relection of ladJxy
    plt.plot(ladJxy+ladJyx)
    plt.show()

def test_periodic_rqs_parallel_(n_bins = 8,
                                n_transforms = 2,
                                min_bin_width = 0.001,
                                knot_slope_range = [0.001,50.0],
                                ):
    m = 1000
    import numpy as np
    import matplotlib.pyplot as plt

    x = cast_32_(np.linspace(-1.0, 1.0, m)[:,np.newaxis,np.newaxis]) ; dim = 1
    x = tf.concat([x,x],axis=1)
    w = tf.stack([cast_32_(np.concatenate([np.random.randn(1,2,dim*n_bins)]*m, axis=0)) for i in range(n_transforms)],axis=0)
    h = tf.stack([cast_32_(np.concatenate([np.random.randn(1,2,dim*n_bins)]*m, axis=0)) for i in range(n_transforms)],axis=0)
    s = tf.stack([cast_32_(np.concatenate([np.random.randn(1,2,dim*(n_bins+1))]*m, axis=0)) for i in range(n_transforms)],axis=0)

    list_shifts = ([0.0, 1.0]*n_transforms)[:n_transforms]

    y, ladJxy = periodic_rqs_(x, list_w=w, list_h=h, list_s=s,
                              list_shifts = list_shifts,
                              forward = True,
                              min_bin_width = min_bin_width, knot_slope_range = knot_slope_range,
                            )

    _x, ladJyx = periodic_rqs_(y, list_w=w, list_h=h, list_s=s,
                               list_shifts = list_shifts,
                               forward = False,
                               min_bin_width = min_bin_width, knot_slope_range = knot_slope_range,
                            )
    
    _err_volume = np.abs(ladJxy+ladJyx)
    print('inversion error (volume):',_err_volume.mean(), _err_volume.max())

    print(ladJyx.shape)
    fig = plt.figure(figsize=(5,4.5))
    plt.scatter(x,y,s=1)  # foward
    plt.scatter(y,_x,s=1) # relection of foward along the line
    plt.scatter(x,_x,s=1) # the line
    plt.show()
    fig = plt.figure(figsize=(5,4.5))
    plt.plot(ladJxy[...,0],color='black')
    plt.plot(ladJyx[...,0],color='black') # relection of ladJxy
    plt.plot(ladJxy[...,0]+ladJyx[...,0],color='red')
    plt.show()

############################################################################################
############################################################################################
# optional : the offical tensorflow implementation of RationalQuadraticSpline

""" * reason why commented out this block:
''' *
    commenting out to remove dependency: tensorflow_probability
    while commented out, the 'use_tfp' argument only works when it is False
        if want to change this (i.e., start using use_tfp=True somewhere):
            uncomment this block (in rqs.py) and install tfp.__version__ '0.16.0'
            install it in the same conda evironment using this command: [probably this, using pip, will work]
                python -m pip install tensorflow-probability==0.16
'''

import tensorflow_probability as tfp
RQS_class_tfp = tfp.bijectors.RationalQuadraticSpline

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

def bin_positons_(MLP_output, # (..., dim*n_bins)
                  dim,
                  n_bins,
                  domain_width,
                  min_bin_width,
                  shape_parallel = [],
                 ):
    MLP_output = tf.reshape(MLP_output, [-1] + shape_parallel + [dim, n_bins]) 
    c = domain_width - n_bins*min_bin_width
    bin_positons = tf.nn.softmax(MLP_output, axis=-1) * c + min_bin_width
    return bin_positons # (..., dim, n_bins)

def knot_slopes_(MLP_output, # (..., dim*(n_bins-1))
                 dim,
                 n_bins,
                 min_knot_slope,
                 shape_parallel = [],
                ):
    MLP_output = tf.reshape(MLP_output, [-1] + shape_parallel + [dim, n_bins-1]) 
    knot_slopes = tf.nn.softplus(MLP_output) + min_knot_slope
    return knot_slopes # (..., dim, n_bins-1)

def rqs_tfp_(
        x, # (..., dim           ) all elements inside the interval specified below
        w, # (..., dim*n_bins    ) unconstrained real numbers
        h, # (..., dim*n_bins    ) unconstrained real numbers
        s, # (..., dim*(n_bins+1)) unconstrained real numbers
        interval = [-1.0, 1.0],  
        forward = True,
        min_bin_width = 0.001,
        knot_slope_range = [0.001, None],
        ):
    min_knot_slope = knot_slope_range[0]

    dim = x.shape[-1]
    shape_parallel = [_ for _ in x.shape[1:-1]] 
    n_bins = w.shape[-1] // dim

    xy_min, xy_max = interval
    domain_width = xy_max - xy_min

    bin_positons_x = bin_positons_(w,
                                   dim = dim,
                                   n_bins = n_bins,
                                   domain_width = domain_width,
                                   min_bin_width = min_bin_width,
                                   shape_parallel = shape_parallel,
                                   ) 
    bin_positons_y = bin_positons_(h,
                                   dim = dim,
                                   n_bins = n_bins,
                                   domain_width = domain_width,
                                   min_bin_width = min_bin_width,
                                   shape_parallel = shape_parallel,
                                   )
    knot_slopes = knot_slopes_(s,
                               dim = dim,
                               n_bins = n_bins,
                               min_knot_slope = min_knot_slope,
                               shape_parallel = shape_parallel,
                               )
 
    RQS_obj = RQS_class_tfp(bin_widths = cast_64_(bin_positons_x),
                            bin_heights = cast_64_(bin_positons_y),
                            knot_slopes = cast_64_(knot_slopes),
                            range_min = cast_64_(xy_min),
                            )
    x = cast_64_(x)
    if forward:
        y = cast_32_(RQS_obj.forward(x))
        ladJ = cast_32_(RQS_obj.forward_log_det_jacobian(x))
    else:
        y = cast_32_(RQS_obj.inverse(x))
        ladJ = cast_32_(RQS_obj.inverse_log_det_jacobian(x))
    return y, ladJ

def periodic_rqs_tfp_(x,                    # (..., dim)
                      list_w,               # 2 * [(..., n_bins * dim)]
                      list_h,               # 2 * [(..., n_bins * dim)]
                      list_s,               # 2 * [(..., (n_bins+1) * dim)]
                      list_shifts,          # 2 * [(..., dim)]
                      interval = [-1.0, 1.0],
                      forward = True,
                      min_bin_width = 0.001,
                      knot_slope_range = [0.001, None],
                     ):
    n_transforms = list_h.shape[0] # len(list_h) # =2 in practice

    if forward: inds_list = [i for i in range(n_transforms)]
    else:       inds_list = [n_transforms-1-i for i in range(n_transforms)]
    
    ladJ = 0.0
    for i in inds_list:
        x = shift_(x, list_shifts[i], interval=interval, forward=True)
        x, ladj = rqs_tfp_(x,
                           w = list_w[i],
                           h = list_h[i],
                           s = list_s[i],
                           interval = interval,
                           forward = forward,
                           min_bin_width = min_bin_width,
                           knot_slope_range = knot_slope_range,
                           )
        ladJ += ladj
        x = shift_(x, list_shifts[i], interval=interval, forward=False)

    return x, ladJ # (..., dim), (..., dim)

############################################################################################
# test the tfp version:

def test_periodic_rqs_tfp_(n_bins = 8,
                           n_transforms = 2,
                           min_bin_width = 0.001,
                           knot_slope_range = [0.001, None],
                           ):
    m = 1000
    import numpy as np
    import matplotlib.pyplot as plt

    x = cast_32_(np.linspace(-1.0, 1.0, m)[:,np.newaxis]) ; dim = 1
    w = cast_32_([np.concatenate([np.random.randn(1,dim*n_bins)]*m, axis=0) for i in range(n_transforms)])
    h = cast_32_([np.concatenate([np.random.randn(1,dim*n_bins)]*m, axis=0) for i in range(n_transforms)])
    s = cast_32_([np.concatenate([np.random.randn(1,dim*(n_bins-1))]*m, axis=0) for i in range(n_transforms)])

    list_shifts = ([0.0, 1.0]*n_transforms)[:n_transforms]

    y, ladJxy = periodic_rqs_tfp_(  x,
                                    list_w=w, list_h=h, list_s=s,
                                    list_shifts = list_shifts,

                                    forward = True,
                                    min_bin_width = min_bin_width,
                                    knot_slope_range = knot_slope_range,
                                )

    _x, ladJyx = periodic_rqs_tfp_( y,
                                    list_w=w, list_h=h, list_s=s,
                                    list_shifts = list_shifts,

                                    forward = False,
                                    min_bin_width = min_bin_width, 
                                    knot_slope_range = knot_slope_range,
                                )
    
    _err_volume = np.abs(ladJxy+ladJyx)
    print('inversion error (volume):',_err_volume.mean(), _err_volume.max())

    fig = plt.figure(figsize=(5,4.5))
    plt.scatter(x,y,s=1)  # foward
    plt.scatter(y,_x,s=1) # relection of foward along the line
    plt.scatter(x,_x,s=1) # the line
    plt.show()
    fig = plt.figure(figsize=(5,4.5))
    plt.plot(ladJxy)
    plt.plot(ladJyx) # relection of ladJxy
    plt.plot(ladJxy+ladJyx)
    plt.show()

def test_periodic_rqs_tfp_parallel_(n_bins = 8,
                                    n_transforms = 2,
                                    min_bin_width = 0.001,
                                    knot_slope_range = [0.001, None],
                                    ):
    m = 1000
    import numpy as np
    import matplotlib.pyplot as plt

    x = cast_32_(np.linspace(-1.0, 1.0, m)[:,np.newaxis,np.newaxis]) ; dim = 1
    x = tf.concat([x,x],axis=1)
    w = tf.stack([cast_32_(np.concatenate([np.random.randn(1,2,dim*n_bins)]*m, axis=0)) for i in range(n_transforms)],axis=0)
    h = tf.stack([cast_32_(np.concatenate([np.random.randn(1,2,dim*n_bins)]*m, axis=0)) for i in range(n_transforms)],axis=0)
    s = tf.stack([cast_32_(np.concatenate([np.random.randn(1,2,dim*(n_bins-1))]*m, axis=0)) for i in range(n_transforms)],axis=0)

    list_shifts = ([0.0, 1.0]*n_transforms)[:n_transforms]

    y, ladJxy = periodic_rqs_tfp_(  x,
                                    list_w=w, list_h=h, list_s=s,
                                    list_shifts = list_shifts,
                                    
                                    forward = True,
                                    min_bin_width = min_bin_width,
                                    knot_slope_range = knot_slope_range,
                                )

    _x, ladJyx = periodic_rqs_tfp_( y,
                                    list_w=w, list_h=h, list_s=s,
                                    list_shifts = list_shifts,

                                    forward = False,
                                    min_bin_width = min_bin_width, 
                                    knot_slope_range = knot_slope_range,
                                )
    
    _err_volume = np.abs(ladJxy+ladJyx)
    print('inversion error (volume):',_err_volume.mean(), _err_volume.max())

    print(ladJyx.shape)
    fig = plt.figure(figsize=(5,4.5))
    plt.scatter(x,y,s=1)  # foward
    plt.scatter(y,_x,s=1) # relection of foward along the line
    plt.scatter(x,_x,s=1) # the line
    plt.show()
    fig = plt.figure(figsize=(5,4.5))
    plt.plot(ladJxy[...,0],color='black')
    plt.plot(ladJyx[...,0],color='black') # relection of ladJxy
    plt.plot(ladJxy[...,0]+ladJyx[...,0],color='red')
    plt.show()

"""