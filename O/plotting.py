''' plotting.py

histograms:
    f : plot_1D_histogram_
    f : interpolate_colors_
    f : plot_2D_histogram_
    f : plot_2D_histogram_matshow_

scatter plot:
    f : plot_points_3D_
    f : plot_points_
    f : save_coordiantes_as_pdb_

misc:
    all the rest

'''

from .util_np import *
import matplotlib.pyplot as plt
from IPython.display import clear_output
from IPython.display import display

## ## ## ##

def plot_mol_larger_(mol):
    mol.RemoveConformer(0)
    from rdkit.Chem import Draw
    from IPython.display import Image, display
    img = Draw.MolToImage(mol, size=(800, 800)) # Specify size here
    display(img)

def plot_1D_histogram_(x, bins=80, range=None, density=True, kwargs_for_histogram={},
                       ax=None,
                       return_max_y = False,
                       **kwargs):
    ''' 1D histogram on the relevant x-grid '''
    hist, x_grid = np.histogram(x, bins=bins, range=range, density=density, **kwargs_for_histogram)
    if mask_0:  hist = np.where(hist==0.0, np.nan, hist)
    else: pass
    x_grid = x_grid[1:] - 0.5*(x_grid[1]-x_grid[0])
    if ax is not None: ax.plot(x_grid, hist, **kwargs)
    else:              plt.plot(x_grid, hist, **kwargs)
    if return_max_y: hist[np.where(np.isfinite(hist))].max() #return hist.max()
    else: pass

def interpolate_colors_(m, c0 = [0,0,1,1], c1 = [1,0,0,1]):
    ''' 
    liner interpolation between two colours
    can be used for plt.contour(..., colors=this) or plt.scatter(..., c=this)
    m  = number of scatter points, or number of contour plot level sets
    c0 = [r,g,b,alpha] vector of initial color (e.g., blue = [0,0,1,1])
    c1 = [r,g,b,alpha] of final color          (e.g., red  = [1,0,0,1])
    '''
    c0 = np.array(c0).astype(np.float64)
    c1 = np.array(c1).astype(np.float64)

    alpha = np.linspace(0,1,m)
    cs = np.array([c0*(1-a)+c1*a for a in alpha])
    return cs

def plot_2D_histogram_(x, y, bins=80, range=[None,None], weights=None, ax=None,
                       grid_transform_=None,
                       scatter = False,
                       **kwargs):
    ''' 2D histogram on the relevant (x,y)-grid '''
    if type(bins) is list: pass
    else: bins = [bins]*2
    xy = np.stack([x.flatten(), y.flatten()],axis=-1)
    hist, axs = np.histogramdd(xy, bins=bins, range=range, density=True, weights=weights)
    axs = [ax[1:]-0.5*(ax[1]-ax[0]) for ax in axs]
    AXs = joint_grid_from_marginal_grids_(*axs,flatten_output=False)
    if grid_transform_ is None: pass
    else: AXs = grid_transform_(AXs)

    if scatter:
        del kwargs['levels']
    else:
        if 's' in kwargs:
            kwargs['linewidths'] = kwargs['s']
            del kwargs['s']
        else: pass

    if ax is None:
        if scatter: plt.scatter(AXs[0].flatten(), AXs[1].flatten(), c=hist, **kwargs)
        else:       plt.contour(*AXs, hist, **kwargs)
    else:
        if scatter: return ax.scatter(AXs[0].flatten(), AXs[1].flatten(), c=hist, **kwargs)
        else:       return ax.contour(*AXs, hist, **kwargs)

def plot_2D_histogram_matshow_(x, y, bins=80, range=[None,None],  weights=None, **kwargs):
    if type(bins) is list: pass
    else: bins = [bins]*2
    ''' 2D histogram without axes '''
    xy = np.stack([x.flatten(), y.flatten()],axis=-1)
    hist, axs = np.histogramdd(xy, bins=bins, range=range, density=True, weights=weights)
    plt.matshow(hist, **kwargs)

##

def plot_points_3D_(X,
                 show_axes=True,
                 figsize=(10,5),
                 autoscale=True,
                 show_colorbar=True,
                 view_elev_azim = [20,-10],
                 axes_labels = ['x','y','z'],
                 **kwargs):
    """ scatter m number of 3D points
        X : (m,d) ; d=3
    """
    #from mpl_toolkits.mplot3d import Axes3D
    from itertools import product, combinations

    fig = plt.figure( figsize=figsize )
    ax = fig.add_subplot(1,1,1,projection='3d')

    #draw cube
    Max = X.max()
    r = [-Max, Max]
    for s, e in combinations(np.array(list(product(r,r,r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s,e), color="black", alpha=1, linestyle='dotted', zorder=100)

    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    ax.set_zlabel(axes_labels[2])
    if not show_axes: ax.set_axis_off()
    else: pass
    img = ax.scatter(X[:,0],X[:,1],X[:,2],**kwargs, zorder=1)
    ax.view_init(elev=28.+view_elev_azim[0], azim=45+view_elev_azim[0])
    if show_colorbar: fig.colorbar(img, shrink=0.5)
    else: pass
    plt.show()

def plot_points_(X,
                 show_axes=True,
                 figsize=(5, 3),
                 autoscale=True,
                 show_colorbar=True,
                 axes_labels = ['x','y','z'],
                 **kwargs):
    """ scatter m number of 2D or 3D points
        X : (m,d) ; d \in [2,3]
    """
    dim = X.shape[1]
        
    fig = plt.figure(figsize=figsize)
    if dim >= 3: ax = fig.add_subplot(111, projection='3d') ; ax.set_zlabel(axes_labels[2])
    elif dim == 2: ax = fig.add_subplot(111)
    else: pass
    
    if autoscale: pass 
    else: ax.autoscale(enable=None, axis='both', tight=False)
        
    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    if not show_axes: ax.set_axis_off()
    else: pass
    if dim >= 3: img = ax.scatter(X[:,0], X[:,1], X[:,2], **kwargs)
    elif dim == 2: img = ax.scatter(X[:,0], X[:,1], **kwargs)
    else: pass
    if show_colorbar: fig.colorbar(img)
    else: pass
    plt.show()

def save_coordiantes_as_pdb_(coordinates, name, la=None, verbose=False, box_line =''):
    ''' to quickly visualise 3D points in VMD:
    Inputs:
        coordinates : (n_frames, n_atoms*3)
        name : str : 'this_file_name'
        la : atom labels : (n_frames, n_atoms) array for b-factor column, or None
    Outputs:
        saved name.pdb file
    '''
    coordinates = np.array(coordinates).round(decimals=2)
    n_frames = coordinates.shape[0]
    if la is None: values = np.zeros((n_frames,int(coordinates.shape[1]/3)))
    else: values = la
    pdb = open(name+'.pdb', "w")
    spaces = {0: "", 1: " ", 2: "  ", 3: "   ", 4: "    ", 5: "     ", 6: "      "}
    zeros =  {0: '', 1: '0', 2: '00', 3: '000', 4: '0000', 5: '00000'}
    frame = 0
    for i in range(0, n_frames):
        pdb.write(box_line + '\n')
        atom_index = 0
        stride = 0
        for j in range(0, int(len(coordinates[frame]) / 3)):
            atom_index = atom_index + 1
            pdb_row = []
            x_index = int(stride) ; stride = stride + 1
            y_index = int(stride) ; stride = stride + 1
            z_index = int(stride) ; stride = stride + 1
            x = str(coordinates[frame, x_index])
            y = str(coordinates[frame, y_index])
            z = str(coordinates[frame, z_index])
            n_spaces_to_add = 7 - len(str(atom_index))
            I = ("ATOM", spaces[n_spaces_to_add], str(atom_index), "  C   HET X")
            n_spaces_to_add = 4 - len(str(atom_index))
            II = (spaces[n_spaces_to_add], str(atom_index), "      ")
            n_0s_to_add = 6 - len(x)
            III = (x, zeros[n_0s_to_add], "  ")
            n_0s_to_add = 6 - len(y)
            IV = (y, zeros[n_0s_to_add], "  ")
            n_0s_to_add = 6 - len(z)
            if values[i,j] <0: sign = '-'
            else:              sign = ' '
            V = (z, zeros[n_0s_to_add], " "+sign+str(np.abs(float(values[i,j])).round(2))+" "+sign+str(np.abs(float(values[i,j])))+"           C")
            pdb_row.append(''.join(I))
            pdb_row.append(''.join(II))
            pdb_row.append(''.join(III))
            pdb_row.append(''.join(IV))
            pdb_row.append(''.join(V))
            pdb.write(''.join(pdb_row) + "\n")
        pdb.write("END" + "\n") # pdb.write("ENDMOL" + "\n")
        frame = frame + 1
        if verbose: print("new frame, "+str(i))
        else: pass
    pdb.close()
    print("saved",name+'.pdb')

##

def S_1D_(x, bins=80, range=None):
    ''' entropy estimate of 1D variable using histogram '''
    hist, ax = np.histogram(x, bins=bins, range=range, density=True)
    dx = ax[1]-ax[0]
    return - dx*(np.ma.log(hist)*hist).sum(), dx

def S_2D_(x, y, bins = [40,40], range = [None,None]):
    ''' joint entropy estimate of two variables using histogram '''
    xy = np.stack([x.flatten(),y.flatten()],axis=-1)
    hist, axs = np.histogramdd(xy, bins=bins, range=range, density=True)
    dxdy = np.prod([ax[1]-ax[0] for ax in axs])
    return - dxdy*(np.ma.log(hist)*hist).sum()

def MI_2D_(x, y, bins = [40,40], range = [None,None]):
    ''' mutual information between two 1D variables estimate using histograms '''
    xy = np.stack([x.flatten(),y.flatten()],axis=-1)
    h, axs = np.histogramdd(xy, bins=bins, range=range, density=True)
    h /= h.sum()
    mi = - (h * np.ma.log(np.ma.divide(h, np.outer(h.sum(1), h.sum(0))))).sum() / (h*np.ma.log(h)).sum() 
    return mi

##

'''
def plot_cell_(traj):
    import nglview as nv
    view = nv.show_mdtraj(traj)
    view.stage.set_parameters(cameraType= "orthographic")
    view.add_unitcell()
    return view#.display()
'''

# OLD:

def plot_vector_(v, r0, ax=None, color='black', **kwargs):
    r1 = np.array(r0)+np.array(v)
    if ax is None:
        for i in range(len(v)):
            plt.plot([r0[0][0],r1[i][0]],[r0[0][1],r1[i][1]],color=color, **kwargs)
    else:
        for i in range(len(v)):
            ax.plot([r0[0][0],r1[i][0]],[r0[0][1],r1[i][1]],color=color, **kwargs)

def plot_cell_(_traj, s=50, cmap='coolwarm', pad_ratio=0.3):
    ''' not used anymore '''
    xyz = np.array(_traj.xyz[0])
    box = np.array(_traj.unitcell_vectors[0])
    Min, Max = xyz.min(0), xyz.max(0)
    pad = (Max-Min)*pad_ratio
    s *= (pad_ratio/0.7)
    Min -= pad
    Max += pad
    n_mol = xyz.shape[0]//3
    size = np.array([1,0.4,0.4]*n_mol)
    color = np.array([1,0.1,0.4]*n_mol)
    scale = 10
    fig,ax = plt.subplots(1,3,figsize=(scale,scale))
    axes = 'XYZ'
    for i in range(3):
        x = np.mod(i,3)
        y = np.mod(i+1,3)
        z = np.mod(i+2,3)
        ax[x].scatter(xyz[:,x],xyz[:,y],c=color,s=size*s, cmap=cmap)
        ax[x].set_xlabel(axes[x])
        ax[x].set_ylabel(axes[y])
        ax[x].axis('scaled')
        ax[x].set_xlim(Min[x],Max[x])
        ax[x].set_ylim(Min[y],Max[y])
        plot_vector_([box[x,[x,y]]], [np.array([0,0])], ax[x], color='black')
        plot_vector_([box[y,[x,y]]], [np.array([0,0])], ax[x], color='black')
        plot_vector_([box[x,[x,y]]], [box[y,[x,y]]], ax[x], color='black')
        plot_vector_([box[y,[x,y]]], [box[x,[x,y]]], ax[x], color='black')
        ax[x].set_box_aspect(1)
    fig.tight_layout()
    plt.show()

def plot_OH1H2_samples_2D_(samples, n_molecules:int, n_atoms_in_molecule:int,
                           pad_ratio=0.1,
                           s=0.01, colors = ['red', 'black', 'grey'],
                           scale = 10,
                           checking_disorder = False,
                           centre = True,
                          ):
    ''' not used anymore '''
    samples = np.array(samples)
    m = samples.shape[0]
    _samples = reshape_to_molecules_np_(samples,n_atoms_in_molecule=n_atoms_in_molecule,n_molecules=n_molecules)[...,:3,:]
    if centre:
        mass = np.array([1.,0.,0.])[np.newaxis,np.newaxis,:,np.newaxis] # take COM of oxygens only.
        _samples -= (_samples*mass).sum(axis=(1,2), keepdims=True) / n_molecules
    else: pass
    O = _samples[:,:,0,:].reshape([m*n_molecules,3])
    H1 = _samples[:,:,1,:].reshape([m*n_molecules,3])
    H2 = _samples[:,:,2,:].reshape([m*n_molecules,3])

    Min, Max = _samples.min((0,1,2)), _samples.max((0,1,2))
    pad = (Max-Min)*pad_ratio
    s *= (pad_ratio/0.7)
    Min -= pad
    Max += pad
    size = np.array([1,0.4,0.4])*s
    if checking_disorder:
        fig,ax = plt.subplots(1,3,figsize=(scale,scale))
        axes = 'XYZ'
        for i in range(3):
            x = np.mod(i,3)
            y = np.mod(i+1,3)
            z = np.mod(i+2,3)
            ax[x].scatter(O[:,x],O[:,y],s=size[0], color=colors[0])
            ax[x].scatter(H1[:,x],H1[:,y],s=size[1], color=colors[1])
            #ax[x].scatter(H2[:,x],H2[:,y],s=size[2], color=colors[2])
            ax[x].set_xlabel(axes[x])
            ax[x].set_ylabel(axes[y])
            ax[x].axis('scaled')
            ax[x].set_xlim(Min[x],Max[x])
            ax[x].set_ylim(Min[y],Max[y])
            ax[x].set_box_aspect(1)
        fig.tight_layout()
        plt.show()
    
        fig,ax = plt.subplots(1,3,figsize=(scale,scale))
        axes = 'XYZ'
        for i in range(3):
            x = np.mod(i,3)
            y = np.mod(i+1,3)
            z = np.mod(i+2,3)
            ax[x].scatter(O[:,x],O[:,y],s=size[0], color=colors[0])
            #ax[x].scatter(H1[:,x],H1[:,y],s=size[1], color=colors[1])
            ax[x].scatter(H2[:,x],H2[:,y],s=size[2], color=colors[2])
            ax[x].set_xlabel(axes[x])
            ax[x].set_ylabel(axes[y])
            ax[x].axis('scaled')
            ax[x].set_xlim(Min[x],Max[x])
            ax[x].set_ylim(Min[y],Max[y])
            ax[x].set_box_aspect(1)
        fig.tight_layout()
        plt.show()
    else: pass
    fig,ax = plt.subplots(1,3,figsize=(scale,scale))
    axes = 'XYZ'
    for i in range(3):
        x = np.mod(i,3)
        y = np.mod(i+1,3)
        z = np.mod(i+2,3)
        ax[x].scatter(O[:,x],O[:,y],s=size[0], color=colors[0])
        ax[x].scatter(H1[:,x],H1[:,y],s=size[1], color=colors[1])
        ax[x].scatter(H2[:,x],H2[:,y],s=size[2], color=colors[2])
        ax[x].set_xlabel(axes[x])
        ax[x].set_ylabel(axes[y])
        ax[x].axis('scaled')
        ax[x].set_xlim(Min[x],Max[x])
        ax[x].set_ylim(Min[y],Max[y])
        ax[x].set_box_aspect(1)
    fig.tight_layout()
    plt.show()

def simple_smoothing_matrix_(S,N,c=1):
    # notperiodic.
    xs = np.linspace(0,1,S)
    z = np.linspace(0,1,N)
    c = -0.5/( (z[1]-z[0])/c )**2 ; W = []
    for x in xs:
        W.append(  np.exp(c*(z-x)**2)  )
    W = np.array(W)
    return W/W.sum(1)[:,np.newaxis]

def simple_smoother_(X,c=1.,S=None):
    ''' 
    not used
    gives running average from stochastic data X : used only for visualisation.
    '''
    # for smoothing arrays(X) shaped as (N1,) or (N1,N2) or any (N1,N2,...,Nd) up to d=13.
    ' c: lim(c -> 0) -> ~ line of best fit '
    ' S: can be None (output same shape), or an int, or a list of ints (one for each axis of the input array). '

    Ns = X.shape ; dim = len(Ns)
    
    if S is None: Ss = Ns
    elif type(S) is int: Ss = tuple([S]*dim)
    else: Ss = S
        
    summation_indices = 'ijklmnopqrstu'
    output_indices = 'abcdefghvwxyz'
    
    einsum_args = []
    string = ''

    for ax in range(dim):
        einsum_args.append(simple_smoothing_matrix_(Ss[ax],Ns[ax],c=c))
        string += (output_indices[ax] + summation_indices[ax] + ',')
    
    string += (summation_indices[:dim] + '->' + output_indices[:dim]) 
    einsum_args.insert(0,string)
    einsum_args.append(X)
    
    return np.einsum(*einsum_args)

def plot_as_ascii_(size = [300,900], c=1.2):
    ''' 
    default setting are fine for quickly  look at a curve
    
    example:
        plt.plot(x,y) ; plot_as_ascii_() # zoom out maximum amount before running this line
    
    size : [height, width] number of symbols
    c : adjust for clarity of image
    '''
    plt.savefig('checking_plot.png')
    
    from PIL import Image

    image = Image.open('checking_plot.png')
    X = np.array(image.convert('RGB').getdata()) # image.convert('RGB')
    X = X.reshape([image.size[1],image.size[0]]+ [X.shape[-1]]).mean(-1) # greyscale
    
    A,B = X.shape
    a,b = size
    Wa = simple_smoothing_matrix_(A,a,c=c)
    Wb = simple_smoothing_matrix_(B,b,c=c)
    Y = Wa.T.dot(X).dot(Wb)

    Y -= Y.min()
    Y /= Y.max()
    Y = 1.0 - Y
    Ymean = Y.mean()
    
    for row in Y:
        print(''.join(np.where(row>Ymean,'#',' ').tolist()))
        
def plot_2D_histograms_of_box_(b,
                               r0,
                               r1=None,
                               cmap='coolwarm', dpi=100,
                               bins = 1500,
                               m = None,
                               levels = 7,
                               aligment_square = 0,
                               scatter = False,
                               s = 1,
                               move_it_left = 0.0,
                              ):
    '''
    plots 3 histograms of atomic positions, viewing at the three different faces of a supercell

    Inputs:
        b : (3,3)
            supercell box
        r0 : (n_samples, N_atoms, 3)
            3 histograms from these coordiantes plotted on the left hand side
        r1 : (n_samples, N_atoms, 3)
            3 histograms from these coordiantes plotted on the right hand side
        scatter : book
            if False a contour plot version of the histograms are plotted
            if True  a scatter plot version of the histograms are plotted
    '''
    axes = [[0,1],[1,2],[0,2]]
    
    r0 = np.array(r0)[:m]
    if m is None:
        m = len(r0)
    else:
        pass 
    
    if r1 is None:
        r1 = np.array(r0)
    else:
        r1 = np.array(r1)[:m]

    assert len(r1) == len(r0)
    
    b = np.array(b)
    
    print('# samples histogramed:',m)
    
    x0 = np.mod(np.einsum('...i,ij->...j',r0,np.linalg.inv(b)),1.)
    x1 = np.mod(np.einsum('...i,ij->...j',r1,np.linalg.inv(b)),1.)

    this = np.einsum('...i,ij->...j',x0,b)

    _range = [[-this.max()*1.4,
                this.max()*1.4]]*2

    translation = [
    - np.array([ 0.0, 0.0 ]).reshape([2,1,1]),
    - np.array([ 0.2 + move_it_left , np.abs(b[2,2]) + 0.2 ]).reshape([2,1,1]),
    - np.array([ b[2,0], np.abs( b[2,2]) +0.2]).reshape([2,1,1]),
    ]
    flip = [
    np.array([1, 1]).reshape([2,1,1]),
    np.array([-1, 1]).reshape([2,1,1]),
    np.array([1, 1]).reshape([2,1,1]),
    ]

    fig, ax = plt.subplots(1, 2 + aligment_square, dpi=dpi, figsize=(10*2,4*2))
    for i in range(3):
        ax0, ax1 = axes[i]

        def grid_transformA_(x):
            A = ax0
            B = ax1
            b2D = b[[A, B]][:,[A, B]]
            # b2D /= np.linalg.norm(b2D, axis=-1,keepdims=True)
            # b2D *= np.linalg.norm(b, axis=-1)[[A,B]][:,np.newaxis]
            return np.einsum('i...,ij->j...',x,b2D)*flip[i]

        grid_transform_ = lambda x : grid_transformA_(x) + translation[i]

        plot_2D_histogram_(x0[...,ax0],
                           x0[...,ax1],
                            bins = bins,
                           range = [[0.,1.]]*2,
                           grid_transform_ = grid_transform_,
                           levels = levels, cmap=cmap, ax = ax[0], scatter=scatter, s=s)

        plot_2D_histogram_(x1[...,ax0],
                           x1[...,ax1],
                            bins = bins,
                           range = [[0.,1.]]*2,
                           grid_transform_ = grid_transform_,
                           levels = levels, cmap=cmap, ax = ax[1], scatter=scatter, s=s)

    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_xlim(*_range[0])
    ax[0].set_ylim(*_range[0])
    ax[1].set_xlim(*_range[0])
    ax[1].set_ylim(*_range[0])
    
    if aligment_square:
        ax[2].plot([0,1],[0,1], linewidth=1, color='black')
        ax[2].plot([0,1],[1,0], linewidth=1, color='black')
        ax[2].set_xlim(*_range[0])
        ax[2].set_ylim(*_range[0])
        ax[2].set_xlim(*_range[0])
        ax[2].set_ylim(*_range[0])
    else: pass
        
    plt.tight_layout()
    plt.show()
