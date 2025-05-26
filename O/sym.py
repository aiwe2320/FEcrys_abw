from .NN.representation_layers import *
from .plotting import *
from .MM.sc_system import *

## ## ## ## 

def cluster_symmetric_torsion_(x, symmetry_order:int=1, offest=np.pi):
    x = np.array(x).flatten()
    assert symmetry_order >= 1
    n = symmetry_order
    decision_boundaries = np.linspace(-np.pi, np.pi, n+1)[:n] + offest/n
    cluster_assignments = np.digitize(x, decision_boundaries)
    cluster_assignments = np.where(cluster_assignments==n, 0, cluster_assignments)
    #check = np.unique(cluster_assignments)
    #if len(check) <n: print(f'!! cluster_symmetric_torsion : {len(check)} of {n} states were sampled')
    #else: pass
    #assert max(check) == n-1
    return cluster_assignments

def test_cluster_symmetric_torsion_(symmetry_order=3, m=200, flattness=0.5):
    '''
    test_cluster_symmetric_torsion_(symmetry_order=10, m=1000, flattness=0.2)
    '''
    n = symmetry_order
    x = np.mod(np.random.randn(m)*flattness,np.pi*2/n)
    x = np.concatenate([x+b for b in np.linspace(-np.pi, np.pi,n+1)[:n]], axis=0)
    x = np.mod(x + PI, 2.0*PI) - PI

    cluster_assignments = cluster_symmetric_torsion_(x, symmetry_order=n)

    plot_1D_histogram_(x, color='black')
    [plot_1D_histogram_(x[cluster_assignments==k]) for k in range(n)]
    [plt.scatter(x[cluster_assignments==k],[0]*len(x[cluster_assignments==k])) for k in range(n)]

class DatasetSymmetryReduction:
    '''

    Chemically identical atoms can be swapped. 
    The functional form of the forcefield over the atoms that can be swapped is the same.
    (i.e., atoms that can be swapped have the same charge, LJ params, and bonded environment)

    When to use:
        can't use  : if rare even not symmetry-related (cannot use in the absence of symmetry)
        not needed : symmetry-related rate events do not affect (ergodicity in) distributions that are invariant to the symmetry
        use when   : when evaluating data on potentials / distributions that are not symmetry-aware
            this includes conventional PGM (current), and ECM (lambda < 1) states (unless lambda 0 distribution ideal gas)

    current methods: methyl, trimethyl
    TODO: add more methods if needed

    lookup_indices:

        (option 1)
        depends on initial permutation
        depending on the crystal unitcell, there are currently four choices 0,1,2,3 per molecule
        can add more in self.LOOKUPS if needed, check plots to adjust.
        Current default is [0,3] which works for one of the cases here with 4 molecules in unitcell

        (option 1) : entropic correction when lookup_indices chosen to reduce all symmetry-related states to one state:
            FE_ergodic = FE_sym_randomised ~~ FE_sym_reduced - (# methyl + # trimethyl) * ln(3)
            [from the point of view of current PGM: 3**(# methyl + # trimethyl) number of FE basins reduced to 1,
             this significantly reduces the demand on a symmetry-unaware model]
             
        (option 2) 
        An unsupervised mode is simply setting lookup_indices = [-1] which will randomise 
        the rotations such that all states are sampled uniformly. This gives E_sym_randomised.
        (option 2) : no entropic correction needed. Fixes the ergodicity problem but does not make PGM training easier.

    '''
    
    def save_sym_reduced_dataset_(self, path_dataset=None, key='_sym_reduced'):
        if hasattr(self, 'path_dataset'):
            path_dataset = str(self.path_dataset)
        else: pass
        path_dataset_sym = path_dataset + key
        dataset = load_pickle_(path_dataset)
        dataset['MD dataset']['xyz'] = self.r
        save_pickle_(dataset, path_dataset_sym)

    def check_energy_(self,):
        self.u_sym = self.sc.u_(self.r, b=self.sc.boxes)
        print('np.abs(self.u_sym - self.sc.u).max():', np.abs(self.u_sym - self.sc.u).max())

    def __init__(self,
                 path_dataset : str,
                 r = None,
                 n_mol=None,
                 n_atoms_mol=None,
                 PDB_single_mol=None,
                 ):
        
        if path_dataset is None: pass
        else:
            self.path_dataset = path_dataset

            self.sc = SingleComponent.initialise_from_save_(self.path_dataset)
            r = self.sc.xyz
            n_mol = self.sc.n_mol
            n_atoms_mol = self.sc.n_atoms_mol
            PDB_single_mol = str(self.sc._single_mol_pdb_file_.absolute())
            self.sc._xyz = []

        # r : (m, N, 3)
        self.r_init = np.array(r)
        self.restart_()
        self.n_mol = n_mol
        self.n_atoms_mol = n_atoms_mol
        self.N = self.n_mol * self.n_atoms_mol
        self.PDB_single_mol = PDB_single_mol
        # 
        self.ic_map = SC_helper(self.PDB_single_mol)

        ## n=3:
        perms00 = [[0,1,2],
                   [0,2,1],
                   [1,0,2],
                   [1,2,0],
                   [2,0,1],
                   [2,1,0],
                  ]
        perms01 = [[perm[i] for i in [0,2,1]] for perm in perms00]
        
        unperm_ = lambda perm : np.array([np.where(np.array(perm).flatten() == i)[0][0] for i in range(3)]).tolist()
        
        perms10 = [unperm_(perm) for perm in perms00]
        perms11 = [[perm[i] for i in [0,2,1]] for perm in perms10]

        # nothing
        self.LOOKUP_00 = dict(zip([tuple(x) for x in perms00], [tuple(x) for x in perms00]))
        # swap last two
        self.LOOKUP_01 = dict(zip([tuple(x) for x in perms00], [tuple(x) for x in perms01]))
        # unpermute
        self.LOOKUP_10 = dict(zip([tuple(x) for x in perms00], [tuple(x) for x in perms10]))
        # untpermuate and swap last two
        self.LOOKUP_11 = dict(zip([tuple(x) for x in perms00], [tuple(x) for x in perms11]))
        self.LOOKUPS = [self.LOOKUP_00, self.LOOKUP_01, self.LOOKUP_10, self.LOOKUP_11]

        # [1,2] < [0,3] for that one, methyls (and trimethyls)

        self.LOOKUP_random_rotation = [ [0,1,2],
                                        [2,0,1],
                                        [1,2,0],
                                      ]

    def restart_(self,):
        self.r = np.array(self.r_init)
        self.n_frames = len(self.r)

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

    def set_ABCD_(self, ind_root_atom, option:int=None):
        self.ic_map.set_ABCD_(ind_root_atom=ind_root_atom, option=option)

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

    def _prepare_sort_methyl_(self,):
        methyl_group_smarts = "[CH3]"
        methyl_pattern = Chem.MolFromSmarts(methyl_group_smarts)
        matches = self.ic_map.mol.GetSubstructMatches(methyl_pattern)
        inds_C = [x[0] for x in matches]

        self.inds_torsions_ch3 = []
        for i in inds_C:
            inds_H = np.where(self.ic_map.ABCD_IC[:,1]==i)[0]
            assert len(inds_H) == 3
            inds_torsions = [self.ic_map.ABCD_IC[j] for j in inds_H]
            self.inds_torsions_ch3.append(inds_torsions)
        self.n_methyl_groups = len(self.inds_torsions_ch3)
        print('# methyl groups:', self.n_methyl_groups)

    def _sort_methyl_(self, ind_methyl, ind_mol, lookup_index=0):
        inds_torsions_0 = np.array(self.inds_torsions_ch3[ind_methyl][0])
        inds_torsions_1 = np.array(self.inds_torsions_ch3[ind_methyl][1])
        inds_torsions_2 = np.array(self.inds_torsions_ch3[ind_methyl][2])

        a = ind_mol*self.n_atoms_mol ; b = a + self.n_atoms_mol

        phi_h0 = get_torsion_np_(reshape_to_molecules_np_(self.r[:,a:b], n_atoms_in_molecule=self.n_atoms_mol, n_molecules=1)[:,0], inds_torsions_0) 

        inds_H = np.array([inds_torsions_0, inds_torsions_1, inds_torsions_2])

        symmetry_order = 3
        C = [cluster_symmetric_torsion_(phi_h0, symmetry_order=symmetry_order)]
        for _ in range(symmetry_order-1): 
            C.append(np.mod(C[-1]+1,symmetry_order))
        C = np.stack(C, axis=-1).reshape([self.n_frames, symmetry_order])

        inds_H_j =  inds_H + ind_mol * self.n_atoms_mol

        if lookup_index >= 0:
            for frame in range(self.n_frames):
                permutation = list(self.LOOKUPS[lookup_index][tuple(C[frame].tolist())])
                self.r[frame,inds_H_j,:] = np.take(self.r[frame,inds_H_j,:], permutation, axis=0)
        else:
            for frame in range(self.n_frames):
                #permutation = np.random.choice(3, 3, replace=False)
                permutation = self.LOOKUP_random_rotation[np.random.choice(3, 1, replace=False)[0]]
                self.r[frame,inds_H_j,:] = np.take(self.r[frame,inds_H_j,:], permutation, axis=0)

    def sort_methyl_(self, lookup_indices=[0,3]):
        if hasattr(self, 'n_trimethyl_groups'): pass
        else: self._prepare_sort_methyl_()

        lookup_indices = (list(lookup_indices)*self.n_mol)[:self.n_mol]
        for i in range(self.n_methyl_groups):
            print('dealing with methyl group',i)
            [self._sort_methyl_(i, j, lookup_index = lookup_indices[j]) for j in range(self.n_mol)];

    def plot_methyl_(self, axes_off=True):
        if hasattr(self, 'n_methyl_groups'): pass
        else: self._prepare_sort_methyl_()

        _range = [-np.pi, np.pi]
        fig, ax = plt.subplots(self.n_mol, self.n_methyl_groups, figsize=(6,6))
        _r = reshape_to_molecules_np_(self.r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=self.n_mol)
        for i in range(self.n_methyl_groups):
            phi_h0 = get_torsion_np_(_r,  self.inds_torsions_ch3[i][0])
            phi_h1 = get_torsion_np_(_r,  self.inds_torsions_ch3[i][1])
            phi_h2 = get_torsion_np_(_r,  self.inds_torsions_ch3[i][2])
            m = len(phi_h0)
            # phi_h0 : (m, n_mol, 1)
            a = i#*3 
            #b = i*3 + 1
            #c = i*3 + 2
            for j in range(self.n_mol):
                plot_1D_histogram_(phi_h0[:,j,0], range=_range, color='C0', ax=ax[j,a], mask_0=True)
                plot_1D_histogram_(phi_h1[:,j,0], range=_range, color='C1', ax=ax[j,a], mask_0=True)
                plot_1D_histogram_(phi_h2[:,j,0], range=_range, color='C2', ax=ax[j,a], mask_0=True)
                ax[j,a].scatter(phi_h0[:,j,0], [-1]*m, color='C0', s=0.1)
                ax[j,a].scatter(phi_h1[:,j,0], [-2]*m, color='C1', s=0.1)
                ax[j,a].scatter(phi_h2[:,j,0], [-3]*m, color='C2', s=0.1)
                for k in [a]:#,b,c]:
                    ax[j,k].set_xlim(_range)
                    if axes_off: ax[j,k].set_axis_off()
                    else: pass
        plt.tight_layout()

    ## ##

    def _prepare_sort_trimethyl_(self,):
        trimethyl_group_smarts = "CC(C)(C)C"
        trimethyl_pattern = Chem.MolFromSmarts(trimethyl_group_smarts)
        matches = self.ic_map.mol.GetSubstructMatches(trimethyl_pattern)
        inds_group_occurrences = [list(x) for x in matches]
        
        self.inds_torsions_trimethyl_cA = []
        self.inds_trimethyl_cA012h3 = []
        self.n_trimethyl_groups = 0 # 1 but anyway
        for inds_group in inds_group_occurrences:
            '''
            cD? - cC - cB - cA0 - cA0h3      (1) - (1) - (1) - [(1) - (3)] ; [move whole group according to the (1) part]
                          - cA1 - cA1h3                      - [(1) - (3)] ; [move ..]
                          - cA2 - cA2h3                      - [(1) - (3)] ; [move ..]
            '''
            ind_cC, ind_cB, ind_cA0, ind_cA1, ind_cA2 = inds_group # (5)
            ind_cD = np.where(np.abs(self.ic_map.ABCD_IC[:,:3]-np.array([[ind_cA0, ind_cB, ind_cC]])).sum(1) == 0)[0][0]
            #assert len(ind_cD) == 1
            ' actually cC and cD both dont matter'

            inds_cA0h3 = [self.ic_map.ABCD_IC[j][0] for j in np.where(self.ic_map.ABCD_IC[:,1]==ind_cA0)[0]] # (3)
            inds_cA1h3 = [self.ic_map.ABCD_IC[j][0] for j in np.where(self.ic_map.ABCD_IC[:,1]==ind_cA1)[0]] # (3)
            inds_cA2h3 = [self.ic_map.ABCD_IC[j][0] for j in np.where(self.ic_map.ABCD_IC[:,1]==ind_cA2)[0]] # (3)
            self.inds_trimethyl_cA012h3.append([inds_cA0h3, inds_cA1h3, inds_cA2h3])

            #self.inds_torsions_trimethyl_cA.append([self.ic_map.ABCD_IC[j] for j in [ind_cA0, ind_cA1, ind_cA2]])
            inds_torsions_trimethyl_cA= []
            for j in [ind_cA0, ind_cA1, ind_cA2]:
                inds_torsions_trimethyl_cA.append(self.ic_map.ABCD_IC[np.where(self.ic_map.ABCD_IC[:,0] == j)[0][0]])

            self.inds_torsions_trimethyl_cA.append(inds_torsions_trimethyl_cA)
            
            self.n_trimethyl_groups += 1

        assert len(self.inds_torsions_trimethyl_cA) == len(self.inds_trimethyl_cA012h3)
        print('# trimethyl groups:', self.n_trimethyl_groups)

    def _sort_trimethyl_(self, ind_trimethyl, ind_mol, lookup_index=0, offest=0):

        occurrence = ind_trimethyl
        i = ind_mol
        inds_torsions = self.inds_torsions_trimethyl_cA[occurrence]
        phi_cA0 = get_torsion_np_(reshape_to_molecules_np_(self.r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=self.n_mol),  inds_torsions[0])
        inds_cA0_cA1_cA2 = np.array([inds_torsions[0], inds_torsions[1], inds_torsions[2]])
        inds_A0h3_A1h3_A2h3 = np.array(self.inds_trimethyl_cA012h3[occurrence])

        symmetry_order = 3

        C = [cluster_symmetric_torsion_(phi_cA0[:,i], symmetry_order=symmetry_order, offest=offest)]
        for _ in range(symmetry_order-1):
            C.append(np.mod(C[-1]+1,symmetry_order))
        C = np.stack(C,axis=-1)
        C = C.reshape([len(C), symmetry_order])

        ''' moving the carbons A0, A1, A2 (the main part) '''
        inds_cA0_cA1_cA2_i = inds_cA0_cA1_cA2 + i * self.n_atoms_mol
        r_cA0_cA1_cA2_in = self.r[:,inds_cA0_cA1_cA2_i,:] # (m,3,3)
        # (3,3) take (3) from axis 0
        #r_cA0_cA1_cA2_out =  np.array([np.take(r_cA0_cA1_cA2_in[frame], C[frame], axis=0) for frame in range(self.n_frames)])
        #self.r[:,inds_cA0_cA1_cA2_i,:] = r_cA0_cA1_cA2_out

        ''' moving the triplets of hydrogens that were on these carbons to match '''
        
        inds_A0h3_A1h3_A2h3_i = inds_A0h3_A1h3_A2h3 + i * self.n_atoms_mol
        '''
        [A0h0,A0h1,A0h2]
        [A1h0,A1h1,A1h2]
        [A2h0,A2h1,A2h2]
        so moving the rows, but can do each hydrogen separately (can do methyl after trimethyl)
        '''
        inds_h0 = inds_A0h3_A1h3_A2h3_i[:,0] # (3)
        r_h0_in = self.r[:,inds_h0,:] 
        #r_h0_out =  np.array([np.take(r_h0_in[frame], C[frame], axis=0) for frame in range(self.n_frames)])
        #self.r[:,inds_h0,:] = r_h0_out

        inds_h1 = inds_A0h3_A1h3_A2h3_i[:,1] # (3)
        r_h1_in = self.r[:,inds_h1,:] 
        #r_h1_out =  np.array([np.take(r_h1_in[frame], C[frame], axis=0) for frame in range(self.n_frames)])
        #self.r[:,inds_h1,:] = r_h1_out

        inds_h2 = inds_A0h3_A1h3_A2h3_i[:,2] # (3)
        r_h2_in = self.r[:,inds_h2,:] 
        #r_h2_out =  np.array([np.take(r_h2_in[frame], C[frame], axis=0) for frame in range(self.n_frames)])
        #self.r[:,inds_h2,:] = r_h2_out

        if lookup_index >= 0:
            for frame in range(self.n_frames):
                permutation = list(self.LOOKUPS[lookup_index][tuple(C[frame].tolist())])
                self.r[frame, inds_cA0_cA1_cA2_i, :] = np.take(r_cA0_cA1_cA2_in[frame],  permutation, axis=0)
                self.r[frame, inds_h0, :] = np.take(r_h0_in[frame], permutation, axis=0)
                self.r[frame, inds_h1, :] = np.take(r_h1_in[frame], permutation, axis=0)
                self.r[frame, inds_h2, :] = np.take(r_h2_in[frame], permutation, axis=0)
        else:
            for frame in range(self.n_frames):
                # permutation = np.random.choice(3, 3, replace=False)
                permutation = self.LOOKUP_random_rotation[np.random.choice(3, 1, replace=False)[0]]
                self.r[frame, inds_cA0_cA1_cA2_i, :] = np.take(r_cA0_cA1_cA2_in[frame],  permutation, axis=0)
                self.r[frame, inds_h0, :] = np.take(r_h0_in[frame], permutation, axis=0)
                self.r[frame, inds_h1, :] = np.take(r_h1_in[frame], permutation, axis=0)
                self.r[frame, inds_h2, :] = np.take(r_h2_in[frame], permutation, axis=0)
        
    def sort_trimethyl_(self, lookup_indices=[0,3], offest=0):
        if hasattr(self, 'n_trimethyl_groups'): pass
        else: self._prepare_sort_trimethyl_()

        lookup_indices = (list(lookup_indices)*self.n_mol)[:self.n_mol]
        for i in range(self.n_trimethyl_groups):
            print('dealing with trimethyl group',i)
            [self._sort_trimethyl_(i, j, lookup_index = lookup_indices[j], offest=offest) for j in range(self.n_mol)];

    def plot_trimethyl_(self, mask_0=True, axes_off=True):
        if hasattr(self, 'n_trimethyl_groups'): pass
        else: self._prepare_sort_trimethyl_()

        _range = [-np.pi, np.pi]
        fig, ax = plt.subplots(self.n_mol, self.n_trimethyl_groups, figsize=(2,10))
        _r = reshape_to_molecules_np_(self.r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=self.n_mol)
        for i in range(self.n_trimethyl_groups):
            phi_h0 = get_torsion_np_(_r,  self.inds_torsions_trimethyl_cA[i][0])
            phi_h1 = get_torsion_np_(_r,  self.inds_torsions_trimethyl_cA[i][1])
            phi_h2 = get_torsion_np_(_r,  self.inds_torsions_trimethyl_cA[i][2])
            m = len(phi_h0)
            # phi_h0 : (m, n_mol, 1)
            a = i#*3 
            #b = i*3 + 1
            #c = i*3 + 2
            for j in range(self.n_mol):
                plot_1D_histogram_(phi_h0[:,j,0], range=_range, color='C0', ax=ax[j], mask_0=mask_0)
                plot_1D_histogram_(phi_h1[:,j,0], range=_range, color='C1', ax=ax[j], mask_0=mask_0)
                plot_1D_histogram_(phi_h2[:,j,0], range=_range, color='C2', ax=ax[j], mask_0=mask_0)
                ax[j].scatter(phi_h0[:,j,0], [-1]*m, color='C0', s=1)
                ax[j].scatter(phi_h1[:,j,0], [-2]*m, color='C1', s=1)
                ax[j].scatter(phi_h2[:,j,0], [-3]*m, color='C2', s=1)
                ax[j].set_xlim(_range)
                if axes_off: ax[j].set_axis_off()
                else: ax[j].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['$-\pi$','$-\pi/2$','$0$','$\pi/2$','$\pi$'])
        plt.tight_layout()
