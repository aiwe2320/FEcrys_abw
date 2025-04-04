
''' representation_layers.py
    c : SC_helper           ; TODO: merge/expand for part 1 of the project
    c : SingleComponent_map ; done
    
    TODO: !! add back rep for single molecule in vacuum  (P1 reproducibility)
    TODO: ! add back rep for molecule with only 3 atoms (P2 reproducibility)
'''

import numpy as np
from rdkit import Chem
import mdtraj

from .github_wutobias_r2z import ZMatrix # REF : https://github.com/wutobias/r2z
from .util_tf import *
from ..util_np import *

from .representation_layers import *

class SC_helper:
    ''' 
    This is longer than needed, mostly as a placholder method for other work.
    '''
    def __init__(self,
                 PDB_single_mol: str, # .pdb file of single molecule in vacuum
                ):
        self.PDB_single_mol = str(PDB_single_mol)
        self.mol = Chem.MolFromPDBFile(self.PDB_single_mol, removeHs = False)
        for i, a in enumerate(self.mol.GetAtoms()): a.SetAtomMapNum(i)
        self.mol_flat = Chem.MolFromSmiles(Chem.MolToSmiles(self.mol)) # visualised easier
        self.mass = np.array([x.GetMass() for x in self.mol.GetAtoms()])
        
        self.inds_heavy     = np.where(self.mass>1.009)[0]  ; self.n_heavy = len(self.inds_heavy)
        self.mask_heavy = np.where(self.mass>1.009,1,0)
        self.inds_hydrogens = np.where(self.mass<=1.009)[0] ; self.n_hydrogens = len(self.inds_hydrogens)
        self.mask_hydrogens = np.where(self.mass<=1.009,1,0)
        
        self.n_atoms_mol = self.mol.GetNumAtoms() ; self.inds_atoms = np.arange(self.n_atoms_mol)
        self.adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix( self.mol )
        self.atom_ranks = self.adjacency_matrix.sum(1)
        assert all([x == self.n_atoms_mol for x in [len(self.mass), len(self.adjacency_matrix),  self.n_heavy+self.n_hydrogens, self.mask_hydrogens.sum()+self.mask_heavy.sum()]])
        print('molecule with', self.n_atoms_mol,'atoms, of which',self.n_heavy,'are heavy atoms, and the rest are', self.n_hydrogens,'hydrogens.')

    def set_ABCD_(self, ind_root_atom = 10, option:int=None):
        '''
        for a given molecule these inputs are chosen by the user:
        Inputs:
            ind_root_atom : int
                index of atom in the molecule to be treated as the centre of the Cartesian block
            option        : int
                Specifies the other two atoms of the Cartesian block.
                    the handy method ZMatrix (from wutobias/r2z) will generate a physical Zmatrix (self.ABCD)
                    As any Zmatrix, this matrix starts with 3 atoms that are different from the rest.
                    While the ind_root_atom is chosen above, the other two atoms need to be also chosen.
                    Try different integeres 0,1,2,3,... for the option, until the option you prefer is printer.
                    After trying, each time the text is printed below to explain what happened.
                    When what happened looks good on self.mol image (atoms on the image are labeled with the indices),
                    remember the option that was made for later work with this molecule.
        '''
        if self.n_atoms_mol > 3:

            ''' ind_root_atom: first time molecule: adjust by looking at self.mol in notebook
            '''
            self.ABCD = None
            root_atm_idx = None

            root_atm_idx_options = []
            for i in range(self.n_atoms_mol):
                check_ABCD = ZMatrix(rdmol = self.mol, root_atm_idx = i).z
                if check_ABCD[1][0] == ind_root_atom:
                    root_atm_idx_options.append(i)
                    if option is not None and len(root_atm_idx_options) > option:
                        root_atm_idx = root_atm_idx_options[option]
                        break
                    else:
                        root_atm_idx = i
                else: pass

            try:
                self.ABCD = ZMatrix(rdmol = self.mol, root_atm_idx = root_atm_idx).z
            except:
                print('! could not use this atom as centre of the Cartesian_Block,')
                if self.atom_ranks[ind_root_atom] < 2:
                    print('this is because this atom is bonded to only one other atom;')
                else: pass
                print('can check for other options by running self.mol in a JN cell')

            self.inds_atoms_CB = [self.ABCD[i][0] for i in [1,0,2]]

            self.ABCD_IC = np.array([self.ABCD[i] for i in range(3, self.n_atoms_mol)])     
            self.inds_atoms_IC = self.ABCD_IC[:,0].tolist()

            self.n_atoms_IC = len(self.inds_atoms_IC)
            assert self.n_atoms_IC == self.n_atoms_mol - 3
                
            ABCD_IC_inverse = np.zeros_like(self.ABCD_IC)
            a = 0
            forward_atom_permuation = np.array(self.inds_atoms_CB + self.inds_atoms_IC)
            for index in forward_atom_permuation:
                ABCD_IC_inverse += np.where(self.ABCD_IC == index,a,0)
                a+=1
            self.ABCD_IC_inverse = ABCD_IC_inverse.tolist()
            self.inds_unpermute_atoms = np.array([np.where(forward_atom_permuation == i)[0][0] for i in range(self.n_atoms_mol)]).tolist()
        else:
            
            ''' ind_root_atom: input ignored, water?, setting the first heavy atom (Oxygen) as the root_atom
            '''
            self.inds_unpermute_atoms = 'None'
            self.ABCD_IC_inverse = 'None'
            self.n_atoms_IC = 'None'
            self.inds_atoms_IC = 'None'
            self.ABCD_IC = 'None'
            self.ABCD = 'None'
            self.inds_atoms_CB = self.inds_heavy.tolist() + list(set(np.arange(3)) - set(self.inds_heavy))

        print('atoms with incides',self.inds_atoms_CB,'are set to be the Cartesian_Block')
        print('position of the molecule specified by atoms with index:',self.inds_atoms_CB[0])
        print('rotation of the molecule specified by atoms with indices:',self.inds_atoms_CB[1:])
        if self.n_atoms_mol > 3: print('conformation of the molecule specified by all other atoms.')
        else: print('no further atoms were observed in the PDB file')

    def get_r_shape_(self, r):
        shape = r.shape
        if len(shape) == 2:
            try:
                n_mol = shape[0] // self.n_atoms_mol
                assert n_mol == shape[0] / self.n_atoms_mol
                shape_in = 'single_frame'
            except:
                n_mol = shape[1] // (self.n_atoms_mol*3)
                assert n_mol == shape[1] / (self.n_atoms_mol*3)
                shape_in = 'flat'
        elif len(shape) == 3:
            n_mol = shape[1] // self.n_atoms_mol
            assert n_mol == shape[1] / self.n_atoms_mol
            shape_in = 'atoms'
        elif len(shape) == 4:
            n_mol = shape[1]
            assert shape[2] == self.n_atoms_mol
            shape_in = 'molecules'
        else: assert len(shape) < 5
        return shape_in, n_mol

    def r_reshape_(self, r, shape_out='molecules', numpy=True):
        r_shape, n_mol = self.get_r_shape_(r)

        if numpy:
            r = np.array(r)
            if r_shape == 'single_frame':
                r = r[np.newaxis,...]
            else: pass

            if shape_out == 'molecules':
                return reshape_to_molecules_np_(r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=n_mol)
            elif shape_out == 'atoms':
                return reshape_to_atoms_np_ (r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=n_mol)
            elif shape_out == 'flat':
                return reshape_to_flat_np_(r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=n_mol)
            else: assert shape_out in ['molecules','atoms','flat']
        else:
            if r_shape == 'atoms':
                r = r[tf.newaxis,...]
            else: pass

            if shape_out == 'molecules':
                return reshape_to_molecules_tf_(r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=n_mol)
            elif shape_out == 'atoms':
                return reshape_to_atoms_tf_ (r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=n_mol)
            elif shape_out == 'flat':
                return reshape_to_flat_tf_(r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=n_mol)
            else: assert shape_out in ['molecules','atoms','flat']

    def b_reshape_m_(self, b, m=1, numpy=True, verbose=False):
        shape = b.shape
        len_shape = len(shape)
        if numpy:
            b = np.array(b)
            if len_shape < 2:
                b = np.stack([b]*m, axis=0)
            elif len_shape == 3 and shape[0] == 1 and m > 1:
                if verbose: print('single box was provided for >1 supercells, assuming NVT')
                else: pass
                b = np.concatenate([b]*m, axis=0)
            elif len_shape > 3:
                assert len_shape <= 3
            else: assert b.shape[0] == m
        else:
            if len_shape < 2:
                b = tf.stack([b]*m, axis=0)
            elif len_shape == 3 and shape[0] == 1 and m > 1:
                if verbose: print('single box was provided for >1 supercells, assuming NVT')
                else: pass
                b = tf.concat([b]*m, axis=0)
            elif len_shape > 3:
                assert len_shape <= 3
            else: assert b.shape[0] == m

        return b

    def wrap_(self, r, b, b_inv = None, output_shape=None, numpy=True):
    
        if output_shape is None:
            r_shape_in = self.get_r_shape_(r)[0]
        else: pass

        r = self.r_reshape_(r, shape_out='molecules', numpy=numpy)
        b = self.b_reshape_m_(b, m=r.shape[0], numpy=numpy, verbose=True)
        if b_inv is None: b_inv = np.linalg.inv(b)
        else:             b_inv = self.b_reshape_m_(b_inv, m=r.shape[0], numpy=numpy, verbose=True)
        string = 'omki,oij->omkj'
        if numpy:
            r_wrapped = np.einsum(string, np.mod(np.einsum(string, r, b_inv), 1.0), b)
        else:
            r_wrapped = tf.einsum(string, tf.math.floormod(tf.einsum(string, r, b_inv), 1.0), b)

        if output_shape is None:
            return self.r_reshape_(r_wrapped, shape_out=r_shape_in, numpy=numpy), [b, b_inv]
        else:
            if output_shape == 'molecules':
                return r_wrapped, [b, b_inv]
            else:
                return self.r_reshape_(r_wrapped, shape_out=output_shape, numpy=numpy), [b, b_inv]

    def unwrap_molecules_np_(self, r, b, output_shape=None):
        # tested properly how many times: 1

        if output_shape is None:
            r_shape_in = self.get_r_shape_(r)[0]
        else: pass

        numpy = True
        r, b_b_inv = self.wrap_(r, b, b_inv = None, output_shape='molecules', numpy=numpy)
        b, b_inv = b_b_inv

        r_rec =  np.einsum('omki,oij->omkj', r, b_inv)

        r_out = np.zeros_like(r)
        ind_0 = self.ABCD[0][0]
        r_out[:,:,ind_0,:] = r[:,:,ind_0,:]

        for i in range(1,self.n_atoms_mol):
            ind_0 = self.ABCD[i][1]
            ind_1 = self.ABCD[i][0]
            v_rec = r_rec[:,:,ind_1,:] - r_rec[:,:,ind_0,:] 
            r_out[:,:,ind_1,:] = np.einsum('omi,oij->omj',np.mod(v_rec + 0.5, 1) - 0.5, b) + r_out[...,ind_0,:]

        if output_shape is None:
            return self.r_reshape_(r_out, shape_out=r_shape_in, numpy=numpy)
        else:
            if output_shape == 'molecules':
                return r_out
            else:
                return self.r_reshape_(r_out, shape_out=output_shape, numpy=numpy)

    def no_jump_molecules_np_(self, r, b):
        numpy = True
        r = self.r_reshape_(r, shape_out='atoms', numpy=numpy)
        b = self.b_reshape_m_(b, m=r.shape[0], numpy=numpy, verbose=True)
        # add method to provide ref and check if it always works
        return tidy_crystal_xyz_(r, b, n_atoms_mol=self.n_atoms_mol, ind_rO=self.inds_atoms_CB[0])
    
    def unwrap_np_(self, r, b, output_shape=None):
        numpy = True
        if output_shape is None:
            r_shape_in = self.get_r_shape_(r)[0]
        else: pass

        r = self.unwrap_molecules_np_(r, b, output_shape='atoms')
        r = self.no_jump_molecules_np_(r, b)

        if output_shape is None:
            return self.r_reshape_(r, shape_out=r_shape_in, numpy=numpy)
        else:
            if output_shape == 'atoms':
                return r
            else:
                return self.r_reshape_(r, shape_out=output_shape, numpy=numpy)

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

class SingleComponent_map(SC_helper):
    ''' !! : molecule must have >3 atoms to use this M_{IC} layer '''
    def __init__(self,
                 PDB_single_mol: str,   
                ):
        super().__init__(PDB_single_mol)
        self.VERSION = 'NEW'
        
    def remove_COM_from_data_(self, r):
        # carefull with np.float32 when averaging a constant
        # b : (3,3)
        r = np.array(r).astype(np.float32)
        self._first_times_crystal_(r.shape)
        r = reshape_to_molecules_np_(r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=self.n_mol)
        ind_rO = self.inds_atoms_CB[0]
        rO = r[...,ind_rO:ind_rO+1,:]
        
        print('COM removed from data without taking into account PBC of the box')
        r_com = rO.mean(-3,keepdims=True)

        return reshape_to_atoms_np_(r-r_com, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=self.n_mol)

    ## ## ## ##

    def _forward_(self, r):
        r = reshape_to_molecules_tf_(r, n_molecules=self.n_mol, n_atoms_in_molecule=self.n_atoms_mol)
        X_IC, ladJ_IC = IC_forward_(r, self.ABCD_IC)
        X_CB, ladJ_CB = CB_forward_(tf.gather(r, self.inds_atoms_CB, axis=-2))
        return X_IC, X_CB, ladJ_IC, ladJ_CB

    def _inverse_(self, X_IC, X_CB):
        r_CB, ladJ_CB = CB_inverse_(X_CB)
        r,    ladJ_IC = IC_inverse_(X_IC, r_CB, ABCD_IC_inverse=self.ABCD_IC_inverse, inds_unpermute_atoms=self.inds_unpermute_atoms)
        r = reshape_to_atoms_tf_(r, n_molecules=self.n_mol, n_atoms_in_molecule=self.n_atoms_mol)
        return r, ladJ_IC, ladJ_CB

    ## ## ## ##

    def _first_times_crystal_(self, shape):
        if len(shape) == 3:
            self.n_mol = shape[1] // self.n_atoms_mol
            assert self.n_mol == shape[1] / self.n_atoms_mol
            self.n_atoms_crys = self.n_atoms_mol * self.n_mol
            self.n_DOF_crys = self.n_atoms_crys*3 - 3
        elif len(shape) == 4:
            assert shape[2] == self.n_atoms_mol
            self.n_mol = shape[1]
            self.n_atoms_crys = self.n_atoms_mol * self.n_mol
            self.n_DOF_crys = self.n_atoms_crys*3 - 3
        else: print('! forward_init_ : shape of input could not be interpreted')
        self.n_mol_supercell = self.n_mol
        
    def _forward_init_(self, r, batch_size = 10000):
        shape = r.shape ;  m = shape[0]
        self._first_times_crystal_(shape)
        X_IC = []
        rO = []
        q = []
        a = []
        d0 = []
        d1 = []
        ladJ_IC = []
        ladJ_CB = []
        for i in range(m//batch_size):
            _from = i*batch_size
            _to = (i+1)*batch_size
            _r = reshape_to_molecules_tf_(r[_from:_to], self.n_mol, self.n_atoms_mol)
            x_IC, x_CB, ladj_IC, ladj_CB = self._forward_(_r)
            rO.append(x_CB[0])
            q.append(x_CB[1])
            a.append(x_CB[2])
            d0.append(x_CB[3])
            d1.append(x_CB[4])
            ladJ_IC.append(ladj_IC)
            ladJ_CB.append(ladj_CB)
            X_IC.append(x_IC)

        # if any left [if none left; seems like it is fine to have this here anyway]
        _r = reshape_to_molecules_tf_(r[_to:], self.n_mol, self.n_atoms_mol)
        x_IC, x_CB, ladj_IC, ladj_CB = self._forward_(_r)
        rO.append(x_CB[0])
        q.append(x_CB[1])
        a.append(x_CB[2])
        d0.append(x_CB[3])
        d1.append(x_CB[4])
        ladJ_IC.append(ladj_IC)
        ladJ_CB.append(ladj_CB)
        X_IC.append(x_IC)

        rO = tf.concat(rO,axis=0)
        q = tf.concat(q,axis=0)
        a = tf.concat(a,axis=0)
        d0 = tf.concat(d0,axis=0)
        d1 = tf.concat(d1,axis=0)
        ladJ_IC = tf.concat(ladJ_IC,axis=0)
        ladJ_CB = tf.concat(ladJ_CB,axis=0)
        X_IC = tf.concat(X_IC,axis=0) ; print('initialising on',X_IC.shape[0],'datapoints provided')
        X_CB = [rO,q,a,d0,d1]

        return X_IC, X_CB, ladJ_IC, ladJ_CB

    def reshape_to_unitcells_(self, x,
                              combined_unitcells_with_batch_axis=False,
                              forward = True,
                              numpy = False,
                             ):
        if numpy: tf_of_np = np
        else:  tf_of_np = tf
        # (m, n_mol_supercell, D) 
        # ->
        #     (m*n_unitcells, n_mol_unitcell, D) 
        #     (m, n_unitcells, n_mol_unitcell, D) 
        shape = x.shape
        m = shape[0] ; D = shape[-1] # not -1 to catch input errors
        if combined_unitcells_with_batch_axis:
            if forward:
                return tf_of_np.reshape(x, [m*self.n_unitcells, self.n_mol_unitcell, D])
            else:
                m = m // self.n_unitcells
                return tf_of_np.reshape(x, [m, self.n_mol_supercell, D])
        else:
            if forward:
                return tf_of_np.reshape(x, [m, self.n_unitcells, self.n_mol_unitcell, D])
            else:
                return tf_of_np.reshape(x, [m, self.n_mol_supercell, D])

    def initalise_(self,
                   r_dataset,
                   b0,
                   batch_size=10000,
                   n_mol_unitcell = 1,
                   COM_remover = NotWhitenFlow, # WhitenFlow
                   focused = True,
                   ):
        # r_dataset : (m,N,3)
        # b0        : (3,3)
        assert b0.shape == (3,3)

        X_IC, X_CB = self._forward_init_(r_dataset, batch_size=batch_size)[:2]
        X_IC = X_IC.numpy()
        X_CB = [x.numpy() for x in X_CB]
        rO, q, a, d0, d1 = X_CB

        assert  self.n_mol_supercell // n_mol_unitcell ==  self.n_mol_supercell / n_mol_unitcell
        self.n_mol_unitcell = n_mol_unitcell
        self.n_unitcells = self.n_mol_supercell // self.n_mol_unitcell

        self.b0_constant = np2tf_(np.array(b0).astype(np.float32)) # reshape just to catch error
        self.supercell_Volume = det_3x3_(self.b0_constant, keepdims=True)
        self.ladJ_unit_box_forward = - tf.math.log(self.supercell_Volume)*(self.n_mol-1.0)
        self.b0_inv_constant  = np2tf_(np.linalg.inv(b0).astype(np.float32)) 
        self.h0_constant = box_forward_(self.b0_constant[tf.newaxis,...])[0][0]
        rO = np.einsum('...i,ij->...j', rO, self.b0_inv_constant.numpy())
        rO_flat = reshape_to_flat_np_(rO, self.n_mol, 1) # (m, 3*n_mol)

        'checking for molecules not to be jumping more than half box length'
        check = get_ranges_centres_(rO_flat, axis=[0])
        assert all([x < 0.5 for x in check[0]])

        self.WF = COM_remover(rO_flat, removedims=3)
        xO_flat = self.WF._forward_np(rO_flat)                # (m, 3*n_mol-3)
        self.ranges_xO, self.centres_xO = get_ranges_centres_(xO_flat, axis=[0]) # (3*n_mol-3,)

        # X_IC : (m, n_mol, n_atoms_IC, 3)
        # q    : (m, n_mol, 4)
        # a    : (m, n_mol, 1)
        # d0   : (m, n_mol, 1)
        # d1   : (m, n_mol, 1)

        bonds = X_IC[...,0]  ; bonds = np.concatenate([d0,d1,bonds],axis=-1)
        angles = X_IC[...,1] ; angles = np.concatenate([a,angles],axis=-1)
        torsions = X_IC[...,2]

        # bonds    : (m, n_mol, n_atoms_IC + 2)
        # angles   : (m, n_mol, n_atoms_IC + 1)
        # torsions : (m, n_mol, n_atoms_IC    )
        # q        : (m, n_mol, 4)
        
        bonds    = self.reshape_to_unitcells_(bonds   , combined_unitcells_with_batch_axis=True, forward=True, numpy = True) 
        angles   = self.reshape_to_unitcells_(angles  , combined_unitcells_with_batch_axis=True, forward=True, numpy = True) 
        torsions = self.reshape_to_unitcells_(torsions, combined_unitcells_with_batch_axis=True, forward=True, numpy = True) 
        q        = self.reshape_to_unitcells_(q       , combined_unitcells_with_batch_axis=True, forward=True, numpy = True)

        # bonds    : (m*n_unitcells, n_mol_unitcell, n_atoms_IC + 2)
        # angles   : (m*n_unitcells, n_mol_unitcell, n_atoms_IC + 1)
        # torsions : (m*n_unitcells, n_mol_unitcell, n_atoms_IC    )
        # q        : (m*n_unitcells, n_mol_unitcell, 4)

        self.focused = focused

        # setting axis = [0,1] recovers old behavior
        self.Focused_Bonds = FocusedBonds(bonds,          axis=[0], focused=self.focused)
        self.Focused_Angles = FocusedAngles(angles,       axis=[0], focused=self.focused)
        self.Focused_Torsions = FocusedTorsions(torsions, axis=[0], focused=self.focused)
        self.Focused_Hemisphere = FocusedHemisphere(q,              focused=True        )

        self.n_DOF_mol = self.n_atoms_IC*3 + 3 + 3
        self.ln_base_C = - np2tf_( self.n_mol_supercell*self.n_DOF_mol*np.log(2.0) )
        self.ln_base_P = - np2tf_( 3*(self.n_mol_supercell-1)*np.log(2.0) )
        
        self.set_periodic_mask_()

    def set_periodic_mask_(self,):

        self.periodic_mask = np.concatenate([
                            self.Focused_Bonds.mask_periodic[:,0],      # all zero
                            self.Focused_Angles.mask_periodic[:,0],     # all zero
                            self.Focused_Torsions.mask_periodic[:,0],   # True periodic_mask_same_in_all_molecules
                            self.Focused_Hemisphere.mask_periodic[:,0], # True periodic_mask_same_in_all_molecules
                            ], axis=-1).reshape([self.n_DOF_mol]) # reshape just to check.

    @property
    def current_masks_periodic_torsions_and_Phi(self,):
        '''
        for match_topology_ when different instances of this obj are used for the same set of coupling layers (multimap)
        '''
        return [self.Focused_Torsions.mask_periodic,
                self.Focused_Hemisphere.Focused_Phi.mask_periodic,
               ]

    def match_topology_(self, ic_maps:list):
        '''
        for the multimap functionality, when different instances of this obj are used for the same set of coupling layers
        '''
        self.Focused_Torsions.set_ranges_(
            merge_periodic_masks_([ic_map.current_masks_periodic_torsions_and_Phi[0] for ic_map in ic_maps])
        )
        self.Focused_Hemisphere.set_ranges_(
            merge_periodic_masks_([ic_map.current_masks_periodic_torsions_and_Phi[1] for ic_map in ic_maps])
        )
        self.set_periodic_mask_()


    def ln_base_C_(self, z):
        # ln(1/(2**(n_DOFs conformations & rotations))) 
        return self.ln_base_C
    
    def ln_base_P_(self, z):
        # ln(1/(2**(n_DOFs positions))) 
        return self.ln_base_P

    def ln_base_(self,inputs):
        return self.ln_base_P_(inputs[0]) + self.ln_base_C_(inputs[1])

    def sample_base_C_(self, m):
        # p_{0} (base distribution) conformations & rotations:
        return tf.clip_by_value(tf.random.uniform(shape=[m, self.n_mol_supercell, self.n_DOF_mol], minval=-1.0, maxval=1.0), -1.0, 1.0)
  
    def sample_base_P_(self, m):
        # p_{0} (base distribution) positions:
        return tf.clip_by_value(tf.random.uniform(shape=[m, 3*(self.n_mol_supercell-1)], minval=-1.0,  maxval=1.0), -1.0, 1.0)
    
    def sample_base_(self, m):
        # p_{0} (base distribution) as a whole:
        return  [self.sample_base_P_(m), self.sample_base_C_(m)]
    
    ##
    
    def forward_(self, r):
        # r : (m, N, 3)
        ladJ = 0.0
        ######################################################
        X_IC, X_CB, ladJ_IC, ladJ_CB = self._forward_(r)
        ladJ += tf.reduce_sum(ladJ_IC + ladJ_CB, axis=-2) # (m,1)
        rO, q, a, d0, d1 = X_CB # rO : (m, n_mol, 3)


        #### EXTERNAL : ####
        rO = tf.einsum('...i,ij->...j', rO, self.b0_inv_constant)
        ladJ += self.ladJ_unit_box_forward

        xO, ladJ_whiten = self.WF.forward(reshape_to_flat_tf_(rO, n_molecules=self.n_mol, n_atoms_in_molecule=1))
        # xO                  : (m, 3*(n_mol-1))
        # ladJ_whiten         : (m,1)
        ladJ += ladJ_whiten 

        xO, ladJ_scale_xO = scale_shift_x_(xO, physical_ranges_x = self.ranges_xO, physical_centres_x = self.centres_xO, forward = True)
        # xO                  : (m, 3*(n_mol-1))
        # ladJ_scale_xO       : (,)
        ladJ += ladJ_scale_xO
        
        #### INTERNAL : ####

        bonds = X_IC[...,0]  ; bonds  = tf.concat([d0,d1,bonds],axis=-1)
        angles = X_IC[...,1] ; angles = tf.concat([a,angles],axis=-1)
        torsions = X_IC[...,2]

        # bonds    : (m, n_mol, n_atoms_IC + 2)
        # angles   : (m, n_mol, n_atoms_IC + 1)
        # torsions : (m, n_mol, n_atoms_IC    )
        # q        : (m, n_mol, 4)

        bonds    = self.reshape_to_unitcells_(bonds   , combined_unitcells_with_batch_axis=True, forward=True) 
        angles   = self.reshape_to_unitcells_(angles  , combined_unitcells_with_batch_axis=True, forward=True) 
        torsions = self.reshape_to_unitcells_(torsions, combined_unitcells_with_batch_axis=True, forward=True) 
        q        = self.reshape_to_unitcells_(q       , combined_unitcells_with_batch_axis=True, forward=True)

        # bonds    : (m*n_unitcells, n_mol_unitcell, n_atoms_IC + 2)
        # angles   : (m*n_unitcells, n_mol_unitcell, n_atoms_IC + 1)
        # torsions : (m*n_unitcells, n_mol_unitcell, n_atoms_IC    )
        # q        : (m*n_unitcells, n_mol_unitcell, 4)

        x_bonds, ladJ_scale_bonds = self.Focused_Bonds(bonds, forward=True)
        # x_bonds             : same
        # ladJ_scale_bonds    : (1,1)
        ladJ += ladJ_scale_bonds * self.n_unitcells

        x_angles, ladJ_scale_angles = self.Focused_Angles(angles, forward=True)
        # x_angles            : same
        # ladJ_scale_angles   : (1,1)
        ladJ += ladJ_scale_angles * self.n_unitcells

        x_torsions, ladJ_scale_torsions = self.Focused_Torsions(torsions, forward=True)
        # x_torsions          : same
        # ladJ_scale_torsions : (1,1)
        ladJ += ladJ_scale_torsions * self.n_unitcells

        x_q, ladJ_rotations = self.Focused_Hemisphere(q, forward=True)
        ladJ_rotations = tf.reduce_sum(tf.reshape(ladJ_rotations, [-1, self.n_unitcells, 1]), axis=-2)
        # x_q                 : (same,3)
        # ladJ_rotations      : (m,1)
        ladJ += ladJ_rotations 

        # x_bonds    : (m*n_unitcells, n_mol_unitcell, n_atoms_IC + 2)
        # x_angles   : (m*n_unitcells, n_mol_unitcell, n_atoms_IC + 1)
        # x_torsions : (m*n_unitcells, n_mol_unitcell, n_atoms_IC    )
        # x_q        : (m*n_unitcells, n_mol_unitcell, 4)

        x_bonds    = self.reshape_to_unitcells_(x_bonds   , combined_unitcells_with_batch_axis=True, forward=False) 
        x_angles   = self.reshape_to_unitcells_(x_angles  , combined_unitcells_with_batch_axis=True, forward=False) 
        x_torsions = self.reshape_to_unitcells_(x_torsions, combined_unitcells_with_batch_axis=True, forward=False) 
        x_q        = self.reshape_to_unitcells_(x_q       , combined_unitcells_with_batch_axis=True, forward=False)

        # x_bonds    : (m, n_mol, n_atoms_IC + 2)
        # x_angles   : (m, n_mol, n_atoms_IC + 1)
        # x_torsions : (m, n_mol, n_atoms_IC    )
        # x_q        : (m, n_mol, 4)

        X = tf.concat([x_bonds, x_angles, x_torsions, x_q], axis=-1) 
        # X : (m, n_mol, n_DOF_mol)
        #X = self.reshape_to_unitcells_(bonds, combined_unitcells_with_batch_axis=False, forward=True)
        ## X : (m, n_unitcells, n_mol_unitcell, n_DOF_mol)

        variables = [xO , X]
        return variables, ladJ
    
    def sample_nvt_h_(self, m):
        # delta
        return tf.stack([self.h0_constant]*m)

    def inverse_(self, variables_in):
        # variables_in:
        # h  : (m,6)
        # xO : (m, 3*(n_mol-1))
        # X  : (m, n_cells, n_mol_cell*high)
        ladJ = 0.0
        ######################################################
        xO , X = variables_in
        
        toA = self.n_atoms_IC+2
        toB = toA + self.n_atoms_IC+1
        toC = toB + self.n_atoms_IC

        x_bonds = X[...,:toA]
        x_angles = X[...,toA:toB]
        x_torsions = X[...,toB:toC]
        x_q = X[...,toC:]

        x_bonds    = self.reshape_to_unitcells_(x_bonds   , combined_unitcells_with_batch_axis=True, forward=True) 
        x_angles   = self.reshape_to_unitcells_(x_angles  , combined_unitcells_with_batch_axis=True, forward=True) 
        x_torsions = self.reshape_to_unitcells_(x_torsions, combined_unitcells_with_batch_axis=True, forward=True) 
        x_q        = self.reshape_to_unitcells_(x_q       , combined_unitcells_with_batch_axis=True, forward=True)

        q, ladJ_rotations = self.Focused_Hemisphere(x_q, forward=False)
        ladJ_rotations = tf.reduce_sum(tf.reshape(ladJ_rotations, [-1, self.n_unitcells, 1]), axis=-2)
        ladJ += ladJ_rotations 
        torsions, ladJ_scale_torsions = self.Focused_Torsions(x_torsions, forward=False)
        ladJ += ladJ_scale_torsions * self.n_unitcells
        angles, ladJ_scale_angles = self.Focused_Angles(x_angles, forward=False)
        ladJ += ladJ_scale_angles * self.n_unitcells
        bonds, ladJ_scale_bonds = self.Focused_Bonds(x_bonds, forward=False)
        ladJ += ladJ_scale_bonds * self.n_unitcells

        bonds    = self.reshape_to_unitcells_(bonds   , combined_unitcells_with_batch_axis=True, forward=False) 
        angles   = self.reshape_to_unitcells_(angles  , combined_unitcells_with_batch_axis=True, forward=False) 
        torsions = self.reshape_to_unitcells_(torsions, combined_unitcells_with_batch_axis=True, forward=False) 
        q        = self.reshape_to_unitcells_(q       , combined_unitcells_with_batch_axis=True, forward=False)

        d0 = bonds[...,:1]
        d1 = bonds[...,1:2]
        bonds = bonds[...,2:]
        a = angles[...,:1]
        angles = angles[...,1:]
        X_IC = tf.stack([bonds, angles, torsions], axis=-1)

        ######################################################

        xO, ladJ_scale_xO = scale_shift_x_(xO, physical_ranges_x = self.ranges_xO, physical_centres_x = self.centres_xO, forward = False)
        ladJ += ladJ_scale_xO
        rO, ladJ_whiten = self.WF.inverse(xO)
        ladJ += ladJ_whiten 
        rO = reshape_to_atoms_tf_(rO, n_atoms_in_molecule=1, n_molecules=self.n_mol)
        rO = tf.einsum('...i,ij->...j', rO, self.b0_constant)
        ladJ -= self.ladJ_unit_box_forward

        X_CB = [rO, q, a, d0, d1]

        r, ladJ_IC, ladJ_CB = self._inverse_(X_IC=X_IC, X_CB=X_CB)
        ladJ += tf.reduce_sum(ladJ_IC + ladJ_CB, axis=-2)
        # r : (m, N, 3)
        return r, ladJ

    def xO_reshape_(self, x, forward=True):
        # PGMcrys_v2 only
        if forward:
            # x : (m, 3*(n_mol-1))
            # y : (m, n_mol, 3) with first vector zeros (dummy variables only for shape)
            x = tf.reshape(x, [-1,self.n_mol-1,3])
            x = tf.concat([tf.zeros_like(x[:,:1]), x], axis=1)
            x = tf.reshape(x, [-1, self.n_mol*3])
            x = reshape_to_atoms_tf_(x, n_atoms_in_molecule=1, n_molecules=self.n_mol)
        else:
            # y : (m, n_mol, 3)
            # x : (m, 3*(n_mol-1))
            x = reshape_to_flat_tf_(x, n_molecules=self.n_mol, n_atoms_in_molecule=1)
            x = tf.reshape(x, [-1,self.n_mol,3])
            #x -= x[:,:1]
            x = x[:,1:]
            x = tf.reshape(x, [-1, (self.n_mol-1)*3])
        return x

    @property
    def flow_mask_xO(self,):
        # since one atom is effectively removed from the crystal system due to translational invariance
        # mask for transformations of positions where there is one dummy atoms present only for shape
        # 1 = transform, 0 = keep constant (identity map)
        # (1, n_mol, 3) first vector zeros, others ones; not transforming the first vector.
        mask = self.xO_reshape_(np2tf_(np.ones([1,(self.n_mol-1)*3])), forward=True).numpy().astype(np.int32)
        return mask # (1, n_mol, 3) # can self.reshape_to_unitcells_ later if needed

    @property
    def flow_mask_X(self,):
        # all intramolecular DOFs are relevant (conformations and rotations)
        # (1, n_mol, n_DOF_mol) of ones
        mask = np.ones([1, self.n_mol ,self.n_DOF_mol]).astype(np.int32)
        return mask # (1, n_mol, n_DOF_mol) # can self.reshape_to_unitcells_ if needed
    



