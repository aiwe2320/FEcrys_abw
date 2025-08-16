from .mm_helper import *

## ## ## ## ## ## ## ## ## ## ## ##

class methods_for_permutation:

    ## ## ## ## ## ## ## ## ## ## ## ##
    ## moved from OPLS:

    @property
    def _current_r_(self,):
        # positions
        # unit = nanometer
        return self._unpermute_(self.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value)

    @property
    def _current_v_(self,):
        # velocities
        # unit = nanometer/picosecond
        return self._unpermute_(self.simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)._value)
    
    @property
    def _current_F_(self,):
        # forces
        # unit = kilojoule/(nanometer*mole) ; [F = -∇U]
        return self._unpermute_(self.simulation.context.getState(getForces=True).getForces(asNumpy=True)._value)
    
    @property
    def _system_mass_(self,):
        return self._unpermute_(np.array([self.system.getParticleMass(i)._value for i in range(self.system.getNumParticles())]),
                                axis=0)

    def _set_r_(self, r):
        assert r.shape == (self.N,3)
        self.simulation.context.setPositions(self._permute_(r))

    def _set_v_(self, v):
        assert v.shape == (self.N,3)
        self.simulation.context.setVelocities(self._permute_(v))

    def forward_atom_index_(self, inds):
        # select atom in openmm using this layer that converts to the intended index
        return self._unpermute_crystal_from_top_[inds]
    
    def inverse_atom_index_(self, inds):
        return self._permute_crystal_to_top_[inds]

    ## ## ## ## ## ## ## ## ## ## ## ##
    ## moved from COST_FIX_permute_xyz_after_a_trajectory:

    def set_arrays_blank_(self,):
        self._xyz_top_ = []
        self._xyz = []
        self._u = []
        self._temperature = []
        self._boxes = []
        self._COMs = []
        self.n_frames_saved = 0
        # self._Ps = []
        # self._v = []

    def save_frame_(self,):
        self._xyz_top_.append( self.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value ) # nm
        self._u.append( self._current_u_ )            #  kT
        self._temperature.append( self._current_T_  ) # K
        self._boxes.append( self._current_b_ )        # nm
        self._COMs.append( self._current_COM_ )       # nm
        self.n_frames_saved += 1                      # frames
        # self._Ps.append( self._current_P_ )
        # self._v.append(self._current_v_)

    def run_simulation_(self, n_saves, stride_save_frame:int=100, verbose_info : str = ''):
        self.stride_save_frame = stride_save_frame
        for i in range(n_saves):
            self.simulation.step(stride_save_frame)
            self.save_frame_()
            info = 'frame: '+str(self.n_frames_saved)+' T sampled:'+str(self.temperature.mean().round(3))+' T expected:'+str(self.T)+verbose_info
            print(info, end='\r')

        self._xyz += [x for x in self._unpermute_(np.array(self._xyz_top_), axis=-2)] # permute after
        # ! interupting run_simulation_ will not save any xyz data. 

    def u_(self, r, b=None):
        '''
        speed up evaluation also
        '''
        n_frames = r.shape[0]
        r = np.array(self._permute_(r)) # permute before 
        
        _r = np.array(self._current_r_)
        _b = np.array(self._current_b_)
        
        U = np.zeros([n_frames,1])
        if b is None:
            for i in range(n_frames):
                self.simulation.context.setPositions(r[i])
                U[i,0] = self._current_U_
        else:
            for i in range(n_frames):
                self.simulation.context.setPositions(r[i])
                self._set_b_(b[i])
                U[i,0] = self._current_U_

        self._set_r_(_r)
        self._set_b_(_b)

        return U*self.beta
    
class itp2FF(MM_system_helper):
    def __init__(self,):
        super().__init__()
        self.using_gmx_loader = True

    @property
    def itp_mol(self) -> Path:
        return self.misc_dir / f"{self._FF_name_}_{self.name}.itp"
    
    @property
    def top_crys(self) -> Path:
        return self.misc_dir / f"x_x_{self._FF_name_}_{self.name}.top"

    @property
    def gro_mol(self) -> Path:
        return self.misc_dir / f"{self._FF_name_}_{self.name}.gro"

    @property
    def pdb_mol(self) -> Path:
        return self.misc_dir / f"{self._FF_name_}_{self.name}.pdb"

    @property
    def single_mol_pdb(self) -> Path:
        return self.misc_dir / f"{self._FF_name_}_single_mol_{self.name}.pdb" 
    @property
    def _single_mol_pdb_file_(self):
        return self.misc_dir / f"{self._FF_name_}_single_mol_{self.name}.pdb" 
    
    @property
    def single_mol_pdb_permuted(self) -> Path:
        return self.misc_dir / f"{self._FF_name_}_single_mol_permuted_{self.name}.pdb" # to match itp
    
    @property
    def single_mol_permutations(self,) -> Path:
        return self.misc_dir / f"{self._FF_name_}_single_mol_permutations_{self.name}"

    def set_pemutation_to_match_topology_(self,):
        try: 
            self._permute_molecule_to_top_, self._unpermute_molecule_from_top_ = load_pickle_(str(self.single_mol_permutations.absolute()))
        except:
            ' explained in detail in OPLS class with same method '
            print(f'first time running {self._FF_name_}_general for this moelcule? setting permuation in case needed:')

            pdb = str(self.single_mol_pdb.absolute())
            gro = str(self.gro_mol.absolute())

            pdb_from_gro = str(self.pdb_mol.absolute())
            pdb_reordered = str(self.single_mol_pdb_permuted.absolute())

            if os.path.exists(self.pdb_mol.absolute()): pass 
            else:
                import MDAnalysis as mda # # conda install -c conda-forge mdanalysis
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    universe = mda.Universe(gro)
                    with mda.Writer(pdb_from_gro) as x:
                        x.write(universe)

            reorder_atoms_mol_(mol_pdb_fname=pdb, template_pdb_fname=pdb_from_gro, output_pdb_fname=pdb_reordered)

            r_gro = mdtraj.load(pdb_reordered, pdb_reordered).xyz[0]
            r_pdb = mdtraj.load(pdb, pdb).xyz[0]

            D = cdist_(r_pdb, r_gro, metric='euclidean') # (n_atoms_mol,n_atoms_mol)
            self._permute_molecule_to_top_     = D.argmin(0) # (n_atoms_mol,1)
            self._unpermute_molecule_from_top_ = D.argmin(1) # (n_atoms_mol,1)

            assert np.abs(self._permute_molecule_to_top_[self._unpermute_molecule_from_top_] - np.arange(self.n_atoms_mol)).sum() == 0

            print('forward permutation works:',np.abs(r_pdb[self._permute_molecule_to_top_] - r_gro).max())
            print('inverse permatation works:',np.abs(r_pdb - r_gro[self._unpermute_molecule_from_top_]).max())

            save_pickle_([self._permute_molecule_to_top_, self._unpermute_molecule_from_top_], str(self.single_mol_permutations.absolute()))
            
        self._permute_crystal_to_top_ = np.concatenate([self._permute_molecule_to_top_ + i*self.n_atoms_mol for i in range(self.n_mol)], axis=0)
        self._unpermute_crystal_from_top_ = np.concatenate([self._unpermute_molecule_from_top_ + i*self.n_atoms_mol for i in range(self.n_mol)], axis=0)
        assert np.abs(self._permute_crystal_to_top_[self._unpermute_crystal_from_top_] - np.arange(self.N)).sum() == 0

        if np.sum(np.abs(self._permute_crystal_to_top_ - np.arange(self.N))) == 0:
            self.atom_order_PDB_match_itp = True
            print('permutation not used, because not needed')
        else:
            self.atom_order_PDB_match_itp = False
            self._permute_   = lambda r, axis=-2 : np.take(r, self._permute_crystal_to_top_, axis=axis)
            self._unpermute_ = lambda r, axis=-2 : np.take(r, self._unpermute_crystal_from_top_, axis=axis)
            print(f'! using {self._FF_name_} ff with permuation of atoms turned ON.')
            if self.n_mol > 1: print('this assumes all molecules in the input PDB have the same permutation as the first molecule')
            else: pass
            print("'permuation of atoms turned ON' -> to reduce cost during run_simulation_ the method for saving xyz frames is slightly adjusted")
            self.inject_methods_from_another_class_(methods_for_permutation, include_properties=True)

    def initialise_FF_(self,):
        ''' run this only after (n_mol and n_atoms_mol) defined in __init__ of SingleComponent '''
        if os.path.exists(self.single_mol_pdb.absolute()): pass
        else: process_mercury_output_(self.PDB, self.n_atoms_mol, single_mol = True, custom_path_name=str(self.single_mol_pdb.absolute()))
            
        if os.path.exists(self.itp_mol.absolute()): pass
        else: print('!! expected file not found:',self.itp_mol.absolute(),)

        self.set_pemutation_to_match_topology_()

    def set_FF_(self,):
        ''' run this just before self.system initialisation  '''
        self.reset_n_mol_top_()
        self.ff = parmed.gromacs.GromacsTopologyFile(str(self.top_crys.absolute())) # better
        # self.ff = mm.app.GromacsTopFile(self.top_crys.absolute(), periodicBoxVectors=self.b0)
    
    def reset_n_mol_top_(self,):

        lines_to_add = [
            f'#include "{self._FF_name_}_{self.name}.itp"',
            '\n',
            '[ defaults ]',
            '; nbfunc        comb-rule       gen-pairs       fudgeLJ      fudgeQQ',
            self._FF_name_defaults_line_,
            '\n',
            '[ system ]',
            '; Name',
            self._system_name_,
            '\n',
            '[ molecules ]',
            '; Compound          #mols',
            f'{self._compound_name_}               {self.n_mol}',
            '\n',
            ]
        file_top = open(str(self.top_crys.absolute()), 'w')
        for line in lines_to_add:
            file_top.write(line + "\n")
        file_top.close()


class OPLS_general(itp2FF):
    def __init__(self,):
        super().__init__()
        self._FF_name_ = 'OPLS'
        #                           '; nbfunc        comb-rule       gen-pairs       fudgeLJ      fudgeQQ',
        self._FF_name_defaults_line_ = '1               3               yes             0.5          0.5  '
        self._system_name_ = 'Generic title'
        self._compound_name_ = 'UNK'
        
    @classmethod
    @property
    def FF_name(self,):
        return 'OPLS'

class GAFF_general(itp2FF):
    def __init__(self,):
        super().__init__()
        self._FF_name_ = 'GAFF'
        #                           '; nbfunc        comb-rule       gen-pairs       fudgeLJ      fudgeQQ',
        self._FF_name_defaults_line_ = '1               2               yes             0.5          0.83333333  '
        self._system_name_ = 'Generic title'
        self._compound_name_ = 'UNK'
        
    @classmethod
    @property
    def FF_name(self,):
        return 'GAFF'
