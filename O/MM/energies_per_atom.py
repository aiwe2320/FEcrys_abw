''' not used so far
    exprimental method
    only for analysis, visualisation purposes 
'''

from .sc_system import *

##

class Force_Helper:
    def __init__(self,
                 sc,
                 kwargs_initialise_simulation,
                 force_name:str,
                 ):
        self.force_name = force_name
        self.kwargs_initialise_simulation = kwargs_initialise_simulation
        self.sc = sc
        self.init_r0 = self.sc._current_r_
        self.init_b0 =  self.sc._current_b_
        self.init_v = self.sc._current_v_

        self.initial_parameters = {}
        self.initial_parameters_exceptions = {}

        #self.isolate_force_()

        index = 0
        for force in self.sc.system.getForces():
            if force.getName() == self.force_name:
                index += 1
            else:
                self.sc.system.removeForce(index)

        assert len(self.sc.system.getForces()) == 1
        assert self.sc.system.getForces()[0].getName() == self.force_name 
        self.force = self.sc.system.getForces()[0]

    def set_changes_(self,):
        self.sc.initialise_simulation_(**self.kwargs_initialise_simulation)

    def evaluate_(self, r, b=None, inds_atoms_remove=None, energy=True, force=False):
        self.revert_to_default_()
        if inds_atoms_remove is not None: self.remove_atoms_(inds_atoms_remove)
        else: pass

        if energy: u = self.sc.u_(r, b=b)
        else: pass
        if force: F = self.sc.F_GPU_(r, b=b)
        else: pass

        self.revert_to_default_()

        if energy and not force:
            return u, 0.0
        elif energy and force:
            return u, F
        else:
            return 0.0, F

class Bonds(Force_Helper):
    def __init__(self,
                 sc,
                 kwargs_initialise_simulation,
                 ):
        super().__init__(sc = sc,
                         kwargs_initialise_simulation = kwargs_initialise_simulation,
                         force_name = 'HarmonicBondForce',
                         )
        self.n_bonds = self.force.getNumBonds()

        for i in range(self.n_bonds):
            self.initial_parameters[i] = [i] + list(self.force.getBondParameters(i))

    def remove_atoms_(self, inds_atoms_remove:list):
        for i in range(self.n_bonds):
            p1,p2,r0,k = self.force.getBondParameters(i)
            if p1 in inds_atoms_remove: self.force.setBondParameters(*[i,p1,p2,r0,k*0.0])
            else: pass
        self.set_changes_()

    def revert_to_default_(self,):
        for i in range(self.n_bonds):
            self.force.setBondParameters(*self.initial_parameters[i])
        self.set_changes_()

class Angles(Force_Helper):
    def __init__(self,
                 sc,
                 kwargs_initialise_simulation,
                 ):
        super().__init__(sc = sc,
                         kwargs_initialise_simulation = kwargs_initialise_simulation,
                         force_name = 'HarmonicAngleForce',
                         )
        self.n_angles = self.force.getNumAngles()

        for i in range(self.n_angles):
            self.initial_parameters[i] = [i] + list(self.force.getAngleParameters(i))

    def remove_atoms_(self, inds_atoms_remove:list):
        for i in range(self.n_angles):
            p1,p2,p3,theta,k = self.force.getAngleParameters(i)
            if p1 in inds_atoms_remove: self.force.setAngleParameters(*[i,p1,p2,p3,theta,k*0.0])
            else: pass
        self.set_changes_()

    def revert_to_default_(self,):
        for i in range(self.n_angles):
            self.force.setAngleParameters(*self.initial_parameters[i])
        self.set_changes_()

class Torsions(Force_Helper):
    def __init__(self,
                 sc,
                 kwargs_initialise_simulation,
                 ):
        super().__init__(sc = sc,
                         kwargs_initialise_simulation = kwargs_initialise_simulation,
                         force_name = 'PeriodicTorsionForce',
                         )
        self.n_torsions = self.force.getNumTorsions()

        for i in range(self.n_torsions):
            self.initial_parameters[i] = [i] + list(self.force.getTorsionParameters(i))

    def remove_atoms_(self, inds_atoms_remove:list):
        for i in range(self.n_torsions):
            p1,p2,p3,p4,L,phi,k = self.force.getTorsionParameters(i)
            if p1 in inds_atoms_remove: self.force.setTorsionParameters(*[i,p1,p2,p3,p4,L,phi,k*0.0])
            else: pass
        self.set_changes_()

    def revert_to_default_(self,):
        for i in range(self.n_torsions):
            self.force.setTorsionParameters(*self.initial_parameters[i])
        self.set_changes_()

class RB_Torsions(Force_Helper):
    def __init__(self,
                 sc,
                 kwargs_initialise_simulation,
                 ):
        super().__init__(sc = sc,
                         kwargs_initialise_simulation = kwargs_initialise_simulation,
                         force_name = 'RBTorsionForce',
                         )
        self.n_torsions = self.force.getNumTorsions()

        for i in range(self.n_torsions):
            self.initial_parameters[i] = [i] + list(self.force.getTorsionParameters(i))

    def remove_atoms_(self, inds_atoms_remove:list):
        for i in range(self.n_torsions):
            p1,p2,p3,p4, c0,c1,c2,c3,c4,c5 = self.force.getTorsionParameters(i)
            if p1 in inds_atoms_remove: 
                self.force.setTorsionParameters(*[i,p1,p2,p3,p4, 
                    c0*0.0, c1*0.0, c2*0.0, c3*0.0, c4*0.0, c5*0.0])
            else: pass
        self.set_changes_()

    def revert_to_default_(self,):
        for i in range(self.n_torsions):
            self.force.setTorsionParameters(*self.initial_parameters[i])
        self.set_changes_()

class Nonbonded(Force_Helper):
    def __init__(self,
                 sc,
                 kwargs_initialise_simulation,
                 ):
        super().__init__(sc = sc,
                         kwargs_initialise_simulation = kwargs_initialise_simulation,
                         force_name = 'NonbondedForce',
                         )
        self.n_particles = self.force.getNumParticles()
        for i in range(self.n_particles):
            self.initial_parameters[i] = [i] + list(self.force.getParticleParameters(i))

        self.n_exceptions = self.force.getNumExceptions()
        for i in range(self.n_exceptions):
            self.initial_parameters_exceptions[i] = [i] + list(self.force.getExceptionParameters(i))

    def remove_atoms_(self, inds_atoms_remove:list):
        for i in inds_atoms_remove:
            qi,sigmai,epsi = self.force.getParticleParameters(i)
            self.force.setParticleParameters(*[i, qi*0.0, sigmai, epsi*0.0])

        for i in range(self.n_exceptions):
            # check
            pi,pj,qiqj,sigmaij,epsij = self.force.getExceptionParameters(i)
            if pi in inds_atoms_remove: self.force.setExceptionParameters(*[i,pi,pj,qiqj*0.0,sigmaij,epsij*0.0])
            else: pass
        self.set_changes_()

    def revert_to_default_(self,):
        for i in range(self.n_particles):
            self.force.setParticleParameters(*self.initial_parameters[i])
        for i in range(self.n_exceptions):
            self.force.setExceptionParameters(*self.initial_parameters_exceptions[i])
        self.set_changes_()

class CustomNonbonded(Force_Helper):
    def __init__(self,
                 sc,
                 kwargs_initialise_simulation,
                 ):
        super().__init__(sc = sc,
                         kwargs_initialise_simulation = kwargs_initialise_simulation,
                         force_name = 'CustomNonbondedForce',
                         )
        self.n_particles = self.force.getNumParticles()

        for i in range(self.n_particles):
            self.initial_parameters[i] = list(self.force.getParticleParameters(i))

    def remove_atoms_(self, inds_atoms_remove:list):
        for i in inds_atoms_remove:
            param1, param2 = self.force.getParticleParameters(i)
            # sigma, epsilon
            self.force.setParticleParameters(i, [param1*0.0, param2])
        self.set_changes_()

    def revert_to_default_(self,):
        for i in range(self.n_particles):
            self.force.setParticleParameters(i, self.initial_parameters[i]) # second arg : list (of two numbers)
        self.set_changes_()

class EnergiesPerAtom:
    ''' experimental method: very slow in large N cells.
    Aim: per-atom evaluation of the forcefield
    Motivation:
        energy (u_per_atom):
            energy of a given atom = energy of the complete system - energy of system with the given atom alchemically removed
        forces (F_pairwise): [may be doing incorrectly in the long-range part]
            force on atom A from atom B = force on atom A from all atoms - forces in a system where atom A is missing
            force on atom B from atom A = - (force on atom A from atom B) ; A != B
    Use-case:
        u_per_atom : can be used to compare two different openmm FFs; which type of potential energy, in which part of molecule disagree.
    '''
    def __init__(self,
                 instance_SingleComponent,
                 #args_initialise_object,
                 #args_initialise_system,
                 #args_initialise_simulation,
                 ):
        self.obj = SingleComponent
        self.args_initialise_object = dict(instance_SingleComponent.args_initialise_object)
        self.args_initialise_system = dict(instance_SingleComponent.args_initialise_system)
        self.args_initialise_simulation = dict(instance_SingleComponent.args_initialise_simulation)
        self.args_initialise_simulation['P'] = None
        self.args_initialise_simulation['minimise'] = False
        self.initalise_()

    def initalise_a_fresh_instance_(self,):
        sc_full = self.obj(**self.args_initialise_object)
        sc_full.verbose = False
        sc_full.initialise_system_(**self.args_initialise_system)
        sc_full.initialise_simulation_(**self.args_initialise_simulation)
        return sc_full
    
    def initalise_(self,):
        self.sc_full = self.initalise_a_fresh_instance_()
        self.n_forces = len(self.sc_full.system.getForces())
        count_done = 0

        self.FORCES = {}
        for force in self.sc_full.system.getForces():
            name = force.getName()

            if name == 'HarmonicBondForce':
                self.FORCES['bonds'] = Bonds(self.initalise_a_fresh_instance_(), self.args_initialise_simulation) ; count_done += 1
            
            elif name == 'HarmonicAngleForce':
                self.FORCES['angles'] = Angles(self.initalise_a_fresh_instance_(), self.args_initialise_simulation) ; count_done += 1
           
            elif name == 'PeriodicTorsionForce':
                self.FORCES['torsions'] = Torsions(self.initalise_a_fresh_instance_(), self.args_initialise_simulation) ; count_done += 1
           
            elif name == 'RBTorsionForce':
                self.FORCES['RB_torsions'] = RB_Torsions(self.initalise_a_fresh_instance_(), self.args_initialise_simulation) ; count_done += 1
           
            elif name == 'NonbondedForce':
                self.FORCES['NB'] = Nonbonded(self.initalise_a_fresh_instance_(), self.args_initialise_simulation) ; count_done += 1
          
            elif name == 'CustomNonbondedForce':
                self.FORCES['cNB'] = CustomNonbonded(self.initalise_a_fresh_instance_(), self.args_initialise_simulation) ; count_done += 1
           
            elif name == 'CMMotionRemover': count_done += 1
            
            else: print('\n','!! force with name',name,'is not in this list; skipping it.','\n')

        print(count_done,'of the',self.n_forces,'types of forces were identified')

    def check_inputs_(self, r, b):
        r = np.array(r)
        # just in case no batch axis in the input arrays:
        if len(r.shape) == 2: r = r[np.newaxis,...]
        else: pass
        if b is not None:
            b = np.array(b)
            if len(b.shape) == 2: b = b[np.newaxis,...]
            else: pass
        else: pass
        return r, b

    def evaluate_(self, r, b=None, return_forces = False):
        ''' fast to evaluate: just different types of energies for the entire system
        Inputs:
            r : (m,N,3)
            b : (m,3,3)
        Outputs: [K = number of self.n_forces]
            u_detail : (m,K) 
            F_detail : (m,K,N,3)
        '''
        r, b = self.check_inputs_(r,b)

        u_detail = np.zeros([len(r), len(self.FORCES.keys())])
        F_detail = np.zeros([len(r), len(self.FORCES.keys()), self.sc_full.N, 3])

        a = 0
        for key in self.FORCES.keys():
            u, F = self.FORCES[key].evaluate_(r, b=b, force=return_forces)
            u_detail[:,a] = u.flatten()
            F_detail[:,a,...] = F
            a += 1

        if return_forces:
            return u_detail, F_detail
        else:
            return u_detail

    def evaluate_per_atom_(self, r, b=None, return_forces = False):
        ''' slow to evaluate: just different types of energies per atom
        Inputs:
            r : (m,N,3)
            b : (m,3,3)
        Outputs:
            u_per_atom : (m,K,N)     # 
            F_per_atom : (m,K,N,N,3) #
            F_pairwise : (m,K,N,N,3) # 
        '''
        r, b = self.check_inputs_(r,b)

        atoms_set = set(np.arange(self.sc_full.N))

        u_per_atom = np.zeros([len(r), len(self.FORCES.keys()), self.sc_full.N])
        F_per_atom = np.zeros([len(r), len(self.FORCES.keys()), self.sc_full.N, self.sc_full.N, 3])
        F_pairwise = np.zeros([len(r), len(self.FORCES.keys()), self.sc_full.N, self.sc_full.N, 3])

        a = 0
        for key in self.FORCES.keys():
            u_full, F_full = self.FORCES[key].evaluate_(r, b=b, inds_atoms_remove=None, force=return_forces)
            u_full = u_full.flatten()
            
            if key not in ['NB','cNB']:
                '''
                all pair energies already (upper) triagular
                '''
                for i in range(self.sc_full.N):
                    atom_index = self.sc_full.forward_atom_index_(i)

                    u_missing, F_missing = self.FORCES[key].evaluate_(r, b=b, inds_atoms_remove=[atom_index], force=return_forces)
                    u_per_atom[:,a,i] = u_full - u_missing.flatten()

                    F_per_atom[:,a,i,...] = F_full - F_missing

                    _from = i+1
                    F_pairwise[:,a,i,_from:,:] = F_per_atom[:,a,i,_from:,:] # ignore the diagonal
                    _from = i+1
                    F_pairwise[:,a,_from:,i,:] = - F_pairwise[:,a,i,_from:,:] # ignore the diagonal

            else:
                '''
                all pair energies are devided by 2 but need to isolate the row for energies, and row without self for pairwise force.
                '''
                for i in range(self.sc_full.N):

                    atom_index = self.sc_full.forward_atom_index_(i)
    
                    # A : row_col_removed
                    u_A, F_A = self.FORCES[key].evaluate_(r, b=b, inds_atoms_remove=[atom_index], force=return_forces)
                    # B : self keept, others removed
                    u_B, F_B = self.FORCES[key].evaluate_(r, b=b, inds_atoms_remove=list(atoms_set - {atom_index}), force=return_forces)

                    u_per_atom[:,a,i] = 0.5*(u_full - u_A.flatten() + u_B.flatten())
                    F_per_atom[:,a,i] = 0.5*(F_full - F_A + F_B)
                    
                    '''
                    _from = i+1
                    F_pairwise[:,a,i,_from:,:] = (F_per_atom[:,a,i] - F_B)[:,_from:] # ignore the diagonal
                    #F_pairwise[:,a,i,_from:,:] = F_per_atom[:,a,i,_from:,:] # ignore the diagonal

                    _from = i+1
                    F_pairwise[:,a,_from:,i,:] = - F_pairwise[:,a,i,_from:,:] # ignore the diagonal
                    '''
                    F_pairwise[:,a,i] = F_per_atom[:,a,i] - F_B
                    F_pairwise[:,a,i,i] = 0.0 # ignore the diagonal

            a += 1

        '''
        np.abs(F_per_atom[frame].sum(1) - F_detail[frame]).sum() -> ~0

        which = NB
        np.abs(F_per_atom[frame,which]).sum() ~=
        np.abs(F_pairwise[frame,which]).sum() + 2*np.abs(np.where(F_pairwise[frame,which]==0.0, F_per_atom[frame,which], 0.0)).sum()/2

        F_per_atom seems uselss, it can just be added up along the atoms to get the completely force array:
        np.abs(F_detail - F_per_atom.sum(2)).sum() -> ~0

        F_pairwise is symmetric and may or may not be useful.
        '''

        if return_forces:
            return u_per_atom, F_per_atom,  F_pairwise
        else:
            return u_per_atom

'''
test = EnergiesPerAtom(
    obj = SingleComponent,
    args_initialise_object = {       'PDB': './MM/GAFF_sc/succinic_acid/succinic_acid_G_equilibrated_v2_cell_211.pdb',
                                     'n_atoms_mol': 14,
                                     'name': 'succinic_acid'},
    
    args_initialise_system = {       'PME_cutoff': 0.36,
                                     'removeCMMotion': True,
                                     'nonbondedMethod': app.PME,
                                     'custom_EwaldErrorTolerance': 0.0001,
                                     'constraints': None},

    args_initialise_simulation = {   'rbv': None,
                                     'minimise': False,
                                     'T': 300.0,
                                     'timestep_ps': 0.002,
                                     'collision_rate': 1,
                                     'P': None,
                                     'barostat_type': 1,
                                     'barostat_1_scaling': [True, True, True],
                                     'stride_barostat': 25,
                                     'custom_integrator': None},
    
)
u_detail, F_detail = test.evaluate_(test.sc_full.r0, return_forces=True)
u_per_atom, F_per_atom,  F_pairwise = test.evaluate_per_atom_(test.sc_full.r0, return_forces=True)
'''

# towards estiamte of pressure:
# sum A : (vec B -> A)(force A -> B)
# 0.5 * np.einsum('kli,lkj->lij',vij,Fij).sum(0) / det(b) # (N,3,3) units?
# velocity part : check units
