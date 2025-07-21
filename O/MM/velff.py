from .sc_system import *

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

itp_nonbond_params_velff = { 
    ('t0',    't0')     :    [1.238989408e-03,     1.041535918e-06],
    'others' : 'others',
}

class velff(itp2FF):
    def __init__(self,):
        super().__init__()
        self._FF_name_ = 'velff'
        ''' notes:

        the following self._FF_name_defaults_line_ gives warnings that are dealt with by running self.recast_NB_()
            run self.recast_NB_() as soon as self.system is defined (added: done by default)

        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
            
        nbfunc : same as OPLS (1) ; similar to this py files was tested for OPLS that already works by default.

        gen-pairs : 1-4 interactions are active for all non-bonded (NB) interactions

        nrexcl = 2  : non-bonded interactions are excluded only for atoms separated by 0,1,2 bonds
            ! default gmx parsers (parmed or openmm) : this is currently not supported (because kept for >=3 bonds apart)

        comb-rule 1 : geometric mean for getting both eps_ij, sig_ij (or C6_ij from C6_i, C6_j, C12_ij from ...)
            this is supported only if combination rules are standard (e.g., OPLS), but not the case here, where for OPLS:
                    NonbondedForce : some LJ, no Coulombic       # defined automatically
                        < 1-5  : exceptions : qq_ij = 0 ; LJ = 0
                            set all zero
                        = 1-5  : exceptions : qq_ij = fudgeQQ * qi * qj ; LJ = fudgeLJ * combine(i, j, LJ_params)
                            combined manually as exceptions
                        > 1-5  : qi * qj ; LJ = 0
                            all remaining electrostatic (without any fudge)
                    CustomNonbondedForce : most LJ, no Coulombic # defined automatically
                        > 1-5  : LJ = customizable_energy_function(i, j, LJ_param, combination_rule)
                        <= 1-5 : exclusions for 1-5, ..., 1-1 interactions

        atoms types : itp file has LJ parameters already as combined as C6_ij and C12_ij for different atom type pairs
            ! default gmx parsers (parmed or openmm) : this is currently not supported

            the way this is dealt with in this ff (where fudgeQQ = fudgeLJ = 1):
                NonbondedForce : only Coulombic # defined manually below (to replace all automatically defined Coulombic)
                    < 1-4  : exceptions : qq_ij = 0 ; LJ = 0
                    >= 1-4 : qi * qj ; LJ = 0
                CustomNonbondedForce : only LJ  # defined manually below (to replace all automatically defined LJ)
                    >= 1-4 : LJ = LJ_function(i, j, tabulated_LJ_params)
                        tabulated_LJ_params is a function of i and j 
                    < 1-4  : exclusions for 1-3, 1-2, 1-1
                    
                    LJ_energy = dispersion_correction(switching_function(LJ_function(...,r_cut),r_switch))
                    dispersion_correction : x -> x + const*n_mol*n_mol / V
                        the exactness of the const matters for delta_u when volumes are different
                        the const takes into account the shapes of the smooth 1D functions > r_switch
        '''
        #                           '; nbfunc        comb-rule       gen-pairs       fudgeLJ      fudgeQQ',
        self._FF_name_defaults_line_ = '1               1               yes             1.0          1.0  '
        self._system_name_ = 'veliparib'
        self._compound_name_ = 'vel'
    
    @classmethod
    @property
    def FF_name(self,):
        return 'velff'
    
    def recast_NB_(self, verbose=True):

        forces_add = velff_LJ_force_(self, C6_C12_types_dictionary=itp_nonbond_params_velff) + velff_C_force_(self) 

        remove_force_by_names_(self.system, 
                               ['CustomNonbondedForce','NonbondedForce'], # ,'HarmonicBondForce','HarmonicAngleForce','PeriodicTorsionForce','RBTorsionForce']
                               verbose=verbose,
                               )
        
        for force in forces_add:
            self.system.addForce(force)

        if verbose: print(f'added {len(forces_add)} forces to the system: {[x.getName() for x in forces_add]}')
        else: pass

    def corrections_to_ff_(self, verbos=True):
        self.recast_NB_(verbose=verbos)

## ## ## ## ## ## ## 

def remove_force_by_names_(system, names:list, verbose=True):
    def remove_force_by_name_(_name):
        index = 0
        for force in system.getForces():
            if force.getName() == _name: 
                system.removeForce(index)
                return [_name]
            else: index += 1
        return []
        
    removed = []
    for name in names:
        while name in [force.getName() for force in system.getForces()]:
            removed += remove_force_by_name_(name)
    if verbose: print(f'removed {len(removed)} forces from the system: {removed}')
    else: pass
        
def _get_pairs_mol_inner_(mol, n=3):
    '''
    n atoms in a row is n-1 bonds

        n = 3 for this velff because nrexcl = 2
            within 2 bonds away removed
            within 3 bonds away kept (1-2-3-4 ; 1-4 kept)
    '''
    
    AM = Chem.rdmolops.GetAdjacencyMatrix( mol )
    n_atoms_mol = len(AM)

    import networkx as nx
    gr = nx.Graph()
    for i in range(n_atoms_mol):
        for j in range(n_atoms_mol):
            if AM[i,j]>0.1:
                gr.add_edge(i,j)
            else: pass 
    assert np.abs(nx.adjacency_matrix(gr).toarray() - AM).sum() == 0
    
    within_n_bonds_away = np.eye(n_atoms_mol)*0
    for i in range(n_atoms_mol):
        for j in range(n_atoms_mol):
            n_atoms_in_a_row = len(nx.shortest_path(gr, i,j))
            if n_atoms_in_a_row <= n: # n-1 bonds away
                within_n_bonds_away[i,j] = 1
            else: pass
                
    return within_n_bonds_away # 1 : inner, 0 : outer

def _get_pairs_(remove_mol_ij, n_mol):

    n_atoms_mol = len(remove_mol_ij)

    include_ij = np.eye(n_atoms_mol*n_mol)*0 + 1 # 1 : include
    for i in range(n_mol):
        a = n_atoms_mol*i
        b = n_atoms_mol*(i+1)
        include_ij[a:b,a:b] -= remove_mol_ij # remove 1-1, 1-2, 1-3
        
    return include_ij # 1 : include, 0 : dont include


def velff_LJ_force_(sc, C6_C12_types_dictionary):
    ''' LJ : all Lennard-Jones '''

    include_ij = _get_pairs_(_get_pairs_mol_inner_(sc.mol, n=3), sc.n_mol)
    atom_types = [x.type for x in sc.ff.atoms]
    
    _table_C6 = np.eye(sc.N)*0.0
    _table_C12 = np.eye(sc.N)*0.0
    
    for i in range(sc.N):
        for j in range(sc.N):
            if i >= j:
                type_A = atom_types[i]
                type_B = atom_types[j]
                try:  C6, C12 = C6_C12_types_dictionary[(type_A,type_B)]
                except: C6, C12 = C6_C12_types_dictionary[(type_B,type_A)]
                
                ''' testing > 1-4 earlier
                sig_i, eps_i = opls_q_sig_eps[atom_types[i]][1:]
                sig_j, eps_j = opls_q_sig_eps[atom_types[j]][1:]
                
                eps_ij = np.sqrt(eps_i*eps_j)
                sig_ij = np.sqrt(sig_i*sig_j)
    
                C6  = 4.0 * eps_ij * (sig_ij**6)
                C12 = 4.0 * eps_ij * (sig_ij**12)
                '''
                # can add if needed: filter for fudge factor (multiply it to both C6 and C12 )
                
                _table_C6[i, j] = C6
                _table_C12[i, j] = C12
                _table_C6[j, i] = _table_C6[i, j]
                _table_C12[j, i] = _table_C12[i, j]
            else: pass
    
    table_C6 = mm.Discrete2DFunction(sc.N, sc.N, _table_C6.flatten().tolist())
    table_C12 = mm.Discrete2DFunction(sc.N, sc.N, _table_C12.flatten().tolist())
    
    force = mm.CustomNonbondedForce('ecm_lambda * ((1/r^12)*C12 - (1/r^6)*C6) ; C6 = table_C6(p1,p2) ; C12 = table_C12(p1,p2)')
    force.addGlobalParameter('ecm_lambda', 1.0)
    
    force.addPerParticleParameter('p')
    force.addTabulatedFunction('table_C6', table_C6)
    force.addTabulatedFunction('table_C12', table_C12)

    for i in range(sc.N):
        force.addParticle([i])

    for i in range(sc.N):
        for j in range(sc.N):
            if i >= j:
                if include_ij[i,j] < 0.5:
                    force.addExclusion(i,j)
                else: pass
            else: pass
    
    nb_method = mm.CustomNonbondedForce.CutoffPeriodic
    force.setNonbondedMethod(nb_method)
    force.setCutoffDistance(sc.PME_cutoff * mm.unit.nanometers)
    force.setUseSwitchingFunction(True)
    force.setSwitchingDistance(sc.SwitchingFunction_factor * sc.PME_cutoff * mm.unit.nanometers)
    force.setUseLongRangeCorrection(True)
    
    return [force]

def velff_C_force_(sc):
    ''' C : all Coulombic '''
    include_ij = _get_pairs_(_get_pairs_mol_inner_(sc.mol, n=3), sc.n_mol)

    q = np.concatenate([sc.partial_charges_mol]*sc.n_mol,axis=0) # (N,)

    force = mm.NonbondedForce()
    force.setNonbondedMethod( get_force_by_name_(sc.system, 'NonbondedForce').getNonbondedMethod() ) # 4 here ; 5 (LJ-PME), but cannot use it with velff
    force.setEwaldErrorTolerance(sc.custom_EwaldErrorTolerance)
    force.setCutoffDistance(sc.PME_cutoff * mm.unit.nanometers)
    force.setIncludeDirectSpace(True)
    force.setUseSwitchingFunction(True)
    force.setSwitchingDistance(sc.SwitchingFunction_factor * sc.PME_cutoff * mm.unit.nanometers)
    force.setUseDispersionCorrection(True)

    for i in range(sc.N):
        force.addParticle(*[q[i] * unit.elementary_charge, 0.0, 0.0, ])

    for i in range(sc.N):
        for j in range(sc.N):
            if i >= j:
                if include_ij[i,j] < 0.5:
                    force.addException(*[i, j, 0.0, 0.0, 0.0])
                else: pass # can add if needed: filter for fudge factor (multiply it to qq_ij)
            else: pass

    return [force]

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

''' test prooving that mm.Discrete2DFunction working correctly in velff_LJ_force_ (not just taking first N values from the flat NxN array)

def test_velff_LJ_():

    def velff_LJ_force_old_(sc, C6_C12_types_dictionary):
        include_ij = _get_pairs_(_get_pairs_mol_inner_(sc.mol, n=3), sc.n_mol)
        atom_types = [x.type for x in sc.ff.atoms]
        
        _table_C6 = np.eye(sc.N)*0.0
        _table_C12 = np.eye(sc.N)*0.0
        
        for i in range(sc.N):
            for j in range(sc.N):
                if i >= j:
                    type_A = atom_types[i]
                    type_B = atom_types[j]
                    try:  C6, C12 = C6_C12_types_dictionary[(type_A,type_B)]
                    except: C6, C12 = C6_C12_types_dictionary[(type_B,type_A)]
                    _table_C6[i, j] = C6
                    _table_C12[i, j] = C12
                    _table_C6[j, i] = _table_C6[i, j]
                    _table_C12[j, i] = _table_C12[i, j]
                else: pass
        
        _ljr_ =    '; ljr = ( (1/r^12)*C12 - (1/r^6)*C6 ) '
        # switch:
        _x  =      '; x = (r - s) / (c - s) '
        _steps_ =  '; cr = step(c-r) ; rc = step(r-c) ; sr = step(s-r) ; rs = step(r-s) '
        _switch_ = '; switch = rs * cr * (1 - 6*(x^5) + 15*(x^4) - 10*(x^3)) + rc + sr '

        main = 'cr * switch * (ljr - 0 )'
        lj_expression = main + _switch_ + _steps_ + _x + _ljr_

        force = mm.CustomBondForce(lj_expression)
        force.setUsesPeriodicBoundaryConditions(True)
        
        force.addGlobalParameter("c", sc.PME_cutoff  * mm.unit.nanometers)
        force.addGlobalParameter("s", sc.SwitchingFunction_factor * sc.PME_cutoff * mm.unit.nanometers)
        
        force.addPerBondParameter('C6')
        force.addPerBondParameter('C12')

        for i in range(sc.N):
            for j in range(sc.N):
                if i >= j:
                    if include_ij[i,j] > 0.5:
                        force.addBond(i, j, [_table_C6[i,j], _table_C12[i,j]])
                    else: pass
                else: pass
        return force
    
    name = 'veliparib'
    n_atoms_mol = 34
    FF_class = velff 

    # new/current with LongRangeCorrection OFF
    sc = SingleComponent(PDB='./O/MM/molecules/veliparib/veliparib_I_unitcell_cell111.pdb', name=name, n_atoms_mol=n_atoms_mol, FF_class=FF_class)
    sc.verbose = False
    sc.initialise_system_(PME_cutoff=0.36, removeCMMotion=True)
    remove_force_by_names_(sc.system, ['NonbondedForce','HarmonicBondForce','HarmonicAngleForce','PeriodicTorsionForce','RBTorsionForce'])
    get_force_by_name_(sc.system, 'CustomNonbondedForce').setUseLongRangeCorrection(False)
    sc.initialise_simulation_(T=300, timestep_ps=0.001, minimise=True)
    print('######################################################################')

    # old/naive without LongRangeCorrection
    sc = SingleComponent(PDB='./O/MM/molecules/veliparib/veliparib_I_unitcell_cell111.pdb', name=name, n_atoms_mol=n_atoms_mol, FF_class=FF_class)
    sc.verbose = False
    sc.initialise_system_(PME_cutoff=0.36, removeCMMotion=True)
    force = velff_LJ_force_old_(sc, itp_nonbond_params_velff)
    remove_force_by_names_(sc.system, ['CustomNonbondedForce','NonbondedForce','HarmonicBondForce','HarmonicAngleForce','PeriodicTorsionForce','RBTorsionForce'])
    sc.system.addForce(force)
    sc.initialise_simulation_(T=300, timestep_ps=0.001, P=1, minimise=True)

    # should both give the same numbers before and after minimisation, suggesting same overall shape of PES:
    # u before minimisation: 303.63141669435004 kT
    # u after  minimisation: -77.35333936837897 kT

'''
