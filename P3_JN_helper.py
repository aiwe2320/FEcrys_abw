from O.plotting import *

# this still imports tensorflow although it is not needed here
#   any installation that allows the imports is fine for this part

# not fixing because needed in the other notebooks
#   requires correct installation of  tensorflow

from O.interface import NN_interface_sc_multimap

class P3_JN_helper:
    def __init__(self, molecule_abbreviation):
        
        self.molecule_abbreviation = molecule_abbreviation 
        if  molecule_abbreviation == 'SA':
            self.name = 'succinic_acid'
            self.n_atoms_mol = 14
            self.forms = ['A','B','G']
            self.form_to_form  = {'A':'Ae', 'B':'Be', 'G':'G'}
            
            self.list_n_mol = [8,16,24,32]
            self.cells_by_form_nmol = {
                                    'A': {          16:'222', 24:'223', 32:'224'},
                                    'B': { 8:'212', 16:'222', 24:'223', 32:'224'},
                                    'G': { 8:'211', 16:'221', 24:'321', 32:'222'},
                                    }
            self.npt_no_jump_fa_index_used = 11 # not sure why this was different from self.fa_index
            self.fa_index = 4
            
            self.ecm_key   = '_v2'
            self.ecm_batch_size_increments = 10000
            self.ecm_max_nlambdas = 30
            self.ecm_SE_tol = 0.01875

            self.nvt_key = ''
            self.nvt_nmol_to_nframes_ = lambda n_mol: n_mol*15625

            self.pgm_ind_root_atom = self.fa_index
            self.pgm_option = 2
            ''' nn.set_ic_map_step1(ind_root_atom=self.PGM_Ind_Root_Atom, option=self.PGM_Option)
            molecule with 14 atoms, of which 8 are heavy atoms, and the rest are 6 hydrogens.
            atoms with incides [4, 11, 3] are set to be the Cartesian_Block
            position of the molecule specified by atoms with index: 4
            rotation of the molecule specified by atoms with indices: [11, 3]
            conformation of the molecule specified by all other atoms.
            '''
            self.pgm_nmol_to_ntrainingbatches_ = lambda n_mol: n_mol*(15625//25)
            self.form_to_nmol_unitcell = {'A':2, 'B':2, 'G':4} # for nn.set_ic_map_step3 but this was not used in P3 version
            self.pgm_nheads = 2

        elif molecule_abbreviation == 'VEL':
            self.name = 'veliparib'
            self.n_atoms_mol = 34
            self.forms = ['I','II']
            self.list_n_mol = [8,9,12,16,18,24,27,32]
            self.form_to_form = dict(zip(self.forms,self.forms))
            
            self.cells_by_form_nmol = {
                                    'I' : { 8:'111', 16:'121', 24:'131', 32:'221'},
                                    'II': { 9:'131', 12:'221', 18:'321', 24:'222', 27:'331'},
                                    }
            self.npt_no_jump_fa_index_used = 10 # not sure why this was different from self.fa_index
            self.fa_index = 5
            
            self.ecm_key   = 'L30_'
            self.ecm_batch_size_increments = 20000
            self.ecm_max_nlambdas = 30
            self.ecm_SE_tol = 0.046
            
            self.nvt_key = ''
            self.nvt_nmol_to_nframes_ = lambda n_mol: n_mol*18750

            self.pgm_ind_root_atom = self.fa_index
            self.pgm_option = 0
            ''' nn.set_ic_map_step1(ind_root_atom=self.PGM_Ind_Root_Atom, option=self.PGM_Option)
            molecule with 34 atoms, of which 18 are heavy atoms, and the rest are 16 hydrogens.
            atoms with incides  [5, 10, 1] are set to be the Cartesian_Block
            position of the molecule specified by atoms with index: 5
            rotation of the molecule specified by atoms with indices: [10, 1]
            conformation of the molecule specified by all other atoms.
            '''
            self.pgm_nmol_to_ntrainingbatches_ = lambda n_mol: n_mol*(15625//25)
            self.form_to_nmol_unitcell = {'I':8, 'II':3} # for nn.set_ic_map_step3 but this was not used in P3 version
            self.pgm_nheads = 2

        elif  molecule_abbreviation == 'MIV':
            self.name = 'mivebresib'
            self.n_atoms_mol = 51
            self.forms = ['I','II','IIIs','IIIb']
            self.list_n_mol = [8,16]
            self.form_to_form = dict(zip(self.forms,self.forms))

            self.cells_by_form_nmol = {
                                    'I'   : { 8:'111', 16:'121'},
                                    'II'  : { 8:'221', 16:'222'},
                                    'IIIs': { 8:'111', 16:'211'},
                                    'IIIb': { 8:'111', 16:'211'},
                                    }
            self.npt_no_jump_fa_index_used = 17 # not sure why this was different from self.fa_index
            self.fa_index = 22
           
            self.ecm_key  = 'L30_'
            self.ecm_batch_size_increments = 20000
            self.ecm_max_nlambdas = 30
            self.ecm_SE_tol =  0.046

            self.nvt_key = '_long'
            self.nvt_nmol_to_nframes_ = lambda n_mol: 500000 

            self.pgm_ind_root_atom = self.fa_index 
            self.pgm_option =  0
            ''' nn.set_ic_map_step1(ind_root_atom=self.PGM_Ind_Root_Atom, option=self.PGM_Option)
            molecule with 51 atoms, of which 32 are heavy atoms, and the rest are 19 hydrogens.
            atoms with incides [22, 25, 19] are set to be the Cartesian_Block
            position of the molecule specified by atoms with index: 22
            rotation of the molecule specified by atoms with indices: [25, 19]
            conformation of the molecule specified by all other atoms.
            '''
            self.pgm_nmol_to_ntrainingbatches_ = lambda n_mol: {8:10000, 16:15000}[n_mol]
            self.form_to_nmol_unitcell = {'I':8, 'II':2, 'IIIs':8, 'IIIb':8} # for nn.set_ic_map_step3 but this was not used in P3 version
            self.pgm_nheads = 4

        else: print('! molecule_abbreviation not recognised')

        self.n_forms = len(self.forms)
        self.form_to_color = dict(zip(self.forms, ['C'+str(x) for x in range(self.n_forms)] ))
        
        self.u_mean = np.zeros([self.n_forms, len(self.list_n_mol)]) + np.nan
        
        print(self.name)
        #print('Note: [ This object (P3_JN_helper) is used to organise the results from the paper (P3) in a jupyter notebook.')
        #print('For starting a new molecule, or using the method properly, this object P3_JN_helper may not be entirely helpful. ]')

        self.form_cell_n_mol_groups = []
        for form in self.cells_by_form_nmol.keys():
            for n_mol in self.cells_by_form_nmol[form].keys():
                cell = self.cells_by_form_nmol[form][n_mol]
                self.form_cell_n_mol_groups.append([form,cell,n_mol])

        self.tabulate_ecm_BAR_result_(verbose=False)
        self.tabulate_ecm_MBAR_result_(verbose=False)

    def print(self, *x, verbose=True):
        if verbose: print(*x)
        else: pass

    ## ## ## ##

    def script_NPT_(self, form, n_mol, test_rerun=False):
        _form = self.form_to_form[form]
        cell = self.cells_by_form_nmol[form][n_mol]
        cell_as_list = [int(x) for x in  cell]
        if test_rerun: rerun = '_rerun'
        else: rerun = ''

        py_script = f'''
        from .{DIR_main.split('/')[-2]}.MM.sc_system import *

        # NPT script (equilibrating the ideal supercell to the forcefield):
        # this is part of a loop, but more clear to write here as a single example

        print('starting:', {_form}, {cell})
        try:
            PDB_unitcell = '{DIR_main}MM/GAFF_sc{rerun}/{self.name}/{self.name}_{_form}_unitcell.pdb'
            PDB_supercell = supercell_from_unitcell_(PDB_unitcell, cell={cell_as_list}, save_output=True)[-1]
            sc = SingleComponent(PDB = str(PDB_supercell), n_atoms_mol={self.n_atoms_mol}, name={self.name}, FF_name='GAFF')
            sc.initialise_system_(PME_cutoff=0.36, nonbondedMethod=app.PME)
            sc.initialise_simulation_(timestep_ps = 0.002, # ps
                                    P = 1,               # atm
                                    T = 300,             # K
                                    barostat_type = 2,   # fully felxible
                                    minimise = True,
                                    )
            sc.run_simulation_(50000, 50)
            sc._xyz = tidy_crystal_xyz_(sc.xyz, sc.boxes, n_atoms_mol=sc.n_atoms_mol, ind_rO={self.npt_no_jump_fa_index_used})
            sc.save_simulation_data_('{DIR_main}MM/GAFF_sc{rerun}/{self.name}/data/{self.name}_dataset_{_form}_NPT_cell_{cell}')
            print('finished:', {_form}, {cell})
            del sc
        except:
            # unless all 'finished', check for keyword 'problem' in nohup.out
            print('!! problem in:', {_form}, {cell}) # catch error from cell length sampling smaller than PME_cutoff
        '''
        return py_script # print(py_script) to visualise in JN

    def script_NVT_(self, form, n_mol, test_rerun=False):
        _form = self.form_to_form[form]
        cell = self.cells_by_form_nmol[form][n_mol]
        if test_rerun: rerun = '_rerun'
        else: rerun = ''

        py_script = f'''
        from .{DIR_main.split('/')[-2]}.MM.sc_system import *

        # NVT script (running one long NVT trajectory sampling the metastable state of interest; locally ergodic MD samples):
        # this is part of a loop, but more clear to write here as a single example

        # TODO: show in a seperate JN how the equilibrated_cell
        # '{DIR_main}MM/GAFF_sc{rerun}/{self.name}/{self.name}_{_form}_equilibrated_cell_{cell}.pdb'
        # was extracted from the trajectory '{DIR_main}MM/GAFF_sc{rerun}/{self.name}/data/{self.name}_dataset_{_form}_NPT_cell_{cell}'
        # [NB: this was done differently in MIV Form III]

        print('starting:', {_form}, {cell})
        try:
            PDB_supercell = '{DIR_main}MM/GAFF_sc{rerun}/{self.name}/{self.name}_{_form}_equilibrated_cell_{cell}.pdb'
            sc = SingleComponent_GAFF(PDB = str(PDB_supercell), n_atoms_mol={self.n_atoms_mol}, name={self.name})
            sc.initialise_system_(PME_cutoff=0.36, nonbondedMethod=app.PME)
            sc.initialise_simulation_(timestep_ps = 0.002, # ps
                                    P = None,              # atm
                                    T = 300,               # K
                                    minimise = True,
                                    )
            sc.simulation.step(200*50) # removing warmup, this is not saved
            sc.run_simulation_({self.nvt_nmol_to_nframes_(n_mol)}, 50)
            sc.save_simulation_data_('{DIR_main}MM/GAFF_sc{rerun}/{self.name}/data/{self.name}_dataset_{_form}_NVT_cell_{cell}{self.nvt_key}')
            print('finished:', {_form}, {cell})
            del sc
        except: 
            # unless all 'finished', check for keyword 'problem' in nohup.out
            # just incase timestep too high, but 0.002ps was always fine in these unperterbed simulations.
            print('!! problem in:', {_form}, {cell})
        '''
        return py_script # print(py_script) to visualise in JN

    def script_ECM_(self, form, n_mol):
        _form = self.form_to_form[form]
        cell = self.cells_by_form_nmol[form][n_mol]

        test_rerun = True
        if test_rerun: rerun = 'rerun_'
        else: rerun = ''

        if self.name != 'succinic_acid': key = ''
        else: key = str(self.ecm_key)

        py_script = f'''
        from .{DIR_main.split('/')[-2]}.MM.ecm_basic import *

        # ECM script (getting ground truth FE estimate for the metastable state of interest):
        # this is part of a loop, but more clear to write here as a single example

        try:
            ecm = ECM_basic(
                    name = '{rerun}ecm_{self.name}_{_form}_{cell}_300K_unsupervised_FA{self.fa_index}_{self.ecm_key}',
                    working_dir_folder_name = '{DIR_main}MM/GAFF_sc/ecm_runs/',
                    ARGS_oss = {self.name}_ARGS_oss(form={_form}, cell={cell}, key='{key}_'),
                    k_EC = 6000.0,
                    COM_removal_by_fixing_one_atom_index_of_this_atom = {self.fa_index},
                    overwrite = False,
                    path_lambda_1_dataset = '{DIR_main}MM/GAFF_sc/{self.name}/data/{self.name}_dataset_{_form}_NVT_cell_{cell}{self.nvt_key}',
                    )
            ecm.unsupervised_FE_(
                    batch_size_increments = {self.ecm_batch_size_increments},
                    max_dataset_size_per_lambda = 60000,
                    max_n_lambdas = {self.ecm_max_nlambdas},
                    SE_tol_per_molecule = {self.ecm_SE_tol},
                    )
            ecm.estimate_FE_using_mBAR_()
        except:
            # unless all 'finished', check for keyword 'problem' in nohup.out
            # this is important in a loop here because perterbed simulation can break due timestep 
            # for this reason ecm timestep was four times lower in VEL and MIV (0.0005ps < 0.002ps)
            # Note: all MD frames were saved with the same stride (stride_save_frame = 0.1 / timestep)
            print('!! problem in:', {_form}, {cell})
        '''
        return py_script # print(py_script) to visualise in JN

    def script_PGM_(self, form, n_mol):
        _form = self.form_to_form[form]
        cell = self.cells_by_form_nmol[form][n_mol]

        test_rerun = True
        if test_rerun: rerun = 'rerun_'
        else: rerun = ''

        py_script = f'''
            from .{DIR_main.split('/')[-2]}.interface import *
            
            # PGM script (training a model on the MD data from the long, locally ergodic, NVT trajectory):
            # this is part of a loop, but more clear to write here as a single example

            # this script is training M_/mol version of the model using PGMcrys_v1
            # this exact script implements M_/mol with as slightly adjusted method in M_IC ('focused' on a unitcell rather then a molecule)
            # By default, PGMcrys_v1 can only replicate a fresh instance of: M_/mol and M_/multimap, but not M_/2mol.
            # However, PGMcrys_v1 is able to correctly load ALL of the old model instances mentioned in the paper (inclusing M_/2mol).
            # [These instances include: M_/mol,(2h) , M_/mol,(4h), M_/2mol, and M_/multimap (= M_/multimap,(2h))]
            # This backwards-compatibility with loading older the models is verified in a dedicated jupyter notebook.
            
            nn = NN_interface_sc_multimap(
                name = '{rerun}{self.name}_300K_eqm_box_{_form}_{cell}',
                paths_datasets = ['{DIR_main}MM/GAFF_sc/{self.name}/data/{self.name}_dataset_{form}_NVT_cell_{cell}{self.nvt_key}',],
                running_in_notebook = False, # setting True and running (nn.train) in a seperate JN cell evolves a plot as a function of time
                training = True,             # False only when need to quickly check the result without accessing the MD data xyz coordinates
                model_class = PGMcrys_v1,
                )
            nn.set_ic_map_step1(ind_root_atom={self.pgm_ind_root_atom}, option={self.pgm_option}) # 'option' specifies the other two atoms of the Cartesian block
            nn.set_ic_map_step2(check_PES=True)
            nn.set_ic_map_step3(n_mol_unitcells=[{self.form_to_nmol_unitcell[form]},])

            nn.set_model(n_layers = 4, learning_rate=0.001, n_att_heads={self.pgm_nheads}, evaluation_batch_size=5000)

            nn.set_trainer(n_batches_between_evaluations=50)
            
            n_batches = {self.pgm_nmol_to_ntrainingbatches_(n_mol)}
            print('training for n_batches =',n_batches)
            nn.train(   n_batches = {self.pgm_nmol_to_ntrainingbatches_(n_mol)},    # how many training batches (NB: default training_batch_size is 1000)

                        save_misc = True,     # True; needed later in nn.solve_BAR_using_pymbar_
                        save_BAR = True,      # saves an input for pymbar every 50 training batches (n_batches_between_evaluations=50).
                                              # each input saved has 5000 random points from self.r_validation and 5000 new points from model

                        test_inverse = True,  # extra cost; not necesary, but was True in the paper.
                        # evaluate_on_training_data = True, extra cost; not necesary, but was True in the paper.
                    )

            nn.load_misc_()
            nn.solve_BAR_using_pymbar_(rerun=True)
            nn.save_samples_(20000)
            nn.save_model_()

            print('finished:', {_form}, {cell})
            del nn

        '''
        return py_script # print(py_script) to visualise in JN

    ## ## ## ##

    def ECM_name_template(self, form, n_mol):
        _form = self.form_to_form[form]
        cell = self.cells_by_form_nmol[form][n_mol]
        return f"{self.name}_{_form}_{cell}_300K_unsupervised_FA{self.fa_index}_{self.ecm_key}"

    def path_ECM_inds_rand(self, form, n_mol):
        return DIR_main+f'/MM/GAFF_sc/ecm_runs/ecm__{self.ECM_name_template(form, n_mol)}__inds_rand_lambda1_60000'

    def path_ECM_log_eval(self, form, n_mol):
        return DIR_main+f'/MM/GAFF_sc/ecm_runs/ecm__{self.ECM_name_template(form, n_mol)}__lambda_evaluations'

    def path_ECM_log_BAR(self, form, n_mol):
        return DIR_main+f'/MM/GAFF_sc/ecm_runs/ecm__{self.ECM_name_template(form, n_mol)}__log_of_BAR_results'

    def path_ECM_log_MBAR(self, form, n_mol):
        return DIR_main+f'/MM/GAFF_sc/ecm_runs/ecm__{self.ECM_name_template(form, n_mol)}__log_of_mBAR_results'

    ## ## 

    def ecm_latt_FE_SE_u_mean_BAR(self, form, n_mol):
        res = load_pickle_(self.path_ECM_log_BAR(form, n_mol))
        ' final estimates using BAR (all data used) '
        FE, SE = res[-1][:2]
        u_mean = res[-1][-2][-1][1]
        return FE/n_mol, SE/n_mol, u_mean/n_mol

    def tabulate_ecm_BAR_result_(self, verbose=True):
        # pupulating these for FE differences
        self.FE_ecm_BAR = np.zeros([self.n_forms, len(self.list_n_mol)]) + np.nan
        self.SE_ecm_BAR = np.zeros([self.n_forms, len(self.list_n_mol)]) + np.nan

        self.u_mean     = np.zeros([self.n_forms, len(self.list_n_mol)]) + np.nan
        self.u_mean_by_form_nmol = {}
        self.S_ecm_BAR  = np.zeros([self.n_forms, len(self.list_n_mol)]) + np.nan

        ##
        self.print('ECM lattice FE estimates based on BAR:\n', verbose=verbose)
        string_ = lambda **x : "{a:<10} | {b:<10} | {c:<10} | {d:<10} | {e:<10} || {f:<10}".format(**x)
        self.print(string_(a='form', b='n_mol', c='cell', d='FE /kT', e='SE /kT', f='S /k'), verbose=verbose)
        self.print(string_(a='='*10, b='='*10, c='='*10, d='='*10, e='='*10, f='='*10), verbose=verbose)

        round_dp = 3

        a = 0
        for form in self.forms:
            b = 0
            for n_mol in self.list_n_mol:
                try: 
                    FE, SE, u_mean = self.ecm_latt_FE_SE_u_mean_BAR(form=form,n_mol=n_mol)
                    S = u_mean - FE
                    self.print(string_(a=form, b=n_mol, 
                                       c=self.cells_by_form_nmol[form][n_mol], 
                                       d= np.round(FE,round_dp), e=np.round(SE,round_dp),
                                       f=np.round(S,round_dp)), verbose=verbose)
                    #plt.scatter([n_mol], [FE], color=self.form_to_color[form])
                    self.FE_ecm_BAR[a,b] = FE
                    self.SE_ecm_BAR[a,b] = SE
                    self.u_mean[a,b]     = u_mean
                    self.S_ecm_BAR[a,b]  = S
                    self.u_mean_by_form_nmol[(form, n_mol)] = u_mean
                except: pass
                b += 1
            self.print(string_(a='-'*10, b='-'*10, c='-'*10, d='-'*10, e='-'*10, f='-'*10), verbose=verbose)
            a += 1 
        self.print('', verbose=verbose)

    def ecm_latt_FE_SE_MBAR(self, form, n_mol):
        res = load_pickle_(self.path_ECM_log_MBAR(form, n_mol))
        ' final estimates using MBAR (all data used; evaluated only once at the end of the ecm run) '
        FE = (res[-4] + res[0])
        SE = res[-3]
        return FE/n_mol, SE/n_mol

    def tabulate_ecm_MBAR_result_(self, verbose=True):
        # pupulating these for FE differences
        self.FE_ecm_MBAR = np.zeros([self.n_forms, len(self.list_n_mol)]) + np.nan
        self.SE_ecm_MBAR = np.zeros([self.n_forms, len(self.list_n_mol)]) + np.nan

        self.S_ecm_MBAR  = np.zeros([self.n_forms, len(self.list_n_mol)]) + np.nan

        self.print('ECM lattice FE estimates based on MBAR:\n', verbose=verbose)
        string_ = lambda **x : "{a:<10} | {b:<10} | {c:<10} | {d:<10} | {e:<10} || {f:<10}".format(**x)
        self.print(string_(a='form', b='n_mol', c='cell', d='FE /kT', e='SE /kT', f='S /k'), verbose=verbose)
        self.print(string_(a='='*10, b='='*10, c='='*10, d='='*10, e='='*10, f='='*10), verbose=verbose)

        round_dp = 3

        a = 0
        for form in self.forms:
            b = 0
            for n_mol in self.list_n_mol:
                try:
                    FE, SE = self.ecm_latt_FE_SE_MBAR(form=form,n_mol=n_mol)
                    S = self.u_mean[a,b] - FE
                    self.print(string_(a=form, b=n_mol, 
                                       c=self.cells_by_form_nmol[form][n_mol], 
                                       d= np.round(FE,round_dp), e=np.round(SE,round_dp),
                                       f=np.round(S,round_dp)), verbose=verbose)
                    #plt.scatter([n_mol], [FE], color=self.form_to_color[form])
                    self.FE_ecm_MBAR[a,b] = FE
                    self.SE_ecm_MBAR[a,b] = SE
                    self.S_ecm_MBAR[a,b]   = S
                except: pass
                b += 1
            self.print(string_(a='-'*10, b='-'*10, c='-'*10, d='-'*10, e='-'*10, f='-'*10), verbose=verbose)
            a += 1 
        # self.S_ecm_MBAR = self.u_mean - self.FE_ecm_MBAR
        self.print('', verbose=verbose)

    def plot_SI_ecm_figure_(self,):
        fig = plt.figure(figsize=(5,4), dpi=300)

        if self.molecule_abbreviation == 'SA':
            window_x = [-5,160]
            window_y = [-19.45, -18.3]
            line2_until = 160
        elif self.molecule_abbreviation == 'VEL':
            window_x = [-5*(100/160),100]
            window_y = [355,360]
            line2_until = 100
        else:
            window_x = [-5*(100/160),100]
            window_y = [375+0.5, 384-0.5]
            line2_until = 100

        a = 0
        for form in self.forms:
            b = 0
            color = self.form_to_color[form]
            for n_mol in self.list_n_mol:
                try:
                    res = load_pickle_(self.path_ECM_log_BAR(form, n_mol))
                    n_lambdas = len(np.array(res[-1][-2][:,0]))
                    when_n_lambdas_reached = np.where(np.array([len(x[-1]) for x in res]) == n_lambdas)[0][0]
                    FEs_BAR = np.array([x[0] for x in res])/n_mol
                    SEs_BAR = np.array([x[1] for x in res])/n_mol

                    plt.plot(np.arange(len(FEs_BAR)), FEs_BAR, color=color)
                    plt.fill_between(np.arange(len(FEs_BAR)), FEs_BAR-SEs_BAR, FEs_BAR+SEs_BAR, alpha=0.4, color=color)

                    FE_mBAR = self.FE_ecm_MBAR[a,b]
                    SE_mBAR = self.SE_ecm_MBAR[a,b]
                    line_until = len(FEs_BAR)
                    plt.plot([-5,line_until], [FE_mBAR]*2, color=color)
                    plt.plot([-5,line_until], [FE_mBAR-SE_mBAR]*2, color=color, linestyle='dotted')
                    plt.plot([-5,line_until], [FE_mBAR+SE_mBAR]*2, color=color, linestyle='dotted')

                    plt.plot([line2_until-5,line2_until], [FE_mBAR]*2, color=color)
                    plt.plot([line2_until-5,line2_until], [FE_mBAR-SE_mBAR]*2, color=color, linestyle='dotted')
                    plt.plot([line2_until-5,line2_until], [FE_mBAR+SE_mBAR]*2, color=color, linestyle='dotted')
                    plt.scatter([when_n_lambdas_reached ],[FE_mBAR], color='black', zorder=10, s=15)

                    res_MBAR = load_pickle_(self.path_ECM_log_MBAR(form, n_mol))
                    print(form, n_mol)
                    print('n_lambdas (reached at black dot):', n_lambdas, '=', res_MBAR[-1].shape[0])
                    n_MD_frames = np.array([x for x in res_MBAR[3].values()]).sum()
                    print('number of MD frames (datapoints); total (',n_MD_frames,'), total/n_lambdas (',np.round(n_MD_frames/n_lambdas, 3),')')
                except: pass
                b += 1
            a +=1

        plt.xlim(*window_x)
        plt.ylim(*window_y)
        plt.xlabel('progress of ECM simulations')
        plt.ylabel('lattice FE / kT')
        plt.show()

    ## ## ## ##

    def NVT_dataset_name(self, form, n_mol):
        _form = self.form_to_form[form]
        cell = self.cells_by_form_nmol[form][n_mol]
        return f'{self.name}_dataset_{_form}_NVT_cell_{cell}{self.nvt_key}'
    
    def path_NVT_dataset_full(self, form, n_mol):
        return DIR_main+f'/MM/GAFF_sc/{self.name}/data/{self.NVT_dataset_name(form, n_mol)}'

    def path_NVT_dataset_CUT(self, form, n_mol):
        # make
        return DIR_main+f'/MM/GAFF_sc/{self.name}/data/{self.NVT_dataset_name(form, n_mol)}_CUT'

    def _check_paths_NVT_dataset_full_exist_(self,):
        all_there = True
        for [form, cell, n_mol] in self.form_cell_n_mol_groups:
            exists = os.path.exists(self.path_NVT_dataset_full(form,n_mol))
            if exists: pass
            else: all_there = False
        return all_there
    
    def _check_paths_ECM_inds_rand_exist_(self,):
        all_there = True
        for [form, cell, n_mol] in self.form_cell_n_mol_groups:
            exists = os.path.exists(self.path_ECM_inds_rand(form,n_mol))
            if exists: pass
            else: all_there = False
        return all_there
    
    def _CUT_NVT_datasets_to_ruduce_file_size_and_make_loading_during_evalation_faster_(self,):
        ' the whole FEcrys folder is ~200GB in size after keeping only the arrays from paper '
        ' this excluding ALL of the ECM xyz data (except lambda 1.0) (which is being cut out here) '
        ' this cutting step will hopefully reduce the file size to something smaller '
        assert self._check_paths_NVT_dataset_full_exist_()
        assert self._check_paths_ECM_inds_rand_exist_()

        for [form, cell, n_mol] in self.form_cell_n_mol_groups:
            save_as = self.path_NVT_dataset_CUT(form, n_mol)
            if not os.path.exists(save_as):
                # these indices exist from being used in ECM, to do a similar thing, so can use them here also:
                inds_keep = load_pickle_(self.path_ECM_inds_rand(form, n_mol)) ; m = 60000 
                nvt_dataset = load_pickle_(self.path_NVT_dataset_full(form, n_mol))
                # 'MD dataset' : 'xyz', 'COMs', 'b', 'u', 'T', 'rbv', 'stride_save_frame'
                # only cutting the arrays ('xyz', 'COMs', 'b', 'u'), leaving 'T' to remember length.
                nvt_dataset['MD dataset']['u_full'] = np.array(nvt_dataset['MD dataset']['u']) # backup
                nvt_dataset['MD dataset']['xyz'] = nvt_dataset['MD dataset']['xyz'][inds_keep][:m]
                nvt_dataset['MD dataset']['COMs'] = nvt_dataset['MD dataset']['COMs'][inds_keep][:m]
                nvt_dataset['MD dataset']['b'] = nvt_dataset['MD dataset']['b'][inds_keep][:m]
                nvt_dataset['MD dataset']['u'] = nvt_dataset['MD dataset']['u'][inds_keep][:m]
                save_pickle_(nvt_dataset, save_as)
                del nvt_dataset
            else: print('already exists:',save_as)

    ## ## ## ##

    def initialise_PGM_result_evalaution_(self,):

        self.nn_evaluation = {}
        
        if self.molecule_abbreviation == 'SA':
            '''
            loading succinic acid pgm results based on /mol and /2mol versions of the model
            '''
            for [form, cell, n_mol] in self.form_cell_n_mol_groups:
                _form = self.form_to_form[form]
                if form == 'G':
                    nn = NN_interface_sc_multimap(
                        name = f'{self.name}_300K_eqm_box_{_form}_{cell}',
                        paths_datasets = [self.u_mean_by_form_nmol[(form,n_mol)]*n_mol], #[self.path_NVT_dataset_CUT(form,n_mol)],
                        running_in_notebook = True, training = False, #model_class = PGMcrys_v1,
                        )
                    self.nn_evaluation[(form, n_mol, '/2mol')] = nn
                    nn = NN_interface_sc_multimap(
                        name = f'{self.name}_300K_eqm_box_{_form}_{cell}_pm',
                        paths_datasets = [self.u_mean_by_form_nmol[(form,n_mol)]*n_mol], #[self.path_NVT_dataset_CUT(form, n_mol)],
                        running_in_notebook = True, training = False, #model_class = PGMcrys_v1,
                        )
                    self.nn_evaluation[(form, n_mol, '/mol')] = nn
                else:
                    nn = NN_interface_sc_multimap(
                        name = f'{self.name}_300K_eqm_box_{_form}_{cell}',
                        paths_datasets = [self.u_mean_by_form_nmol[(form,n_mol)]*n_mol], #[self.path_NVT_dataset_CUT(form, n_mol)],
                        running_in_notebook = True, training = False,# model_class = PGMcrys_v1,
                        )
                    self.nn_evaluation[(form, n_mol, '/mol')] = nn
            '''
            loading succinic acid pgm results based on /mol,multimap version of the model
            '''
            forms_model1 = [       'B',   'G'   ]
            cells_model1 = [       '212', '211' ]

            forms_model2 = ['A',   'B',   'G'   ]
            cells_model2 = ['222', '222', '221' ]

            forms_model3 = ['A',   'B',   'G'   ]
            cells_model3 = ['223', '223', '321' ]

            forms_model4 = ['A',   'B',   'G'   ]
            cells_model4 = ['224', '224', '222' ]

            form_groups = [forms_model1, forms_model2, forms_model3, forms_model4]
            cell_groups = [cells_model1, cells_model2, cells_model3, cells_model4]
            
            for i in range(4):
                forms = form_groups[i]
                cells = cell_groups[i]
                n_mol = self.list_n_mol[i]
                _forms = [self.form_to_form[x] for x in forms]
                nn = NN_interface_sc_multimap(
                    name = f'{self.name}_300K_eqm_box_'+'_'.join(_forms)+'_'+'_'.join(cells),

                    #paths_datasets =[DIR_main+f'/MM/GAFF_sc/{self.name}/data/{self.name}_dataset_'+_form+'_NVT_cell_'+cell+'_CUT' for _form, cell in zip(_forms, cells)],
                    paths_datasets= [self.u_mean_by_form_nmol[(form,n_mol)]*n_mol for form in forms],
                    
                    running_in_notebook = True, training = False, #model_class = PGMcrys_v1,
                    )
                self.nn_evaluation[('forms', self.list_n_mol[i], '/mol,multimap')] = nn

                #for form in forms:
                #    nn_copy = copy.deepcopy(nn)
                #    nn_copy.nns = [dict(zip(forms,nn_copy.nns))[form]]
                #    nn_copy.n_crystals = 1
                #    self.nn_evaluation[(form, self.list_n_mol[i], '/mol,multimap')] = nn_copy

        elif self.molecule_abbreviation == 'VEL':
            '''
            loading veliparib pgm results based on /mol model
            '''
            for [form, cell, n_mol] in self.form_cell_n_mol_groups:
                _form = self.form_to_form[form]
                nn = NN_interface_sc_multimap(
                    name = f'{self.name}_300K_eqm_box_{_form}_{cell}',
                    paths_datasets = [self.u_mean_by_form_nmol[(form,n_mol)]*n_mol], #[self.path_NVT_dataset_CUT(form, n_mol)],
                    running_in_notebook = True, training = False, #model_class = PGMcrys_v1,
                    )
                self.nn_evaluation[(form, n_mol, '/mol')] = nn
        else:
            '''
            loading veliparib pgm results based on /mol,2h and /mol,4h versions of the model
            '''
            for [form, cell, n_mol] in self.form_cell_n_mol_groups:
                _form = self.form_to_form[form]
                
                if form in ['IIIs','IIIb'] and n_mol == 16: key = '_ext'
                else: key = ''
                #'''
                nn = NN_interface_sc_multimap(
                    name = f'{self.name}_300K_eqm_box_{_form}_{cell}'+key,
                    paths_datasets = [self.u_mean_by_form_nmol[(form,n_mol)]*n_mol], #[self.path_NVT_dataset_CUT(form, n_mol)],
                    running_in_notebook = True, training = False, #model_class = PGMcrys_v1,
                    )
                self.nn_evaluation[(form, n_mol, '/mol,2h')] = nn
                #'''
                key = '_alter'
                nn = NN_interface_sc_multimap(
                    name = f'{self.name}_300K_eqm_box_{_form}_{cell}'+key,
                    paths_datasets = [self.u_mean_by_form_nmol[(form,n_mol)]*n_mol], #[self.path_NVT_dataset_CUT(form, n_mol)],
                    running_in_notebook = True, training = False, #model_class = PGMcrys_v1,
                    )
                self.nn_evaluation[(form, n_mol, '/mol,4h')] = nn

        ## 

        for key in self.nn_evaluation.keys():
            nn = self.nn_evaluation[key]
            nn.load_misc_()
            nn.load_energies_during_training_()
            nn.solve_BAR_using_pymbar_()
            nn.load_inv_test_results_()

        self.pgm_versions_involved = list(set([x[-1] for x in self.nn_evaluation.keys()]))
        print(self.name, 'pgm results form model versions', self.pgm_versions_involved, 'were imported')

        n = len(self.pgm_versions_involved) # !! [np.zeros(...)]*n is 'leaking' to all keys (TODO: carefull)
        self.FE_pgm_BAR = dict(zip(self.pgm_versions_involved,[np.zeros([self.n_forms, len(self.list_n_mol)]) + np.nan for _ in range(n)]))
        self.SD_pgm_BAR = dict(zip(self.pgm_versions_involved,[np.zeros([self.n_forms, len(self.list_n_mol)]) + np.nan for _ in range(n)]))
        self.SE_pgm_BAR = dict(zip(self.pgm_versions_involved,[np.zeros([self.n_forms, len(self.list_n_mol)]) + np.nan for _ in range(n)]))
        self.S_pgm_BAR = dict(zip(self.pgm_versions_involved,[np.zeros([self.n_forms, len(self.list_n_mol)]) + np.nan for _ in range(n)]))
        self.n_pgm_versions_involved = n

        for model_abbreviation in self.pgm_versions_involved:
            self.tabulate_pgm_result_by_model_(model_abbreviation, verbose=False)

    ## ## 

    def tabulate_pgm_result_by_model_(self, model_abbreviation=0, verbose=True):
        if type(model_abbreviation) is int: model_abbreviation = self.pgm_versions_involved[model_abbreviation]
        else: pass

        self.print('PGM (mode version '+model_abbreviation+') lattice FE estimates:\n', verbose=verbose)
        string_ = lambda **x : "{a:<10} | {b:<10} | {c:<10} | {d:<10} | {e:<10} | {f:<10} || {g:<10}".format(**x)
        self.print(string_(a='form', b='n_mol', c='cell', d='FE /kT', e='SE /kT', f='SD /kT', g='S /k'), verbose=verbose)
        self.print(string_(a='='*10, b='='*10, c='='*10, d='='*10, e='='*10, f='='*10, g='='*10), verbose=verbose)

        round_dp = 3

        if model_abbreviation != '/mol,multimap':

            a = 0
            for form in self.forms:
                b = 0
                for n_mol in self.list_n_mol:
                    try:
                        FE = self.nn_evaluation[(form, n_mol, model_abbreviation)].nns[0].BAR_V_FE/n_mol
                        SE = self.nn_evaluation[(form, n_mol, model_abbreviation)].nns[0].BAR_V_SE/n_mol
                        SD = self.nn_evaluation[(form, n_mol, model_abbreviation)].nns[0].BAR_V_SD/n_mol
                        S = self.u_mean[a,b] - FE
                        
                        self.print(string_(a=form, b=n_mol, 
                                        c=self.cells_by_form_nmol[form][n_mol], 
                                        d= np.round(FE,round_dp), e=np.round(SE,round_dp),
                                        f=np.round(SD,round_dp), g=np.round(S,round_dp)), verbose=verbose)
                        self.FE_pgm_BAR[model_abbreviation][a,b] = FE
                        self.SD_pgm_BAR[model_abbreviation][a,b] = SD
                        self.SE_pgm_BAR[model_abbreviation][a,b] = SE
                        self.S_pgm_BAR[model_abbreviation][a,b] = S
                    except: pass
                    b += 1
                self.print(string_(a='-'*10, b='-'*10, c='-'*10, d='-'*10, e='-'*10, f='-'*10, g='-'*10), verbose=verbose)
                a += 1 
            
        else:
            ' TODO: this was annoying, change something.. '

            b = 0
            nmol_to_inds_forms = {8:[1,2],16:[0,1,2],24:[0,1,2],32:[0,1,2]}
            for n_mol in self.list_n_mol:
                nn_B8_G8 = self.nn_evaluation[('forms', n_mol, model_abbreviation)]
                inds_forms = nmol_to_inds_forms[n_mol]
                for i in range(len(inds_forms)):
                    a = inds_forms[i]
                    form = self.forms[a]
                    FE = nn_B8_G8.nns[i].BAR_V_FE/n_mol
                    SE = nn_B8_G8.nns[i].BAR_V_SE/n_mol
                    SD = nn_B8_G8.nns[i].BAR_V_SD/n_mol
                    S = self.u_mean[a,b] - FE
                    #self.print(string_(a=form, b=n_mol, 
                    #                c=self.cells_by_form_nmol[form][n_mol], 
                    #                d= np.round(FE,round_dp), e=np.round(SE,round_dp),
                    #                f=np.round(SD,round_dp), g=np.round(S,round_dp)), verbose=verbose)
                    self.FE_pgm_BAR[model_abbreviation][a,b] = FE
                    self.SD_pgm_BAR[model_abbreviation][a,b] = SD
                    self.SE_pgm_BAR[model_abbreviation][a,b] = SE
                    self.S_pgm_BAR[model_abbreviation][a,b] = S
                b += 1
                #self.print(string_(a='-'*10, b='-'*10, c='-'*10, d='-'*10, e='-'*10, f='-'*10, g='-'*10), verbose=verbose)
            if verbose:
                a = 0
                for form in self.forms:
                    b = 0
                    for n_mol in self.list_n_mol:
                        try:
                            FE = self.FE_pgm_BAR[model_abbreviation][a,b]
                            SD = self.SD_pgm_BAR[model_abbreviation][a,b]
                            SE = self.SE_pgm_BAR[model_abbreviation][a,b]
                            S = self.S_pgm_BAR[model_abbreviation][a,b]
                            self.print(string_(a=form, b=n_mol, 
                                            c=self.cells_by_form_nmol[form][n_mol], 
                                            d= np.round(FE,round_dp), e=np.round(SE,round_dp),
                                            f=np.round(SD,round_dp), g=np.round(S,round_dp)), verbose=verbose)
                        except: pass
                        b += 1
                    self.print(string_(a='-'*10, b='-'*10, c='-'*10, d='-'*10, e='-'*10, f='-'*10, g='-'*10), verbose=verbose)
                    a += 1 
            else: pass

        self.print('', verbose=verbose)

    def plot_maintext_plot_by_model_(self, model_abbreviation=0):
        if type(model_abbreviation) is int: model_abbreviation = self.pgm_versions_involved[model_abbreviation]
        else: pass

        fig, ax = plt.subplots(1,2, figsize=(10,5))

        x_ax = np.array(self.list_n_mol)

        fig.suptitle('colors = Forms ; model version: '+model_abbreviation)

        for a in [0,1]:
            for i in range(self.n_forms):
                inds_finite = np.where(np.isfinite(self.FE_ecm_BAR[i]))[0]
                _x_ax = np.array(x_ax[inds_finite])
                if a == 0:
                    ys_ecm_mbar = np.array(self.FE_ecm_MBAR[i])[inds_finite]
                    ys_ecm_bar = np.array(self.FE_ecm_BAR[i])[inds_finite]
                    ys_pgm_bar = np.array(self.FE_pgm_BAR[model_abbreviation][i])[inds_finite]
                    title = 'absolute lattice free energy /kT'
                    y_label = 'f / kT'
                else:
                    ys_ecm_mbar = np.array(self.S_ecm_MBAR[i])[inds_finite]
                    ys_ecm_bar = np.array(self.S_ecm_BAR[i])[inds_finite]
                    ys_pgm_bar = np.array(self.S_pgm_BAR[model_abbreviation][i])[inds_finite]
                    title = 'absolute lattice entropy / k'
                    y_label = 's / kT'
                    
                SE_ecm_mbar = np.array(self.SE_ecm_MBAR[i])[inds_finite]
                SE_ecm_bar = np.array(self.SE_ecm_BAR[i])[inds_finite]
        
                ax[a].fill_between(_x_ax, ys_ecm_mbar-SE_ecm_mbar, ys_ecm_mbar+SE_ecm_mbar, color='black', alpha=1.0, zorder=100)
                ax[a].plot(_x_ax, ys_ecm_bar+SE_ecm_bar, color='black', linestyle='dotted', linewidth=1.8, zorder=100)
                ax[a].plot(_x_ax, ys_ecm_bar-SE_ecm_bar, color='black', linestyle='dotted', linewidth=1.8, zorder=100)
                
                form = self.forms[i]
                color = self.form_to_color[form]
                SE_pgm = np.array(self.SE_pgm_BAR[model_abbreviation][i])[inds_finite]
                
                ax[a].fill_between(_x_ax,ys_pgm_bar-SE_pgm, ys_pgm_bar+SE_pgm, color=color, alpha=0.8, zorder=50)
                ax[a].plot(_x_ax , ys_pgm_bar, color=color, zorder=50, label=form)
        
            ax[a].set_title(title)
            ax[a].set_ylabel(y_label, size=13)
            ax[a].set_xlabel('n_mol', size=13)
            ax[a].set_xticks(x_ax)

        plt.tight_layout()
        #ax[a].legend(loc="lower right")

    def plot_SI_pgm_figure_by_model_(self, model_abbreviation=0):
        if type(model_abbreviation) is int: model_abbreviation = self.pgm_versions_involved[model_abbreviation]
        else: pass

        fig, ax = plt.subplots(self.n_forms, len(self.list_n_mol), figsize=(10,4), dpi=150)

        fig.suptitle('lattice FE / kT ;\n rows : Forms ;\n columns : n_mol ;\n model version: '+model_abbreviation, y=1.15, fontsize=8*1.2)

        form_to_a = dict(zip(self.forms, np.arange(self.n_forms)))
        nmol_to_b = dict(zip(self.list_n_mol, np.arange(len(self.list_n_mol))))

        if self.molecule_abbreviation == 'SA':    window_y = [-19.45, -18.3]
        elif self.molecule_abbreviation == 'VEL': window_y = [355,360]
        else:                                     window_y = [375+0.5, 384-0.5]

        for key in self.nn_evaluation.keys():

            if key[-1] == model_abbreviation and key[0] != 'forms':
                form = key[0]
                n_mol = key[1]
                color = self.form_to_color[form]
                a = form_to_a[form]
                b = nmol_to_b[n_mol]
                
                nn = self.nn_evaluation[key].nns[0]
                
                ax[a,b].plot(nn.evaluation_grid, nn.BAR_V_FEs/n_mol, color='red', zorder=10, linewidth=1)
                ax[a,b].plot(nn.evaluation_grid, nn.BAR_V_FEs/n_mol + nn.BAR_V_SEs/n_mol, color='red', zorder=10, linestyle='dotted')
                ax[a,b].plot(nn.evaluation_grid, nn.BAR_V_FEs/n_mol - nn.BAR_V_SEs/n_mol, color='red', zorder=10, linestyle='dotted')
                ax[a,b].plot(nn.evaluation_grid, nn.BAR_V_FEs_raw/n_mol, color='black', zorder=9, linewidth=0.5)
                ax[a,b].fill_between(nn.evaluation_grid,
                                    nn.BAR_V_FEs_raw/n_mol-nn.BAR_V_SEs_raw/n_mol, 
                                    nn.BAR_V_FEs_raw/n_mol+nn.BAR_V_SEs_raw/n_mol,
                                    color='C0', zorder=1, linewidth=0.5, alpha=0.6)
                
                ax[a,b].set_ylim(*window_y)
                #ax[a,b].set_yticks(window_y)
                ax[a,b].set_xlabel('training \n batches', size=8)
                ax[a,b].set_title('form:'+str(form)+';n_mol:'+str(n_mol), size=8)
                ax[a,b].set_xticks([nn.evaluation_grid[0]-49, nn.evaluation_grid[-1]+1])
                ax[a,b].tick_params(axis='y', labelsize=8)
                ax[a,b].tick_params(axis='x', labelsize=8)
                #ax[a,b].set_ylabel('f /kT')
            else: pass

            if key[-1] == model_abbreviation and key[0] == 'forms':
                if len(self.nn_evaluation[key].nns) == 2: this = {0:'B',1:'G'}
                else:                                     this = {0:'A',1:'B',2:'G'}
                n_mol = key[1]
                for i in range(len(self.nn_evaluation[key].nns)):
                    form = this[i]
                    color = self.form_to_color[form]
                    a = form_to_a[form]
                    b = nmol_to_b[n_mol]

                    nn = self.nn_evaluation[key].nns[i]
                
                    ax[a,b].plot(nn.evaluation_grid, nn.BAR_V_FEs/n_mol, color='red', zorder=10, linewidth=1)
                    ax[a,b].plot(nn.evaluation_grid, nn.BAR_V_FEs/n_mol + nn.BAR_V_SEs/n_mol, color='red', zorder=10, linestyle='dotted')
                    ax[a,b].plot(nn.evaluation_grid, nn.BAR_V_FEs/n_mol - nn.BAR_V_SEs/n_mol, color='red', zorder=10, linestyle='dotted')
                    ax[a,b].plot(nn.evaluation_grid, nn.BAR_V_FEs_raw/n_mol, color='black', zorder=9, linewidth=0.5)
                    ax[a,b].fill_between(nn.evaluation_grid,
                                        nn.BAR_V_FEs_raw/n_mol-nn.BAR_V_SEs_raw/n_mol, 
                                        nn.BAR_V_FEs_raw/n_mol+nn.BAR_V_SEs_raw/n_mol,
                                        color='C0', zorder=1, linewidth=0.5, alpha=0.6)
                    
                    ax[a,b].set_ylim(*window_y)
                    #ax[a,b].set_yticks(window_y)
                    ax[a,b].set_xlabel('training \n batches', size=8)
                    ax[a,b].set_title('form:'+str(form)+';n_mol:'+str(n_mol), size=8)
                    ax[a,b].set_xticks([nn.evaluation_grid[0]-49, nn.evaluation_grid[-1]+1])
                    ax[a,b].tick_params(axis='y', labelsize=8)
                    ax[a,b].tick_params(axis='x', labelsize=8)
                    #ax[a,b].set_ylabel('f /kT')
            else: pass

        for i in range(len(ax)):
            for j in range(len(ax[0])):
                if ' 0' in str(ax[i,j].lines):
                    ax[i,j].set_xticks([])
                    ax[i,j].set_yticks([])
                    ax[i,j].spines['top'].set_visible(False)
                    ax[i,j].spines['right'].set_visible(False)
                    ax[i,j].spines['bottom'].set_visible(False)
                    ax[i,j].spines['left'].set_visible(False)
                else: pass
        #plt.tight_layout()
        plt.show()

    def plot_SI_pgm_energy_overlaps_(self, no_T=False):
        #if type(model_abbreviation) is int: model_abbreviation = self.pgm_versions_involved[model_abbreviation]
        #else: pass
        def plot_this_(nn, ax=None, title='', first_last=[None,None], linewidth=1, color=None):
                    if no_T: high = np.percentile(nn.MD_energies_V, 99.9)
                    else: high = np.percentile(np.concatenate([nn.MD_energies_V, nn.MD_energies_T]), 99.9) # nn.u
                    mean = np.array(nn.u_mean)
                    percentile = high - mean
                    low = mean - percentile
                    high += percentile*2

                    colors = plt.cm.jet(np.linspace(0,1,nn.n_estimates))
                    import matplotlib as mpl
                    custom_cmap = mpl.colors.ListedColormap(colors)
                    state = 0

                    first, last = first_last
                    time = np.arange(nn.n_estimates)
                    alphas = np.array(time)*0.0
                    if last is not None:  
                        _time = np.array(time[-last:])
                        alphas[-last:] = 1.0
                    else:                 
                        _time = np.array(time)
                        alphas += 1.0
                    
                    if first is not None: 
                        _time = np.concatenate([time[:first], _time])
                        alphas[:first] = 0.2
                    else: pass

                    for i in _time:
                        if color is None:
                            plot_1D_histogram_(nn.BG_energies[state,i], density=False, linewidth = linewidth,
                                            range=[low,high], color=custom_cmap.colors[i], bins=40, ax=ax, alpha = alphas[i])
                        else:
                            plot_1D_histogram_(nn.BG_energies[state,i], density=False, linewidth = linewidth,
                                            range=[low,high], color=color, bins=40, ax=ax, alpha = alphas[i])
                    
                    for i in range(nn.n_estimates):
                        if not no_T: plot_1D_histogram_(nn.MD_energies_T[state,i], density=False, range=[low,high], color='black', bins=40,linewidth=1, ax=ax)
                        plot_1D_histogram_(nn.MD_energies_V[state,i], density=False, range=[low,high], color='black', bins=40,linewidth=1, ax=ax)

                    
                    title = title#+'training batches:\n 0 (blue) to '+str(nn.evaluation_grid[-1]+1)+' (red)'
                    if ax is None: plt.title(title, fontsize=10)
                    else: ax.set_title(title, fontsize=10)

                    ax.set_xlabel('potential energy /kT', size=8)
                    ax.set_ylabel('population')
                        
                    #ax.set_xticks([low, mean, high])
                    #ax.tick_params(axis='x', rotation=45)
                    ax.set_xticks([mean])
                    return None
        
        fig, ax = plt.subplots(self.n_forms, len(self.list_n_mol), figsize=(len(self.list_n_mol)*2,self.n_forms*2), dpi=150)
        fig.suptitle('energy overlaps ;\n rows : Forms ;\n columns : n_mol;\n  color (black) : MD data;\n color (not black) : version of model (samples from the model);\n color intensity : training progress', fontsize=8*1.2)

        form_to_a = dict(zip(self.forms, np.arange(self.n_forms)))
        nmol_to_b = dict(zip(self.list_n_mol, np.arange(len(self.list_n_mol))))

        if self.n_pgm_versions_involved == 3:
            colors = ['deepskyblue','green','red']
        elif self.n_pgm_versions_involved == 2:
            colors = ['blue','red']
        elif self.n_pgm_versions_involved == 1:
            colors = ['red']
        else: print('!')

        j = 0
        for model_abbreviation in self.pgm_versions_involved:
            for key in self.nn_evaluation.keys():

                if key[-1] == model_abbreviation and key[0] != 'forms':
                    form = key[0]
                    n_mol = key[1]
                    #color = self.form_to_color[form]
                    a = form_to_a[form]
                    b = nmol_to_b[n_mol]
                    
                    nn = self.nn_evaluation[key].nns[0]
                    
                    plot_this_(nn, 
                            ax = ax[a,b],
                            title = 'form:'+str(form)+' ; n_mol:'+str(n_mol), 
                            first_last = [10,10],
                            linewidth=1, 
                            color=colors[j])
                else: pass

                if key[-1] == model_abbreviation and key[0] == 'forms':
                    if len(self.nn_evaluation[key].nns) == 2: this = {0:'B',1:'G'}
                    else:                                     this = {0:'A',1:'B',2:'G'}
                    n_mol = key[1]
                    for i in range(len(self.nn_evaluation[key].nns)):
                        form = this[i]
                        #color = self.form_to_color[form]
                        a = form_to_a[form]
                        b = nmol_to_b[n_mol]

                        nn = self.nn_evaluation[key].nns[i]
                        
                        plot_this_( nn, 
                                    ax = ax[a,b],
                                    title = 'form:'+str(form)+' ; n_mol:'+str(n_mol), 
                                    first_last = [10,10],
                                    linewidth=1, 
                                    color=colors[j])
                else: pass
            j += 1

        for i in range(len(ax)):
            for j in range(len(ax[0])):
                if ' 0' in str(ax[i,j].lines):
                    ax[i,j].set_xticks([])
                    ax[i,j].set_yticks([])
                    ax[i,j].spines['top'].set_visible(False)
                    ax[i,j].spines['right'].set_visible(False)
                    ax[i,j].spines['bottom'].set_visible(False)
                    ax[i,j].spines['left'].set_visible(False)
                else: pass

        plt.tight_layout()
        plt.show()



