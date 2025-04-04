from . import DIR_main

''' util_np.py

saving/loading anything:
    f : save_pickle_
    f : load_pickle_

numpy array reshaping (single component systems only):
    f : reshape_to_molecules_np_
    f : reshape_to_atoms_np_
    f : reshape_to_flat_np_

misc:
    f : cumulative_average_
    f : sta_array_
    f : half_way_
    f : take_random_
    f : joint_grid_from_marginal_grids_

no-jump (molecules already whole):
    f : tidy_crystal_xyz_

'''
import sys

import os
import re
import copy
import time

from pathlib import Path
import subprocess

import numpy as np
import scipy as sp

from rdkit import Chem
import mdtraj

import pickle

## ## 

def save_pickle_(x, name, verbose=True):
    ''' save any python variable, or instance of an object as a pickled file with name
    '''
    with open(name, "wb") as f: pickle.dump(x, f)
    if verbose: print('saved',name)
    else: pass
    
def load_pickle_(name):
    ''' load the pickled file with name back into python
    '''
    with open(name, "rb") as f: x = pickle.load(f) ; return x

## ## 

def reshape_to_molecules_np_(r, n_molecules, n_atoms_in_molecule):
    '''
    Output: (m, n_mol, n_atoms_mol, 3) array 
    '''
    n_frames = r.shape[0]
    return r.reshape([n_frames, n_molecules, n_atoms_in_molecule, 3])

def reshape_to_atoms_np_(r, n_molecules, n_atoms_in_molecule):
    '''
    Output: (m, n_mol * n_atoms_mol, 3) = (m,N,3) array 
    '''
    n_frames = r.shape[0]
    return r.reshape([n_frames, n_molecules*n_atoms_in_molecule, 3])
    
def reshape_to_flat_np_(r, n_molecules, n_atoms_in_molecule):
    '''
    Output: (m, n_mol * n_atoms_mol * 3) array 
    '''
    n_frames = r.shape[0]
    return r.reshape([n_frames, n_molecules*n_atoms_in_molecule*3])

## ## 

cumulative_average_ = lambda x,axis=None : np.cumsum(x,axis=axis) / np.cumsum(np.ones_like(x),axis=axis)

sta_array_ = lambda x : (x-x.min())/(x.max()-x.min())

cdist_ = sp.spatial.distance.cdist

def half_way_(a,c):
    '''
    output: number between a and c
    '''
    ac = sorted([a,c])
    b = min(ac) + 0.5*(max(ac) - min(ac))
    return b

def take_random_(x, m=20000):
    '''
    output: uniformly takenn random m values from first axis of x (len(x) >= m)
    '''
    return x[np.random.choice(x.shape[0],min([m,x.shape[0]]),replace=False)]

def joint_grid_from_marginal_grids_(*marginal_grids, flatten_output=True):
    
    ''' like np.meshgrid but easier to use 
    Inputs:
        *marginal_grids : more than one flat arrays, these are usually grids made by np.linspace.
            dim = number of input grids
        flatten_output : bool affecting shape of the output array
    Outputs:
        if flatten_output:
            joint_grid : (N,dim) ; N = bins[1]*...*bins[dim]
        else:
            joint_grid : (dim, bins[1], ..., bins[dim])
    '''

    list_marginal_grids = list(marginal_grids)
    letters = 'jklmnopqrst'
    dim = len(list_marginal_grids)
    bins = [len(x) for x in list_marginal_grids]

    Xs = []
    string_input = 'io,'
    string_output = 'oi'
    for i in range(dim):
        X = np.ones([bins[i],dim])
        X[:,i] = np.array(list_marginal_grids[i])
        Xs.append(X)
        if i > 0:
            string_input += letters[i]+'o,'
            string_output += letters[i]
        else: pass

    string = string_input[:-1]+'->'+string_output #; print(string)
    
    joint_grid = np.einsum(string,*Xs)

    if flatten_output:
        joint_grid = joint_grid.T.reshape(-1, dim)
    else: pass

    return joint_grid

def tidy_crystal_xyz_(r, b, n_atoms_mol, ind_rO, batch_size=1000):
    ''' makes molecules not jump in Cartesian space
    !! does not work in unstable systems such as very small cells

    In any case: outputs are expected to evaluate to the same energy as the inputs.

    molecules already whole (true in openmm trajectories)
    steps 1 and 2 below deal only with the general positions of molecules in the lattice (nojump)

    ind_rO : index of any atom in a molecule that has slow dynamics relative to the cell

    '''
    def check_shape_(x):
        x = np.array(x)
        shape = len(x.shape)
        assert shape in [2,3]
        if len(x.shape) == 3: pass
        else: x = x[np.newaxis,...]
        return x

    r = check_shape_(r)
    n_frames = r.shape[0]
    batch_size = min([batch_size, n_frames])
    
    if len(b.shape) == 2: b = np.array([b]*n_frames)
    else: assert b.shape[0] == n_frames
    def wrap_points_(R, box):
        # R   : (... 3), shaped as molecules
        # box : (...,3, 3) # rows
        st = 'oabi,oij->oabj'
        return np.einsum(st, np.mod(np.einsum(st, R, np.linalg.inv(box)), 1.0), box)
    
    N = r.shape[1]
    n_mol = N // n_atoms_mol
    assert n_mol == N / n_atoms_mol
    '''
    # step 1 : put atoms with index rO into box (and bring whole molecule with it)
    '''
    r = reshape_to_molecules_np_(r, n_atoms_in_molecule=n_atoms_mol, n_molecules=n_mol)
    for i in range(n_frames//batch_size):
        _from = i*batch_size
        _to = (i+1)*batch_size
        rO = r[_from:_to,:,ind_rO:ind_rO+1]
        r[_from:_to] = r[_from:_to] - rO + wrap_points_(rO,b[_from:_to])

    if n_frames - _to > 0:
        _from = _to
        rO = r[_from:,:,ind_rO:ind_rO+1]
        r[_from:] = r[_from:] - rO + wrap_points_(rO,b[_from:])
    else: pass
    '''
    # step 2: bring any atoms with index rO that are still jumping to pre-jump position (and bring whole molecule with it)
    using method copied from:
    https://github.com/MDAnalysis/mdanalysis/blob/develop/package/MDAnalysis/transformations/nojump.py
    this should give lattice looking like the first frame throughout a crystaline trajectory
    '''
    def dot_(Ri, mat):
        st = 'abi,ij->abj'
        return np.einsum(st, Ri, mat)

    rO = np.array(r[:,:,ind_rO:ind_rO+1])
    b_inv = np.linalg.inv(b)
    
    rO_revised = np.zeros_like(rO)
    rO_revised[0] = rO[0]
    rO_0 = dot_(rO[0], b_inv[0])
    for i in range(1,n_frames):
        rO_1 = dot_(rO[i], b_inv[i])
        rO_1 -= np.round( rO_1 - rO_0 )
        rO_revised[i] = dot_(rO_1, b[i])

    r = r - rO + rO_revised
    # if remove_COM:
    r -= r[:,:,ind_rO:ind_rO+1].mean(1, keepdims=True)
    # else: pass
    r = reshape_to_atoms_np_(r, n_atoms_in_molecule=n_atoms_mol, n_molecules=n_mol)
    return r

## ## 
