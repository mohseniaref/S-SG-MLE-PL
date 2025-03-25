import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.estimation import extractMLEfunc, MLE_PL, SCM, tyler_estimator_covariance
from src.optimization import S_SG_MLE_PL_BCD
from src.generation import sampledistributionchoice

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

def oneMonteCarlo(p, n, trueCov, sampledist, rank, argsMLEPL, number_of_trials, tol):
    """ A function that does one Monte Carlo and calculates the different estimated phases 
        Inputs : 
        --------
        * p : number of data (images)
        * n : number of pixels
        * trueCov : the true covariance matrix
        * sampledist : if Gaussian distribution => 'Gaussian', if Sca led Gaussian distribution => 'ScaledGaussian', nu (the parameter for the gamma distribution of tau)
        * rank : integer that represents the rank of the structure of the covariance matrix
        * argsMLEPL : maximum number of iteration of BCD, MM and the phase correction choice (3 or 4)
        * number_of_trials : number of Monte Carlo simulations (will be used later)
        Outputs : 
        ---------
        * new_deltaphase_recursif, new_deltaphase, delta2pinsar : 3 vectors of size p representing the estimated phase differences recursively and non recursively and according to the 2p-InSAR approach
        * C_tilde, S : 2 arrays representing the estimated C_tilde, SCM 
    """
    np.random.seed(number_of_trials)
    delta2pinsar = np.zeros(p+1)
    (iter_PL, iter_max_BCD, iter_max_MM, phasecorrectionchoice) = argsMLEPL
    argsMLE = (iter_max_BCD, iter_max_MM, phasecorrectionchoice)
    X = sampledistributionchoice(trueCov, p+1, n, sampledist)
    X = np.asarray(X, dtype=np.complex128)
    X_past = X[0:p, :]
    x_newdata = X[p, :]
    C, _ = tyler_estimator_covariance(X_past, 0.001, 100)
    C = np.asarray(C, dtype=np.complex128)
    _, _, diag_w_past, _ = extractMLEfunc(X_past, 
                               'ScaledGaussian',
                               'Cor-Arg', 
                               p, 
                               argsMLE)
    # MLE-PL (offline)
    _, new_deltaphase, _, _ = extractMLEfunc(X, 
                                       'ScaledGaussian', 
                                       'Cor-Arg', 
                                       rank, 
                                       argsMLE)
    # classic PL (offline)
    _, new_deltaphase_classic_PL, _ , _= extractMLEfunc(X, 
                                                  'ScaledGaussian', 
                                                  'Mod-Arg', 
                                                  rank, 
                                                  argsMLE)

    # S-G-MLE-PL
    C_tilde_structured, new_phase_sequential = S_SG_MLE_PL_BCD(X, 
                                                              X_past, 
                                                              x_newdata,
                                                              C, 
                                                              diag_w_past, 
                                                              iter_max_BCD, 
                                                              phasecorrectionchoice, 
                                                              tol=0.001)
    # 2p-InSAR
    for kk in np.arange(1,p+1):
        delta2pinsar[kk] = (-np.angle(np.dot(X[0,:],np.transpose(np.conj(X[kk,:])))))

    return (new_deltaphase, 
            new_deltaphase_classic_PL, 
            delta2pinsar, 
            new_phase_sequential, 
            C_tilde_structured)

def parallel_Monte_Carlo(p, n, trueCov, sampledist, rank, argsMLEPL, number_of_trials, number_of_threads, tol, Multi):
    """ A function that does several Monte Carlo and calculates the different estimated phases 
        Inputs : 
        --------
        * p : number of data (images)
        * n : number of pixels
        * trueCov : the true covariance matrix
        * sampledist : if Gaussian distribution => 'Gaussian', if Scaled Gaussian distribution => 'ScaledGaussian', nu (the parameter for the gamma distribution of tau)
        * rank : integer that represents the rank of the structure of the covariance matrix
        * argsMLEPL : maximum number of iteration of BCD, MM and the phase correction choice (3 or 4)
        * number_of_trials : number of Monte Carlo simulations (will be used later)
        * number_of_threads : number of threads
        * Multi : True/False
        Outputs : 
        ---------
        * delta_phases_recursif, delta_phases, delta_2p_insar : 3 lists of size number_of_trials, each element has a size p representing the estimated phase differences recursively and non recursively and according to the 2p-InSAR approach
        * C_tilde_entire, S, C_tilde_structured : 3 lists of size number_of_trials, each element is an array of size (p+1, p+1) representing the estimated C_tilde, SCM 
    """
    if Multi:
        delta_phases = []
        delta_phases_classic_PL = []
        delta_2p_insar = []
        new_phase_sequential = []
        C_tilde_sequential = []                                                    
        parallel_results = Parallel(n_jobs=number_of_threads)(delayed(oneMonteCarlo)(p, n, trueCov, sampledist, rank, argsMLEPL, iMC, tol) for iMC in tqdm(range(number_of_trials)))
        # paralle_results : tuple
        for i in range(number_of_trials):
            parallel_results_delta = np.array(parallel_results[i][0:3]) 
            delta_phases.append(parallel_results_delta[0])
            delta_phases_classic_PL.append(parallel_results_delta[1])
            delta_2p_insar.append(parallel_results_delta[2])
            new_phase_sequential.append(np.array(parallel_results[i][3]))
            C_tilde_sequential.append(np.array(parallel_results[i][4]))
        return (delta_phases, 
                delta_phases_classic_PL, 
                delta_2p_insar, 
                new_phase_sequential, 
                C_tilde_sequential)
    else:
        results = [] # results container
        delta_phases = []
        delta_phases_classic_PL = []
        delta_2p_insar = []
        new_phase_sequential = []
        C_tilde_sequential = []
        for iMC in tqdm(range(number_of_trials)):
            results.append(oneMonteCarlo(p, n, trueCov, sampledist, rank, argsMLEPL, iMC, tol))
            delta_phases.append(results[iMC][0])
            delta_phases_classic_PL.append(results[iMC][1])
            delta_2p_insar.append(results[iMC][2])
            new_phase_sequential.append(results[iMC][3])
            C_tilde_sequential.append(results[iMC][4])
        return (delta_phases, 
                delta_phases_classic_PL, 
                delta_2p_insar, 
                new_phase_sequential, 
                C_tilde_sequential)
